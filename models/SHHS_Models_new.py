import torch
import torch.nn as nn
import collections
import math
from torch.autograd.variable import Variable
import einops
from torch import Tensor

from typing import Optional, Any
import  copy
import numpy as np
# from models.custom_layers.eeg_encoders import *

class EEG_SLEEP_BLIP_GM_MultiMode(nn.Module):

    def __init__(self, encs=[None], args=None):
        """
        :param encoder_filters_small, encoder_filters_big: filters of CNN output, quantized by 20
        :param encs_small, encs_big:
        """
        super().__init__()

        d_model =  args.dmodel#64*8
        fc_inner = args.fc_inner
        num_classes = args.num_classes
        dropout = args.dropout if "dropout" in args else 0.1
        self.shared_pred = args.shared_pred if "shared_pred" in args else False
        self.shared_nonbatched_pred = args.shared_nonbatched_pred if "shared_nonbatched_pred" in args else False
        self.skip_predictions = args.skip_predictions if "skip_predictions" in args else False
        # self.clean_train = args.clean_train if "clean_train" in args else False
        self.args = args
        self.num_encoders = 0

        for i, enc in enumerate(encs):
            if enc != None:
                setattr(self, "enc_{}".format(i), enc)
                self.num_encoders +=1

        self.fc_out = nn.Sequential(
                        nn.Linear(d_model, fc_inner),
                        nn.ReLU(),
                        nn.Dropout(dropout),
                        nn.Linear(fc_inner, fc_inner),
                        nn.ReLU(),
                        nn.Dropout(dropout),
                        nn.Linear(fc_inner, num_classes)
                    )

        if not self.shared_pred and not self.shared_nonbatched_pred :
            self.fc_out_eeg = copy.deepcopy(self.fc_out)
            self.fc_out_eog = copy.deepcopy(self.fc_out)

    def forward(self, x, **kwargs):

        return_matches = kwargs["return_matches"] if "return_matches" in kwargs else False

        skip_modality_dict = None
        if "skip_modality" in kwargs and type(kwargs["skip_modality"]) is dict:
            skip_modality_dict = kwargs["skip_modality"]
            # if not self.clean_train:
            #     kwargs["skip_modality"] = "full"

        for i in range(self.num_encoders):
            enc = getattr(self, "enc_{}".format(i))
            x = enc(x, **kwargs)

        output_features = x["output_features"]

        output={"features": output_features}
        if not self.skip_predictions:
            if self.shared_pred:
                merged_features, pred_length = self._concat_preds(output_features)
                merged_features = self.fc_out(merged_features)
                output = self._sep_preds(merged_features=merged_features, output_lens=pred_length)
                output = self._calc_indiv_preds_targemasks(output=output, skip_modality_dict=skip_modality_dict)
                # output = self._calc_skipped(skip_modality_dict=skip_modality_dict, output=output)
            elif self.shared_nonbatched_pred:
                output = {"preds":{}, "features": output_features}
                if "combined" in x["output_features"]:
                    output["preds"]["combined"] = self.fc_out(x["output_features"]["combined"].flatten(start_dim=0, end_dim=1).flatten(start_dim=1))
                if "eeg" in x["output_features"]:
                    output["preds"]["eeg"] = self.fc_out(x["output_features"]["eeg"].flatten(start_dim=0, end_dim=1).flatten(start_dim=1))
                if "eog" in x["output_features"]:
                    output["preds"]["eog"] = self.fc_out(x["output_features"]["eog"].flatten(start_dim=0, end_dim=1).flatten(start_dim=1))
                # output = self._calc_skipped(skip_modality_dict=skip_modality_dict, output=output)
            else:
                output = {"preds":{},"features": output_features}
                if "combined" in x["output_features"]:
                    output["preds"]["combined"] = self.fc_out(x["output_features"]["combined"].flatten(start_dim=0, end_dim=1).flatten(start_dim=1))
                if "eeg" in x["output_features"]:
                    output["preds"]["eeg"] = self.fc_out_eeg(x["output_features"]["eeg"].flatten(start_dim=0, end_dim=1).flatten(start_dim=1))
                if "eog" in x["output_features"]:
                    output["preds"]["eog"] = self.fc_out_eog(x["output_features"]["eog"].flatten(start_dim=0, end_dim=1).flatten(start_dim=1))
                # output = self._calc_skipped(skip_modality_dict=skip_modality_dict, output=output)

        if type(skip_modality_dict) is dict:
            output["incomplete_idx"] = dict( eeg = ~skip_modality_dict["stft_eeg"].bool(),
                                             eog = ~skip_modality_dict["stft_eog"].bool(),
                                             combined = ~torch.logical_or(skip_modality_dict["stft_eeg"].bool(), skip_modality_dict["stft_eog"].bool()))

        if return_matches:
            output["matches"] = x["matches"] if "matches" in x and x["matches"] is not None else {"stft_eeg": None, "stft_eog": None}

        return output

    def _concat_preds(self, output_features):
        output_lens = {}
        allpred_features = []
        for pred in output_features:
            if len(output_features[pred].shape)>2:
                output_features[pred] = output_features[pred].flatten(start_dim=0, end_dim=1).flatten(start_dim=1)
            output_lens[pred] = len(output_features[pred])
            allpred_features.append(output_features[pred])
        output = torch.cat(allpred_features, dim=0)
        return output, output_lens

    def  _sep_preds(self, merged_features, output_lens):
        output = {"preds": {}}
        count = 0
        for pred in output_lens:
            output["preds"][pred] = merged_features[count:count+output_lens[pred]]
            count+= output_lens[pred]
        output["preds"] = collections.OrderedDict(sorted(output["preds"].items()))
        return output


    def  _calc_indiv_preds_targemasks(self, output, skip_modality_dict, **kwargs):

        if skip_modality_dict and "stft_eeg" in skip_modality_dict and "stft_eog" in skip_modality_dict:
            skip_modality_dict["combined"] = skip_modality_dict["stft_eeg"] * 1 + skip_modality_dict["stft_eog"] * 2

        target_mask = {}
        for pred in output["preds"]:
            if skip_modality_dict and "stft_eeg" in skip_modality_dict:
                for i in skip_modality_dict:
                    if pred in i:
                        this_target_mask = skip_modality_dict[i]
                target_mask[pred] = this_target_mask.flatten()
        #This target mask is used to sample the training labels that we want to backpropagate with in case we already know some broken data.
        output["target_mask"] = target_mask
        return output

    def _calc_skipped(self, output, skip_modality_dict):
        if not self.training:
            if skip_modality_dict and "stft_eeg" in skip_modality_dict:
                skip_eeg = skip_modality_dict["stft_eeg"].flatten()
                skip_eog = skip_modality_dict["stft_eog"].flatten()
                skip_all = skip_eeg + skip_eog*2
                output["preds"]["skipped"] = copy.deepcopy(output["preds"]["combined"])
                output["preds"]["skipped"][skip_all==1] = copy.deepcopy(output["preds"]["eog"])[skip_all==1]
                output["preds"]["skipped"][skip_all==2] = copy.deepcopy(output["preds"]["eeg"])[skip_all==2]

        return output
class EEG_SLEEP_BLIP_GM_MultiMode_Concat(nn.Module):

    def __init__(self, encs=[None], args=None):
        """
        :param encoder_filters_small, encoder_filters_big: filters of CNN output, quantized by 20
        :param encs_small, encs_big:
        """
        super().__init__()

        d_model =  args.dmodel#64*8
        fc_inner = args.fc_inner
        num_classes = args.num_classes
        dropout = args.dropout if "dropout" in args else 0.1
        self.shared_pred = args.shared_pred if "shared_pred" in args else False
        self.shared_nonbatched_pred = args.shared_nonbatched_pred if "shared_nonbatched_pred" in args else False
        self.skip_predictions = args.skip_predictions if "skip_predictions" in args else False
        # self.clean_train = args.clean_train if "clean_train" in args else False
        self.args = args
        self.num_encoders = 0

        for i, enc in enumerate(encs):
            if enc != None:
                setattr(self, "enc_{}".format(i), enc)
                self.num_encoders +=1

        self.fc_out = nn.Sequential(
                        nn.Linear(2*d_model, fc_inner),
                        nn.ReLU(),
                        nn.Dropout(dropout),
                        nn.Linear(fc_inner, fc_inner),
                        nn.ReLU(),
                        nn.Dropout(dropout),
                        nn.Linear(fc_inner, num_classes)
                    )

        self.fc_out_eeg = nn.Sequential(
                        nn.Linear(d_model, fc_inner),
                        nn.ReLU(),
                        nn.Dropout(dropout),
                        nn.Linear(fc_inner, fc_inner),
                        nn.ReLU(),
                        nn.Dropout(dropout),
                        nn.Linear(fc_inner, num_classes)
                    )

        self.fc_out_eog = nn.Sequential(
                        nn.Linear(d_model, fc_inner),
                        nn.ReLU(),
                        nn.Dropout(dropout),
                        nn.Linear(fc_inner, fc_inner),
                        nn.ReLU(),
                        nn.Dropout(dropout),
                        nn.Linear(fc_inner, num_classes)
                    )

        if self.shared_pred or self.shared_nonbatched_pred :
            del self.fc_out_eeg, self.fc_out_eog

    def forward(self, x, **kwargs):

        return_matches = kwargs["return_matches"] if "return_matches" in kwargs else False

        skip_modality_dict = None
        if "skip_modality" in kwargs and type(kwargs["skip_modality"]) is dict:
            skip_modality_dict = kwargs["skip_modality"]
            # if not self.clean_train:
            #     kwargs["skip_modality"] = "full"

        for i in range(self.num_encoders):
            enc = getattr(self, "enc_{}".format(i))
            x = enc(x, **kwargs)

        output_features = x["output_features"]

        output={}
        if not self.skip_predictions:
            if self.shared_pred:
                merged_features, pred_length = self._concat_preds(output_features)
                merged_features = self.fc_out(merged_features)
                output = self._sep_preds(merged_features=merged_features, output_lens=pred_length)
                output = self._calc_indiv_preds_targemasks(output=output, skip_modality_dict=skip_modality_dict)
                # output = self._calc_skipped(skip_modality_dict=skip_modality_dict, output=output)
            elif self.shared_nonbatched_pred:
                output = {"preds":{}}
                if "combined" in x["output_features"]:
                    output["preds"]["combined"] = self.fc_out(x["output_features"]["combined"].flatten(start_dim=0, end_dim=1).flatten(start_dim=1))
                if "eeg" in x["ou" \
                              "tput_features"]:
                    output["preds"]["eeg"] = self.fc_out(x["output_features"]["eeg"].flatten(start_dim=0, end_dim=1).flatten(start_dim=1))
                if "eog" in x["output_features"]:
                    output["preds"]["eog"] = self.fc_out(x["output_features"]["eog"].flatten(start_dim=0, end_dim=1).flatten(start_dim=1))
                # output = self._calc_skipped(skip_modality_dict=skip_modality_dict, output=output)
            else:
                output = {"preds":{}}
                if "combined" in x["output_features"]:
                    output["preds"]["combined"] = self.fc_out(x["output_features"]["combined"].flatten(start_dim=0, end_dim=1).flatten(start_dim=1))
                if "eeg" in x["output_features"]:
                    output["preds"]["eeg"] = self.fc_out_eeg(x["output_features"]["eeg"].flatten(start_dim=0, end_dim=1).flatten(start_dim=1))
                if "eog" in x["output_features"]:
                    output["preds"]["eog"] = self.fc_out_eog(x["output_features"]["eog"].flatten(start_dim=0, end_dim=1).flatten(start_dim=1))
                # output = self._calc_skipped(skip_modality_dict=skip_modality_dict, output=output)

        if type(skip_modality_dict) is dict:
            output["incomplete_idx"] = dict( eeg = ~skip_modality_dict["stft_eeg"].bool(),
                                             eog = ~skip_modality_dict["stft_eog"].bool(),
                                             combined = ~torch.logical_or(skip_modality_dict["stft_eeg"].bool(), skip_modality_dict["stft_eog"].bool()))

        if return_matches:
            output["matches"] = x["matches"] if "matches" in x and x["matches"] is not None else {"stft_eeg": None, "stft_eog": None}

        return output

    def _concat_preds(self, output_features):
        output_lens = {}
        allpred_features = []
        for pred in output_features:
            if len(output_features[pred].shape)>2:
                output_features[pred] = output_features[pred].flatten(start_dim=0, end_dim=1).flatten(start_dim=1)
            output_lens[pred] = len(output_features[pred])
            allpred_features.append(output_features[pred])
        output = torch.cat(allpred_features, dim=0)
        return output, output_lens

    def  _sep_preds(self, merged_features, output_lens):
        output = {"preds": {}}
        count = 0
        for pred in output_lens:
            output["preds"][pred] = merged_features[count:count+output_lens[pred]]
            count+= output_lens[pred]
        output["preds"] = collections.OrderedDict(sorted(output["preds"].items()))
        return output


    def  _calc_indiv_preds_targemasks(self, output, skip_modality_dict, **kwargs):

        if skip_modality_dict and "stft_eeg" in skip_modality_dict and "stft_eog" in skip_modality_dict:
            skip_modality_dict["combined"] = skip_modality_dict["stft_eeg"] * 1 + skip_modality_dict["stft_eog"] * 2

        target_mask = {}
        for pred in output["preds"]:
            if skip_modality_dict and "stft_eeg" in skip_modality_dict:
                for i in skip_modality_dict:
                    if pred in i:
                        this_target_mask = skip_modality_dict[i]
                target_mask[pred] = this_target_mask.flatten()
        #This target mask is used to sample the training labels that we want to backpropagate with in case we already know some broken data.
        output["target_mask"] = target_mask
        return output

    def _calc_skipped(self, output, skip_modality_dict):
        if not self.training:
            if skip_modality_dict and "stft_eeg" in skip_modality_dict:
                skip_eeg = skip_modality_dict["stft_eeg"].flatten()
                skip_eog = skip_modality_dict["stft_eog"].flatten()
                skip_all = skip_eeg + skip_eog*2
                output["preds"]["skipped"] = copy.deepcopy(output["preds"]["combined"])
                output["preds"]["skipped"][skip_all==1] = copy.deepcopy(output["preds"]["eog"])[skip_all==1]
                output["preds"]["skipped"][skip_all==2] = copy.deepcopy(output["preds"]["eeg"])[skip_all==2]

        return output

class SleepEnc_BLIP_EEG_EOG_shared_free_cliplike(nn.Module):
        def __init__(self, args, encs=[None]):
            """
            :param encoder_filters_small, encoder_filters_big: filters of CNN output, quantized by 20
            :param encs_small, encs_big:
            """
            super().__init__()
            self.args = args

            d_model = args.d_model
            self.pos = args.pos if "pos" in args else True
            self.outer_rep = args.outer_rep if "outer_rep" in args else False
            self.skip_percentile = args.skip_percentile if "skip_percentile" in args else False
            self.mod_token = args.mod_token if "mod_token" in args else False
            self.dropout = args.dropout if "dropout" in args else False
            self.align_inner = args.align_inner if "align_inner" in args else False
            # self.clean_train = args.clean_train if "clean_train" in args else False

            self.disable_mods = dict(stft_eeg=False, stft_eog=False)

            self.inner_tf_eeg = TF_Block_SA_CA_CA(CA_flag=True, **args)
            self.inner_tf_eog = TF_Block_SA_CA_CA(CA_flag=True, **args)

            self.outer_tf_eeg = TF_Block_SA_CA_CA(CA_flag=True, **args)
            self.outer_tf_eog = TF_Block_SA_CA_CA(CA_flag=True, **args)

            self.cls_token_eeg = nn.Parameter(torch.randn(1, 1, 1, 1, 1, d_model), requires_grad=True)
            self.cls_token_eog = nn.Parameter(torch.randn(1, 1, 1, 1, 1, d_model), requires_grad=True)

            self.cls_token_outer_eeg = nn.Parameter(torch.randn(1, 1, 1, 1, 1, d_model), requires_grad=True)
            self.cls_token_outer_eog = nn.Parameter(torch.randn(1, 1, 1, 1, 1, d_model), requires_grad=True)

            # self.eeg_proj_head = nn.Parameter(torch.randn(d_model, d_model))
            # self.eog_proj_head = nn.Parameter(torch.randn(d_model, d_model))
            #
            # if self.eog_proj_head is not None:
            #     nn.init.normal_(self.eeg_proj_head, std=d_model ** -0.5)
            #     nn.init.normal_(self.eog_proj_head, std=d_model ** -0.5)

            if self.pos and self.pos == "trained":
                self.pos_emb_eeg = pos_embedding(max_pos=200, dim=d_model)
                self.pos_emb_eog = pos_embedding(max_pos=200, dim=d_model)
            elif self.pos and self.pos == "sinusoidal":
                self.pos_emb_eeg = pos_sinusoidal(d_model, max_pos=200)
                self.pos_emb_eog = pos_sinusoidal(d_model, max_pos=200)

            if self.mod_token:
                self.modtype_token = modtype_embedding(num_modalities=2, dim=d_model)

            #This part is new, it doesnt belong to the paper
            self.norm_uni_eeg = nn.InstanceNorm3d(128, momentum=0)
            self.norm_uni_eog = nn.InstanceNorm3d(128, momentum=0)
            self.norm_multi_eeg = nn.InstanceNorm2d(128, momentum=0)
            self.norm_multi_eog = nn.InstanceNorm2d(128, momentum=0)

        def forward(self, x, skip_modality="full", **kwargs):
            xeeg = None
            #TODO: There was here a and skip_modality["stft_eeg"].sum()<=len(skip_modality["stft_eeg"]), which I dont understand
            if skip_modality!="eeg":

                xeeg = x["stft_eeg"][:, :, :, :, 1:, :].float()  # mat
                xeeg = einops.rearrange(xeeg, "b outer mod ch f inner -> b outer inner mod ch f")
                if type(skip_modality) == dict and "stft_eeg" in skip_modality:
                    xeeg_shape = xeeg.shape
                    xeeg = xeeg[~skip_modality["stft_eeg"].bool()]
                    xeeg = einops.rearrange(xeeg, "(b outer) i m c f -> b outer i m c f", outer=xeeg_shape[1], b=int(xeeg.shape[0]/xeeg_shape[1]))
                if self.mod_token:
                    xeeg = self.modtype_token(data=xeeg, mod_num=0)
                if self.pos:
                    xeeg = self.pos_emb_eeg.forward_inner(xeeg)

                cls_token_eeg = self.cls_token_eeg.repeat(xeeg.shape[0], xeeg.shape[1], 1, 1, xeeg.shape[3], 1).to(xeeg.device)
                xeeg = torch.cat([cls_token_eeg, xeeg], dim=2)

            xeog = None
            if skip_modality != "eog":
                xeog = x["stft_eog"][:, :, :, :, 1:, :].float()  # mat
                xeog = einops.rearrange(xeog, "b outer mod ch f inner -> b outer inner mod ch f")
                if type(skip_modality) == dict and "stft_eog" in skip_modality:
                    xeog_shape = xeog.shape
                    xeog = xeog[~skip_modality["stft_eog"].bool()]
                    xeog = einops.rearrange(xeog, "(b outer) i m c f -> b outer i m c f", outer=xeog_shape[1],
                                            b=int(xeog.shape[0] / xeog_shape[1]))
                if self.mod_token:
                    xeog = self.modtype_token(data=xeog, mod_num=1)
                if self.pos:
                    xeog = self.pos_emb_eog.forward_inner(xeog)
                cls_token_eog = self.cls_token_eog.repeat(xeog.shape[0], xeog.shape[1], 1, 1, xeog.shape[3], 1).to(xeog.device)
                xeog = torch.cat([cls_token_eog, xeog], dim=2)

            output = {"output_features": {}}
            output = self.forward_sole(xeeg=xeeg, xeog=xeog, output=output, skip_modality=skip_modality, align_inner = self.align_inner, **kwargs)
            if xeeg is not None and xeog is not None and "inner_eeg" in output["output_features"] and "inner_eog" in output["output_features"]:
                output = self.forward_common(xeeg=xeeg, xeog=xeog, output=output, skip_modality=skip_modality, **kwargs)

            output["output_features"].pop("inner_eeg", None)
            output["output_features"].pop("inner_eog", None)

            return output

        def _keep_common(self, x, common_idx, skip_idx):

            if len(x.shape)==6:
                output = x[einops.rearrange(~common_idx[~skip_idx.bool()], "(b outer) -> b outer", b=x.shape[0],
                                            outer=x.shape[1])]
                output = einops.rearrange(output, "(b outer) i m c f -> b outer i m c f",
                                                 outer=common_idx.shape[1],
                                                 b=int(output.shape[0] / common_idx.shape[1]))
            elif len(x.shape)==3:
                output = x[einops.rearrange(~common_idx[~skip_idx.bool()], "(b outer) -> b outer", b=x.shape[0],
                                            outer=x.shape[1])]
                output = einops.rearrange(output, "(b outer) f -> b outer f",
                                                 outer=common_idx.shape[1],
                                                 b=int(output.shape[0] / common_idx.shape[1]))
            elif len(x.shape)==2:
                #This assumes that batch dim has been squeezed
                x = x.unsqueeze(dim=1)
                output = x[einops.rearrange(~common_idx[~skip_idx.bool()], "(b outer) -> b outer", b=x.shape[0],
                                            outer=x.shape[1])]
                output = einops.rearrange(output, "(b outer) f -> b outer f",
                                                 outer=common_idx.shape[1],
                                                 b=int(output.shape[0] / common_idx.shape[1]))

            return  output
        def forward_common(self, xeeg, xeog, output, **kwargs):

            skip_modality = kwargs["skip_modality"]

            xeeg_common_i = output["output_features"]["inner_eeg"]
            xeog_common_i = output["output_features"]["inner_eog"]

            if xeeg_common_i.shape[1] > 1 and  xeeg_common_i.shape[1] > 1:
                xeeg_common_outer = output["output_features"]["eeg"]
                xeog_common_outer = output["output_features"]["eog"]

            if type(skip_modality) == dict and "stft_eeg" in skip_modality and "stft_eog" in skip_modality:
                common_kept_idx = torch.logical_or(skip_modality["stft_eeg"].bool(), skip_modality["stft_eog"].bool())
                # xeeg_common_i = self._keep_common(xeeg_common_i, output["common_kept_idx"], skip_modality["stft_eeg"])
                # xeog_common_i = self._keep_common(xeog_common_i, output["common_kept_idx"], skip_modality["stft_eog"])
                if "xeeg_common_outer" in locals() and "xeog_common_outer" in locals():
                    xeeg_common_outer = self._keep_common(xeeg_common_outer, common_kept_idx, skip_modality["stft_eeg"])
                    xeog_common_outer = self._keep_common(xeog_common_outer, common_kept_idx, skip_modality["stft_eog"])
                xeeg = self._keep_common(xeeg, common_kept_idx, skip_modality["stft_eeg"])
                xeog = self._keep_common(xeog, common_kept_idx, skip_modality["stft_eog"])

            #Extra after paper: remove if needed
            xeeg_common_i = self.norm_uni_eeg(xeeg_common_i.squeeze()).unsqueeze(dim=-2).unsqueeze(dim=-2)
            xeog_common_i = self.norm_uni_eog(xeog_common_i.squeeze()).unsqueeze(dim=-2).unsqueeze(dim=-2)

            xeeg_ca_common = self.inner_tf_eeg.forward_inner(xeeg, xeog_common_i)[:, :, :1]
            xeog_ca_common = self.inner_tf_eog.forward_inner(xeog, xeeg_common_i)[:, :, :1]

            if "xeeg_common_outer" in locals() and "xeog_common_outer" in locals():

                if self.pos:
                    xeeg_ca_common = self.pos_emb_eeg.forward_outer(xeeg_ca_common)
                    xeog_ca_common = self.pos_emb_eog.forward_outer(xeog_ca_common)

                xeeg_ca_common_outer = self.outer_tf_eeg.forward_outer(xeeg_ca_common, xeog_common_outer)
                xeog_ca_common_outer = self.outer_tf_eog.forward_outer(xeog_ca_common, xeeg_common_outer)

                # Extra after paper: remove if needed
                xeeg_ca_common_outer = self.norm_multi_eeg(xeeg_ca_common_outer.squeeze()).unsqueeze(dim=-2).unsqueeze(dim=-2).unsqueeze(dim=-2)
                xeog_ca_common_outer = self.norm_multi_eog(xeog_ca_common_outer.squeeze()).unsqueeze(dim=-2).unsqueeze(dim=-2).unsqueeze(dim=-2)

                x_common = xeeg_ca_common_outer + xeog_ca_common_outer
            else:

                # Extra after paper: remove if needed
                # xeeg_ca_common_outer = self.norm_multi_eeg(xeeg_ca_common_outer)
                # xeeg_ca_common = self.norm_multi_eog(xeeg_ca_common)

                x_common = xeeg_ca_common + xeog_ca_common #This was wrong

            output["output_features"]["combined"] = x_common

            return output

        def forward_sole(self, xeeg, xeog, output, skip_modality, return_matches=False, **kwargs):

            if xeeg is not None and skip_modality!="eeg" and xeeg.shape[0]>0:
                # and type(skip_modality)==dict and skip_modality["stft_eeg"].sum()<len(skip_modality["stft_eeg"]):
                xeeg_sole = self.inner_tf_eeg.forward_inner(xeeg)
                output["output_features"]["inner_eeg"] = xeeg_sole
                xeeg_cls_sole = xeeg_sole[:, :, :1]
                if xeeg_cls_sole.shape[1]>1:
                    xeeg_outer_sole = self.outer_tf_eeg.forward_outer(xeeg_cls_sole, use_rpos=True)
                    output["output_features"]["eeg"] = xeeg_outer_sole
                    if "align_inner" in kwargs and kwargs["align_inner"]:
                        xeeg_match_sq = xeeg_cls_sole.squeeze()
                    else:
                        xeeg_match_sq = xeeg_outer_sole.squeeze()
                else:
                    output["output_features"]["eeg"] = xeeg_cls_sole

            if xeog is not None and skip_modality != "eog" and xeog.shape[0]>0:
                # and type(skip_modality)==dict and skip_modality["stft_eog"].sum()<len(skip_modality["stft_eog"]):

                xeog_sole = self.inner_tf_eog.forward_inner(xeog)
                output["output_features"]["inner_eog"] = xeog_sole
                xeog_cls_sole = xeog_sole[:, :, :1]
                if xeog_cls_sole.shape[1]>1:
                    xeog_outer_sole = self.outer_tf_eog.forward_outer(xeog_cls_sole, use_rpos=True)
                    output["output_features"]["eog"] = xeog_outer_sole
                    if "align_inner" in kwargs and kwargs["align_inner"]:
                        xeog_match_sq = xeog_cls_sole.squeeze()
                    else:
                        xeog_match_sq = xeog_outer_sole.squeeze()
                else:
                    output["output_features"]["eog"] = xeog_cls_sole


            x_match = None
            if skip_modality != "eeg" and skip_modality != "eog" and return_matches and "xeeg_match_sq" in locals() and "xeog_match_sq" in locals():
                if type(skip_modality) == dict and "stft_eeg" in skip_modality and "stft_eog" in skip_modality:
                    common_kept_idx = torch.logical_or(skip_modality["stft_eeg"].bool(), skip_modality["stft_eog"].bool())
                    xeeg_match_sq = self._keep_common(xeeg_match_sq, common_kept_idx, skip_modality["stft_eeg"])
                    xeog_match_sq = self._keep_common(xeog_match_sq, common_kept_idx, skip_modality["stft_eog"])
                    output["output_features"]["inner_eeg"] = self._keep_common(output["output_features"]["inner_eeg"], common_kept_idx, skip_modality["stft_eeg"])
                    output["output_features"]["inner_eog"] = self._keep_common(output["output_features"]["inner_eog"], common_kept_idx, skip_modality["stft_eog"])

                if len(xeeg_match_sq.shape) == 3 and len(xeog_match_sq.shape) == 3 and xeeg_match_sq.shape[0]>0 and  xeog_match_sq.shape[0]>0:

                    # xeeg_sa_o = torch.einsum('b o f , f p -> b o p', xeeg_sa_o, self.eeg_proj_head)
                    # xeog_sa_o = torch.einsum('b o f , f p -> b o p', xeog_sa_o, self.eog_proj_head)

                    # # cosine similarity as logits
                    # logit_scale = self.logit_scale.exp()

                    if  'big_al' in self.args and self.args['big_al']:
                        xeeg_outer_sole_sq = einops.rearrange(xeeg_match_sq, "b o f -> (b o) f")
                        xeog_outer_sole_sq = einops.rearrange(xeog_match_sq, 'b o f -> (b o) f')

                        # normalized features
                        xeeg_outer_sole_sq_norm = xeeg_outer_sole_sq / xeeg_outer_sole_sq.norm(dim=1, keepdim=True)
                        xeog_outer_sole_sq_norm = xeog_outer_sole_sq / xeog_outer_sole_sq.norm(dim=1, keepdim=True)

                        x_match_eeg = torch.matmul(xeeg_outer_sole_sq_norm,xeog_outer_sole_sq_norm.t())
                        x_match_eog = x_match_eeg.permute(1, 0)

                    else:
                        # normalized features
                        xeeg_outer_sole_sq_norm = xeeg_match_sq / xeeg_match_sq.norm(dim=1, keepdim=True)
                        xeog_outer_sole_sq_norm = xeog_match_sq / xeog_match_sq.norm(dim=1, keepdim=True)

                        x_match_eeg = torch.einsum('b o f , b m f -> b o m', xeeg_outer_sole_sq_norm, xeog_outer_sole_sq_norm)
                        x_match_eog = x_match_eeg.permute(0, 2, 1)

                    x_match = {"stft_eeg": x_match_eeg, "stft_eog": x_match_eog}
                    # else:
                    #     x_match = torch.Tensor([0]).to(xeeg.device)
                elif len(xeeg_match_sq.shape) == 2 and len(xeog_match_sq.shape) == 2:

                    if  'big_al' in self.args and self.args['big_al']:
                        xeeg_outer_sole_sq = einops.rearrange(xeeg_match_sq, "b o f -> (b o) f")
                        xeog_outer_sole_sq = einops.rearrange(xeog_match_sq, 'b o f -> (b o) f')

                        # normalized features
                        xeeg_outer_sole_sq_norm = xeeg_outer_sole_sq / xeeg_outer_sole_sq.norm(dim=1, keepdim=True)
                        xeog_outer_sole_sq_norm = xeog_outer_sole_sq / xeog_outer_sole_sq.norm(dim=1, keepdim=True)

                        x_match_eeg = torch.matmul(xeeg_outer_sole_sq_norm,xeog_outer_sole_sq_norm.t())
                        x_match_eog = x_match_eeg.permute(1, 0)

                    else:
                        # normalized features
                        xeeg_outer_sole_sq_norm = xeeg_match_sq / xeeg_match_sq.norm(dim=1, keepdim=True)
                        xeog_outer_sole_sq_norm = xeog_match_sq / xeog_match_sq.norm(dim=1, keepdim=True)

                        x_match_eeg = torch.einsum('o f , m f -> o m', xeeg_outer_sole_sq_norm, xeog_outer_sole_sq_norm).unsqueeze(dim=0)
                        x_match_eog = x_match_eeg.permute(0, 2, 1)

                    x_match = {"stft_eeg": x_match_eeg, "stft_eog": x_match_eog}

                else:
                    x_match = {"stft_eeg": None, "stft_eog": None}
                    # x_match = torch.Tensor([0]).to(xeeg.device)

            output["matches"] = x_match

            return output
class SleepEnc_BLIP_EEG_EOG_shared_free_cliplike_Concat(nn.Module):
        def __init__(self, args, encs=[None]):
            """
            :param encoder_filters_small, encoder_filters_big: filters of CNN output, quantized by 20
            :param encs_small, encs_big:
            """
            super().__init__()
            self.args = args

            d_model = args.d_model
            self.pos = args.pos if "pos" in args else True
            self.outer_rep = args.outer_rep if "outer_rep" in args else False
            self.skip_percentile = args.skip_percentile if "skip_percentile" in args else False
            self.mod_token = args.mod_token if "mod_token" in args else False
            self.dropout = args.dropout if "dropout" in args else False
            self.align_inner = args.align_inner if "align_inner" in args else False
            # self.clean_train = args.clean_train if "clean_train" in args else False

            self.disable_mods = dict(stft_eeg=False, stft_eog=False)

            self.inner_tf_eeg = TF_Block_SA_CA_CA(CA_flag=True, **args)
            self.inner_tf_eog = TF_Block_SA_CA_CA(CA_flag=True, **args)

            self.outer_tf_eeg = TF_Block_SA_CA_CA(CA_flag=True, **args)
            self.outer_tf_eog = TF_Block_SA_CA_CA(CA_flag=True, **args)

            self.cls_token_eeg = nn.Parameter(torch.randn(1, 1, 1, 1, 1, d_model), requires_grad=True)
            self.cls_token_eog = nn.Parameter(torch.randn(1, 1, 1, 1, 1, d_model), requires_grad=True)

            self.cls_token_outer_eeg = nn.Parameter(torch.randn(1, 1, 1, 1, 1, d_model), requires_grad=True)
            self.cls_token_outer_eog = nn.Parameter(torch.randn(1, 1, 1, 1, 1, d_model), requires_grad=True)

            # self.eeg_proj_head = nn.Parameter(torch.randn(d_model, d_model))
            # self.eog_proj_head = nn.Parameter(torch.randn(d_model, d_model))
            #
            # if self.eog_proj_head is not None:
            #     nn.init.normal_(self.eeg_proj_head, std=d_model ** -0.5)
            #     nn.init.normal_(self.eog_proj_head, std=d_model ** -0.5)

            if self.pos and self.pos == "trained":
                self.pos_emb_eeg = pos_embedding(max_pos=200, dim=d_model)
                self.pos_emb_eog = pos_embedding(max_pos=200, dim=d_model)
            elif self.pos and self.pos == "sinusoidal":
                self.pos_emb_eeg = pos_sinusoidal(d_model, max_pos=200)
                self.pos_emb_eog = pos_sinusoidal(d_model, max_pos=200)

            if self.mod_token:
                self.modtype_token = modtype_embedding(num_modalities=2, dim=d_model)

        def forward(self, x, skip_modality="full", **kwargs):
            xeeg = None
            #TODO: There was here a and skip_modality["stft_eeg"].sum()<=len(skip_modality["stft_eeg"]), which I dont understand
            if skip_modality!="eeg":

                xeeg = x["stft_eeg"][:, :, :, :, 1:, :].float()  # mat
                xeeg = einops.rearrange(xeeg, "b outer mod ch f inner -> b outer inner mod ch f")
                if type(skip_modality) == dict and "stft_eeg" in skip_modality:
                    xeeg_shape = xeeg.shape
                    xeeg = xeeg[~skip_modality["stft_eeg"].bool()]
                    xeeg = einops.rearrange(xeeg, "(b outer) i m c f -> b outer i m c f", outer=xeeg_shape[1], b=int(xeeg.shape[0]/xeeg_shape[1]))
                if self.mod_token:
                    xeeg = self.modtype_token(data=xeeg, mod_num=0)
                if self.pos:
                    xeeg = self.pos_emb_eeg.forward_inner(xeeg)

                cls_token_eeg = self.cls_token_eeg.repeat(xeeg.shape[0], xeeg.shape[1], 1, 1, xeeg.shape[3], 1)
                xeeg = torch.cat([cls_token_eeg, xeeg], dim=2)

            xeog = None
            if skip_modality != "eog":
                xeog = x["stft_eog"][:, :, :, :, 1:, :].float()  # mat
                xeog = einops.rearrange(xeog, "b outer mod ch f inner -> b outer inner mod ch f")
                if type(skip_modality) == dict and "stft_eog" in skip_modality:
                    xeog_shape = xeog.shape
                    xeog = xeog[~skip_modality["stft_eog"].bool()]
                    xeog = einops.rearrange(xeog, "(b outer) i m c f -> b outer i m c f", outer=xeog_shape[1],
                                            b=int(xeog.shape[0] / xeog_shape[1]))
                if self.mod_token:
                    xeog = self.modtype_token(data=xeog, mod_num=1)
                if self.pos:
                    xeog = self.pos_emb_eog.forward_inner(xeog)
                cls_token_eog = self.cls_token_eog.repeat(xeog.shape[0], xeog.shape[1], 1, 1, xeog.shape[3], 1)
                xeog = torch.cat([cls_token_eog, xeog], dim=2)

            output = {"output_features": {}}
            output = self.forward_sole(xeeg=xeeg, xeog=xeog, output=output, skip_modality=skip_modality, align_inner = self.align_inner, **kwargs)
            if xeeg is not None and xeog is not None and "inner_eeg" in output["output_features"] and "inner_eog" in output["output_features"]:
                output = self.forward_common(xeeg=xeeg, xeog=xeog, output=output, skip_modality=skip_modality, **kwargs)

            output["output_features"].pop("inner_eeg", None)
            output["output_features"].pop("inner_eog", None)

            return output

        def _keep_common(self, x, common_idx, skip_idx):

            if len(x.shape)==6:
                output = x[einops.rearrange(~common_idx[~skip_idx.bool()], "(b outer) -> b outer", b=x.shape[0],
                                            outer=x.shape[1])]
                output = einops.rearrange(output, "(b outer) i m c f -> b outer i m c f",
                                                 outer=common_idx.shape[1],
                                                 b=int(output.shape[0] / common_idx.shape[1]))
            elif len(x.shape)==3:
                output = x[einops.rearrange(~common_idx[~skip_idx.bool()], "(b outer) -> b outer", b=x.shape[0],
                                            outer=x.shape[1])]
                output = einops.rearrange(output, "(b outer) f -> b outer f",
                                                 outer=common_idx.shape[1],
                                                 b=int(output.shape[0] / common_idx.shape[1]))
            elif len(x.shape)==2:
                #This assumes that batch dim has been squeezed
                x = x.unsqueeze(dim=1)
                output = x[einops.rearrange(~common_idx[~skip_idx.bool()], "(b outer) -> b outer", b=x.shape[0],
                                            outer=x.shape[1])]
                output = einops.rearrange(output, "(b outer) f -> b outer f",
                                                 outer=common_idx.shape[1],
                                                 b=int(output.shape[0] / common_idx.shape[1]))

            return  output
        def forward_common(self, xeeg, xeog, output, **kwargs):

            skip_modality = kwargs["skip_modality"]

            xeeg_common_i = output["output_features"]["inner_eeg"]
            xeog_common_i = output["output_features"]["inner_eog"]

            if xeeg_common_i.shape[1] > 1 and  xeeg_common_i.shape[1] > 1:
                xeeg_common_outer = output["output_features"]["eeg"]
                xeog_common_outer = output["output_features"]["eog"]

            if type(skip_modality) == dict and "stft_eeg" in skip_modality and "stft_eog" in skip_modality:
                common_kept_idx = torch.logical_or(skip_modality["stft_eeg"].bool(), skip_modality["stft_eog"].bool())
                # xeeg_common_i = self._keep_common(xeeg_common_i, output["common_kept_idx"], skip_modality["stft_eeg"])
                # xeog_common_i = self._keep_common(xeog_common_i, output["common_kept_idx"], skip_modality["stft_eog"])
                if "xeeg_common_outer" in locals() and "xeog_common_outer" in locals():
                    xeeg_common_outer = self._keep_common(xeeg_common_outer, common_kept_idx, skip_modality["stft_eeg"])
                    xeog_common_outer = self._keep_common(xeog_common_outer, common_kept_idx, skip_modality["stft_eog"])
                xeeg = self._keep_common(xeeg, common_kept_idx, skip_modality["stft_eeg"])
                xeog = self._keep_common(xeog, common_kept_idx, skip_modality["stft_eog"])

            xeeg_ca_common = self.inner_tf_eeg.forward_inner(xeeg, xeog_common_i)[:, :, :1]
            xeog_ca_common = self.inner_tf_eog.forward_inner(xeog, xeeg_common_i)[:, :, :1]

            if "xeeg_common_outer" in locals() and "xeog_common_outer" in locals():

                if self.pos:
                    xeeg_ca_common = self.pos_emb_eeg.forward_outer(xeeg_ca_common)
                    xeog_ca_common = self.pos_emb_eog.forward_outer(xeog_ca_common)

                xeeg_ca_common_outer = self.outer_tf_eeg.forward_outer(xeeg_ca_common, xeog_common_outer)
                xeog_ca_common_outer = self.outer_tf_eog.forward_outer(xeog_ca_common, xeeg_common_outer)
                x_common = torch.cat([xeeg_ca_common_outer,xeog_ca_common_outer], dim=2)
            else:
                x_common = torch.cat([xeeg_ca_common,xeeg_ca_common], dim=2)

            output["output_features"]["combined"] = x_common

            return output

        def forward_sole(self, xeeg, xeog, output, skip_modality, return_matches=False, **kwargs):

            if xeeg is not None and skip_modality!="eeg" and xeeg.shape[0]>0:
                # and type(skip_modality)==dict and skip_modality["stft_eeg"].sum()<len(skip_modality["stft_eeg"]):
                xeeg_sole = self.inner_tf_eeg.forward_inner(xeeg)
                output["output_features"]["inner_eeg"] = xeeg_sole
                xeeg_cls_sole = xeeg_sole[:, :, :1]
                if xeeg_cls_sole.shape[1]>1:
                    xeeg_outer_sole = self.outer_tf_eeg.forward_outer(xeeg_cls_sole, use_rpos=True)
                    output["output_features"]["eeg"] = xeeg_outer_sole
                    if "align_inner" in kwargs and kwargs["align_inner"]:
                        xeeg_match_sq = xeeg_cls_sole.squeeze()
                    else:
                        xeeg_match_sq = xeeg_outer_sole.squeeze()
                else:
                    output["output_features"]["eeg"] = xeeg_cls_sole

            if xeog is not None and skip_modality != "eog" and xeog.shape[0]>0:
                # and type(skip_modality)==dict and skip_modality["stft_eog"].sum()<len(skip_modality["stft_eog"]):

                xeog_sole = self.inner_tf_eog.forward_inner(xeog)
                output["output_features"]["inner_eog"] = xeog_sole
                xeog_cls_sole = xeog_sole[:, :, :1]
                if xeog_cls_sole.shape[1]>1:
                    xeog_outer_sole = self.outer_tf_eog.forward_outer(xeog_cls_sole, use_rpos=True)
                    output["output_features"]["eog"] = xeog_outer_sole
                    if "align_inner" in kwargs and kwargs["align_inner"]:
                        xeog_match_sq = xeog_cls_sole.squeeze()
                    else:
                        xeog_match_sq = xeog_outer_sole.squeeze()
                else:
                    output["output_features"]["eog"] = xeog_cls_sole


            x_match = None
            if skip_modality != "eeg" and skip_modality != "eog" and return_matches and "xeeg_match_sq" in locals() and "xeog_match_sq" in locals():
                if type(skip_modality) == dict and "stft_eeg" in skip_modality and "stft_eog" in skip_modality:
                    common_kept_idx = torch.logical_or(skip_modality["stft_eeg"].bool(), skip_modality["stft_eog"].bool())
                    xeeg_match_sq = self._keep_common(xeeg_match_sq, common_kept_idx, skip_modality["stft_eeg"])
                    xeog_match_sq = self._keep_common(xeog_match_sq, common_kept_idx, skip_modality["stft_eog"])
                    output["output_features"]["inner_eeg"] = self._keep_common(output["output_features"]["inner_eeg"], common_kept_idx, skip_modality["stft_eeg"])
                    output["output_features"]["inner_eog"] = self._keep_common(output["output_features"]["inner_eog"], common_kept_idx, skip_modality["stft_eog"])

                if len(xeeg_match_sq.shape) == 3 and len(xeog_match_sq.shape) == 3 and xeeg_match_sq.shape[0]>0 and  xeog_match_sq.shape[0]>0:

                    # xeeg_sa_o = torch.einsum('b o f , f p -> b o p', xeeg_sa_o, self.eeg_proj_head)
                    # xeog_sa_o = torch.einsum('b o f , f p -> b o p', xeog_sa_o, self.eog_proj_head)

                    # # cosine similarity as logits
                    # logit_scale = self.logit_scale.exp()

                    if  'big_al' in self.args and self.args['big_al']:
                        xeeg_outer_sole_sq = einops.rearrange(xeeg_match_sq, "b o f -> (b o) f")
                        xeog_outer_sole_sq = einops.rearrange(xeog_match_sq, 'b o f -> (b o) f')

                        # normalized features
                        xeeg_outer_sole_sq_norm = xeeg_outer_sole_sq / xeeg_outer_sole_sq.norm(dim=1, keepdim=True)
                        xeog_outer_sole_sq_norm = xeog_outer_sole_sq / xeog_outer_sole_sq.norm(dim=1, keepdim=True)

                        x_match_eeg = torch.matmul(xeeg_outer_sole_sq_norm,xeog_outer_sole_sq_norm.t())
                        x_match_eog = x_match_eeg.permute(1, 0)

                    else:
                        # normalized features
                        xeeg_outer_sole_sq_norm = xeeg_match_sq / xeeg_match_sq.norm(dim=1, keepdim=True)
                        xeog_outer_sole_sq_norm = xeog_match_sq / xeog_match_sq.norm(dim=1, keepdim=True)

                        x_match_eeg = torch.einsum('b o f , b m f -> b o m', xeeg_outer_sole_sq_norm, xeog_outer_sole_sq_norm)
                        x_match_eog = x_match_eeg.permute(0, 2, 1)

                    x_match = {"stft_eeg": x_match_eeg, "stft_eog": x_match_eog}
                    # else:
                    #     x_match = torch.Tensor([0]).to(xeeg.device)
                elif len(xeeg_match_sq.shape) == 2 and len(xeog_match_sq.shape) == 2:

                    if  'big_al' in self.args and self.args['big_al']:
                        xeeg_outer_sole_sq = einops.rearrange(xeeg_match_sq, "b o f -> (b o) f")
                        xeog_outer_sole_sq = einops.rearrange(xeog_match_sq, 'b o f -> (b o) f')

                        # normalized features
                        xeeg_outer_sole_sq_norm = xeeg_outer_sole_sq / xeeg_outer_sole_sq.norm(dim=1, keepdim=True)
                        xeog_outer_sole_sq_norm = xeog_outer_sole_sq / xeog_outer_sole_sq.norm(dim=1, keepdim=True)

                        x_match_eeg = torch.matmul(xeeg_outer_sole_sq_norm,xeog_outer_sole_sq_norm.t())
                        x_match_eog = x_match_eeg.permute(1, 0)

                    else:
                        # normalized features
                        xeeg_outer_sole_sq_norm = xeeg_match_sq / xeeg_match_sq.norm(dim=1, keepdim=True)
                        xeog_outer_sole_sq_norm = xeog_match_sq / xeog_match_sq.norm(dim=1, keepdim=True)

                        x_match_eeg = torch.einsum('o f , m f -> o m', xeeg_outer_sole_sq_norm, xeog_outer_sole_sq_norm).unsqueeze(dim=0)
                        x_match_eog = x_match_eeg.permute(0, 2, 1)

                    x_match = {"stft_eeg": x_match_eeg, "stft_eog": x_match_eog}

                else:
                    x_match = {"stft_eeg": None, "stft_eog": None}
                    # x_match = torch.Tensor([0]).to(xeeg.device)

            output["matches"] = x_match

            return output
class SleepEnc_BLIP_EEG_EOG_sharedall_free_cliplike(nn.Module):
        def __init__(self, args, encs=[None]):
            """
            :param encoder_filters_small, encoder_filters_big: filters of CNN output, quantized by 20
            :param encs_small, encs_big:
            """
            super().__init__()
            self.args = args

            d_model = args.d_model
            self.pos = args.pos if "pos" in args else True
            self.outer_rep = args.outer_rep if "outer_rep" in args else False
            self.skip_percentile = args.skip_percentile if "skip_percentile" in args else False
            self.mod_token = args.mod_token if "mod_token" in args else False
            self.dropout = args.dropout if "dropout" in args else False
            self.align_inner = args.align_inner if "align_inner" in args else False
            # self.clean_train = args.clean_train if "clean_train" in args else False

            self.disable_mods = dict(stft_eeg=False, stft_eog=False)

            self.inner_tf = TF_Block_SA_CA_CA(CA_flag=True, **args)

            self.outer_tf = TF_Block_SA_CA_CA(CA_flag=True, **args)

            self.cls_token_eeg = nn.Parameter(torch.randn(1, 1, 1, 1, 1, d_model), requires_grad=True)
            self.cls_token_eog = nn.Parameter(torch.randn(1, 1, 1, 1, 1, d_model), requires_grad=True)

            self.cls_token_outer_eeg = nn.Parameter(torch.randn(1, 1, 1, 1, 1, d_model), requires_grad=True)
            self.cls_token_outer_eog = nn.Parameter(torch.randn(1, 1, 1, 1, 1, d_model), requires_grad=True)

            # self.eeg_proj_head = nn.Parameter(torch.randn(d_model, d_model))
            # self.eog_proj_head = nn.Parameter(torch.randn(d_model, d_model))
            #
            # if self.eog_proj_head is not None:
            #     nn.init.normal_(self.eeg_proj_head, std=d_model ** -0.5)
            #     nn.init.normal_(self.eog_proj_head, std=d_model ** -0.5)

            if self.pos and self.pos == "trained":
                self.pos_emb_eeg = pos_embedding(max_pos=200, dim=d_model)
                self.pos_emb_eog = pos_embedding(max_pos=200, dim=d_model)
            elif self.pos and self.pos == "sinusoidal":
                self.pos_emb_eeg = pos_sinusoidal(d_model, max_pos=200)
                self.pos_emb_eog = pos_sinusoidal(d_model, max_pos=200)

            if self.mod_token:
                self.modtype_token = modtype_embedding(num_modalities=2, dim=d_model)

        def forward(self, x, skip_modality="full", **kwargs):
            xeeg = None
            #TODO: There was here a and skip_modality["stft_eeg"].sum()<=len(skip_modality["stft_eeg"]), which I dont understand
            if skip_modality!="eeg":

                xeeg = x["stft_eeg"][:, :, :, :, 1:, :].float()  # mat
                xeeg = einops.rearrange(xeeg, "b outer mod ch f inner -> b outer inner mod ch f")
                if type(skip_modality) == dict and "stft_eeg" in skip_modality:
                    xeeg_shape = xeeg.shape
                    xeeg = xeeg[~skip_modality["stft_eeg"].bool()]
                    xeeg = einops.rearrange(xeeg, "(b outer) i m c f -> b outer i m c f", outer=xeeg_shape[1], b=int(xeeg.shape[0]/xeeg_shape[1]))
                if self.mod_token:
                    xeeg = self.modtype_token(data=xeeg, mod_num=0)
                if self.pos:
                    xeeg = self.pos_emb_eeg.forward_inner(xeeg)

                cls_token_eeg = self.cls_token_eeg.repeat(xeeg.shape[0], xeeg.shape[1], 1, 1, xeeg.shape[3], 1)
                xeeg = torch.cat([cls_token_eeg, xeeg], dim=2)

            xeog = None
            if skip_modality != "eog":
                xeog = x["stft_eog"][:, :, :, :, 1:, :].float()  # mat
                xeog = einops.rearrange(xeog, "b outer mod ch f inner -> b outer inner mod ch f")
                if type(skip_modality) == dict and "stft_eog" in skip_modality:
                    xeog_shape = xeog.shape
                    xeog = xeog[~skip_modality["stft_eog"].bool()]
                    xeog = einops.rearrange(xeog, "(b outer) i m c f -> b outer i m c f", outer=xeog_shape[1],
                                            b=int(xeog.shape[0] / xeog_shape[1]))
                if self.mod_token:
                    xeog = self.modtype_token(data=xeog, mod_num=1)
                if self.pos:
                    xeog = self.pos_emb_eog.forward_inner(xeog)
                cls_token_eog = self.cls_token_eog.repeat(xeog.shape[0], xeog.shape[1], 1, 1, xeog.shape[3], 1)
                xeog = torch.cat([cls_token_eog, xeog], dim=2)

            output = {"output_features": {}}
            output = self.forward_sole(xeeg=xeeg, xeog=xeog, output=output, skip_modality=skip_modality, align_inner = self.align_inner, **kwargs)
            if xeeg is not None and xeog is not None and "inner_eeg" in output["output_features"] and "inner_eog" in output["output_features"]:
                output = self.forward_common(xeeg=xeeg, xeog=xeog, output=output, skip_modality=skip_modality, **kwargs)

            output["output_features"].pop("inner_eeg", None)
            output["output_features"].pop("inner_eog", None)

            return output

        def _keep_common(self, x, common_idx, skip_idx):

            if len(x.shape)==6:
                output = x[einops.rearrange(~common_idx[~skip_idx.bool()], "(b outer) -> b outer", b=x.shape[0],
                                            outer=x.shape[1])]
                output = einops.rearrange(output, "(b outer) i m c f -> b outer i m c f",
                                                 outer=common_idx.shape[1],
                                                 b=int(output.shape[0] / common_idx.shape[1]))
            elif len(x.shape)==3:
                output = x[einops.rearrange(~common_idx[~skip_idx.bool()], "(b outer) -> b outer", b=x.shape[0],
                                            outer=x.shape[1])]
                output = einops.rearrange(output, "(b outer) f -> b outer f",
                                                 outer=common_idx.shape[1],
                                                 b=int(output.shape[0] / common_idx.shape[1]))
            elif len(x.shape)==2:
                #This assumes that batch dim has been squeezed
                x = x.unsqueeze(dim=1)
                output = x[einops.rearrange(~common_idx[~skip_idx.bool()], "(b outer) -> b outer", b=x.shape[0],
                                            outer=x.shape[1])]
                output = einops.rearrange(output, "(b outer) f -> b outer f",
                                                 outer=common_idx.shape[1],
                                                 b=int(output.shape[0] / common_idx.shape[1]))

            return  output
        def forward_common(self, xeeg, xeog, output, **kwargs):

            skip_modality = kwargs["skip_modality"]

            xeeg_common_i = output["output_features"]["inner_eeg"]
            xeog_common_i = output["output_features"]["inner_eog"]

            if xeeg_common_i.shape[1] > 1 and  xeeg_common_i.shape[1] > 1:
                xeeg_common_outer = output["output_features"]["eeg"]
                xeog_common_outer = output["output_features"]["eog"]

            if type(skip_modality) == dict and "stft_eeg" in skip_modality and "stft_eog" in skip_modality:
                common_kept_idx = torch.logical_or(skip_modality["stft_eeg"].bool(), skip_modality["stft_eog"].bool())
                # xeeg_common_i = self._keep_common(xeeg_common_i, output["common_kept_idx"], skip_modality["stft_eeg"])
                # xeog_common_i = self._keep_common(xeog_common_i, output["common_kept_idx"], skip_modality["stft_eog"])
                if "xeeg_common_outer" in locals() and "xeog_common_outer" in locals():
                    xeeg_common_outer = self._keep_common(xeeg_common_outer, common_kept_idx, skip_modality["stft_eeg"])
                    xeog_common_outer = self._keep_common(xeog_common_outer, common_kept_idx, skip_modality["stft_eog"])
                xeeg = self._keep_common(xeeg, common_kept_idx, skip_modality["stft_eeg"])
                xeog = self._keep_common(xeog, common_kept_idx, skip_modality["stft_eog"])

            xeeg_ca_common = self.inner_tf.forward_inner(xeeg, xeog_common_i)[:, :, :1]
            xeog_ca_common = self.inner_tf.forward_inner(xeog, xeeg_common_i)[:, :, :1]

            if "xeeg_common_outer" in locals() and "xeog_common_outer" in locals():

                if self.pos:
                    xeeg_ca_common = self.pos_emb_eeg.forward_outer(xeeg_ca_common)
                    xeog_ca_common = self.pos_emb_eog.forward_outer(xeog_ca_common)

                xeeg_ca_common_outer = self.outer_tf.forward_outer(xeeg_ca_common, xeog_common_outer)
                xeog_ca_common_outer = self.outer_tf.forward_outer(xeog_ca_common, xeeg_common_outer)
                x_common = xeeg_ca_common_outer + xeog_ca_common_outer
            else:
                x_common = xeeg_ca_common + xeeg_ca_common

            output["output_features"]["combined"] = x_common

            return output

        def forward_sole(self, xeeg, xeog, output, skip_modality, return_matches=False, **kwargs):

            if xeeg is not None and skip_modality!="eeg" and xeeg.shape[0]>0:
                # and type(skip_modality)==dict and skip_modality["stft_eeg"].sum()<len(skip_modality["stft_eeg"]):
                xeeg_sole = self.inner_tf.forward_inner(xeeg)
                output["output_features"]["inner_eeg"] = xeeg_sole
                xeeg_cls_sole = xeeg_sole[:, :, :1]
                if xeeg_cls_sole.shape[1]>1:
                    xeeg_outer_sole = self.outer_tf.forward_outer(xeeg_cls_sole, use_rpos=True)
                    output["output_features"]["eeg"] = xeeg_outer_sole
                    if "align_inner" in kwargs and kwargs["align_inner"]:
                        xeeg_match_sq = xeeg_cls_sole.squeeze()
                    else:
                        xeeg_match_sq = xeeg_outer_sole.squeeze()
                else:
                    output["output_features"]["eeg"] = xeeg_cls_sole

            if xeog is not None and skip_modality != "eog" and xeog.shape[0]>0:
                # and type(skip_modality)==dict and skip_modality["stft_eog"].sum()<len(skip_modality["stft_eog"]):

                xeog_sole = self.inner_tf.forward_inner(xeog)
                output["output_features"]["inner_eog"] = xeog_sole
                xeog_cls_sole = xeog_sole[:, :, :1]
                if xeog_cls_sole.shape[1]>1:
                    xeog_outer_sole = self.outer_tf.forward_outer(xeog_cls_sole, use_rpos=True)
                    output["output_features"]["eog"] = xeog_outer_sole
                    if "align_inner" in kwargs and kwargs["align_inner"]:
                        xeog_match_sq = xeog_cls_sole.squeeze()
                    else:
                        xeog_match_sq = xeog_outer_sole.squeeze()
                else:
                    output["output_features"]["eog"] = xeog_cls_sole


            x_match = None
            if skip_modality != "eeg" and skip_modality != "eog" and return_matches and "xeeg_match_sq" in locals() and "xeog_match_sq" in locals():
                if type(skip_modality) == dict and "stft_eeg" in skip_modality and "stft_eog" in skip_modality:
                    common_kept_idx = torch.logical_or(skip_modality["stft_eeg"].bool(), skip_modality["stft_eog"].bool())
                    xeeg_match_sq = self._keep_common(xeeg_match_sq, common_kept_idx, skip_modality["stft_eeg"])
                    xeog_match_sq = self._keep_common(xeog_match_sq, common_kept_idx, skip_modality["stft_eog"])
                    output["output_features"]["inner_eeg"] = self._keep_common(output["output_features"]["inner_eeg"], common_kept_idx, skip_modality["stft_eeg"])
                    output["output_features"]["inner_eog"] = self._keep_common(output["output_features"]["inner_eog"], common_kept_idx, skip_modality["stft_eog"])

                if len(xeeg_match_sq.shape) == 3 and len(xeog_match_sq.shape) == 3 and xeeg_match_sq.shape[0]>0 and  xeog_match_sq.shape[0]>0:

                    # xeeg_sa_o = torch.einsum('b o f , f p -> b o p', xeeg_sa_o, self.eeg_proj_head)
                    # xeog_sa_o = torch.einsum('b o f , f p -> b o p', xeog_sa_o, self.eog_proj_head)

                    # # cosine similarity as logits
                    # logit_scale = self.logit_scale.exp()

                    if  'big_al' in self.args and self.args['big_al']:
                        xeeg_outer_sole_sq = einops.rearrange(xeeg_match_sq, "b o f -> (b o) f")
                        xeog_outer_sole_sq = einops.rearrange(xeog_match_sq, 'b o f -> (b o) f')

                        # normalized features
                        xeeg_outer_sole_sq_norm = xeeg_outer_sole_sq / xeeg_outer_sole_sq.norm(dim=1, keepdim=True)
                        xeog_outer_sole_sq_norm = xeog_outer_sole_sq / xeog_outer_sole_sq.norm(dim=1, keepdim=True)

                        x_match_eeg = torch.matmul(xeeg_outer_sole_sq_norm,xeog_outer_sole_sq_norm.t())
                        x_match_eog = x_match_eeg.permute(1, 0)

                    else:
                        # normalized features
                        xeeg_outer_sole_sq_norm = xeeg_match_sq / xeeg_match_sq.norm(dim=1, keepdim=True)
                        xeog_outer_sole_sq_norm = xeog_match_sq / xeog_match_sq.norm(dim=1, keepdim=True)

                        x_match_eeg = torch.einsum('b o f , b m f -> b o m', xeeg_outer_sole_sq_norm, xeog_outer_sole_sq_norm)
                        x_match_eog = x_match_eeg.permute(0, 2, 1)

                    x_match = {"stft_eeg": x_match_eeg, "stft_eog": x_match_eog}
                    # else:
                    #     x_match = torch.Tensor([0]).to(xeeg.device)
                elif len(xeeg_match_sq.shape) == 2 and len(xeog_match_sq.shape) == 2:

                    if  'big_al' in self.args and self.args['big_al']:
                        xeeg_outer_sole_sq = einops.rearrange(xeeg_match_sq, "b o f -> (b o) f")
                        xeog_outer_sole_sq = einops.rearrange(xeog_match_sq, 'b o f -> (b o) f')

                        # normalized features
                        xeeg_outer_sole_sq_norm = xeeg_outer_sole_sq / xeeg_outer_sole_sq.norm(dim=1, keepdim=True)
                        xeog_outer_sole_sq_norm = xeog_outer_sole_sq / xeog_outer_sole_sq.norm(dim=1, keepdim=True)

                        x_match_eeg = torch.matmul(xeeg_outer_sole_sq_norm,xeog_outer_sole_sq_norm.t())
                        x_match_eog = x_match_eeg.permute(1, 0)

                    else:
                        # normalized features
                        xeeg_outer_sole_sq_norm = xeeg_match_sq / xeeg_match_sq.norm(dim=1, keepdim=True)
                        xeog_outer_sole_sq_norm = xeog_match_sq / xeog_match_sq.norm(dim=1, keepdim=True)

                        x_match_eeg = torch.einsum('b o f , b m f -> b o m', xeeg_outer_sole_sq_norm, xeog_outer_sole_sq_norm)
                        x_match_eog = x_match_eeg.permute(0, 2, 1)

                    x_match = {"stft_eeg": x_match_eeg, "stft_eog": x_match_eog}

                else:
                    x_match = {"stft_eeg": None, "stft_eog": None}
                    # x_match = torch.Tensor([0]).to(xeeg.device)

            output["matches"] = x_match

            return output
class SleepEnc_EarlyConcat_EEG_EOG_free_cliplike(nn.Module):
        def __init__(self, args, encs=[None]):
            """
            :param encoder_filters_small, encoder_filters_big: filters of CNN output, quantized by 20
            :param encs_small, encs_big:
            """
            super().__init__()
            self.args = args

            # self.enc_0 = encs[0]
            # self.enc_1 = encs[0]

            d_model = args.d_model  # 64*8

            self.pos = args.pos if "pos" in args else True
            self.outer_rep = args.outer_rep if "outer_rep" in args else False
            self.skip_percentile = args.skip_percentile if "skip_percentile" in args else False
            self.mod_token = args.mod_token if "mod_token" in args else False
            self.dropout = args.dropout if "dropout" in args else False

            self.inner_tf= TF_Block_SA_CA_CA(CA_flag=True,**args)

            self.outer_tf = TF_Block_SA_CA_CA(CA_flag=True,**args)

            self.cls_token_eeg = nn.Parameter(torch.randn(1, 1, 1, 1, 1, d_model), requires_grad=True)
            self.cls_token_eog = nn.Parameter(torch.randn(1, 1, 1, 1, 1, d_model), requires_grad=True)

            if self.pos:
                self.pos_emb_eeg = pos_embedding(max_pos=200, dim=d_model)
                self.pos_emb_eog = pos_embedding(max_pos=200, dim=d_model)
                # self.inner_positional_embedding = huy_pos_inner(d_model, pos=False, inner=29, outer=21, modalities=1,
                #                                                 channels=1)
                # self.outer_positional_embedding = huy_pos_outer(d_model, pos=False, inner=29, outer=21, modalities=1,
                #                                                 channels=1)

            if self.mod_token:
                self.modtype_token_inner = modtype_embedding(num_modalities=2, dim=d_model)
                self.modtype_token_outer = modtype_embedding(num_modalities=2, dim=d_model)

        def forward(self, x, skip_modality="random", **kwargs):

            xeeg = x["stft_eeg"][:, :, :, :, 1:, :].float()  # mat
            xeog = x["stft_eog"][:, :, :, :, 1:, :].float()  # mat

            xeeg = einops.rearrange(xeeg, "b outer mod ch f inner -> b outer inner mod ch f")
            xeog = einops.rearrange(xeog, "b outer mod ch f inner -> b outer inner mod ch f")

            if self.mod_token:
                xeeg = self.modtype_token_inner(data=xeeg, mod_num=0)
                xeog = self.modtype_token_inner(data=xeog, mod_num=1)

            if self.pos:
                xeeg = self.pos_emb_eeg.forward_inner(xeeg)
                xeog = self.pos_emb_eog.forward_inner(xeog)

            cls_token_eeg = self.cls_token_eeg.repeat(xeeg.shape[0], xeeg.shape[1], 1, 1, xeeg.shape[3], 1)
            xeeg = torch.cat([cls_token_eeg, xeeg], dim=2)

            cls_token_eog = self.cls_token_eog.repeat(xeog.shape[0], xeog.shape[1], 1, 1, xeog.shape[3], 1)
            xeog = torch.cat([xeog, cls_token_eog], dim=2)


            output = {"output_features": {}}
            if skip_modality=="eeg":
                output = self.forward_sole(x=xeog, output=output, mod_num=1, **kwargs)
            elif skip_modality=="eog":
                output = self.forward_sole(x=xeeg, output=output, mod_num=0, **kwargs)
            else:
                output = self.forward_common(xeeg=xeeg, xeog=xeog, output=output, **kwargs)
                output = self.forward_sole(x=xeeg, output=output, mod_num=0, **kwargs)
                output = self.forward_sole(x=xeog, output=output, mod_num=1, **kwargs)

            return output

        def forward_common(self, xeeg, xeog, output, **kwargs):

            x_common = torch.cat([xeeg, xeog], dim=2)

            x_common = self.inner_tf.forward_inner(x_common)

            xeeg_i = x_common[:, :, :1]
            xeog_i = x_common[:, :, -1:]

            if self.mod_token:
                xeeg_i = self.modtype_token_outer(data=xeeg_i, mod_num=0)
                xeog_i = self.modtype_token_outer(data=xeog_i, mod_num=1)

            if self.pos:
                xeeg_i = self.pos_emb_eeg.forward_outer(xeeg_i)
                xeog_i = self.pos_emb_eog.forward_outer(xeog_i)

            xeeg_i_outershape = xeeg_i.shape[1]

            x_common_outer = torch.cat([xeeg_i, xeog_i], dim=1)

            x_common_outer = self.outer_tf.forward_outer(x_common_outer)
            xeeg_outer = x_common_outer[:, :xeeg_i_outershape]
            xeog_outer = x_common_outer[:, xeeg_i_outershape:]

            x_common = xeeg_outer + xeog_outer

            output["output_features"]["combined"] = x_common
            output["output_features"]["eeg"] = xeeg_outer
            output["output_features"]["eog"] = xeog_outer

            return_matches = kwargs["return_matches"] if "return_matches" in kwargs else False

            x_match = None
            if return_matches:
                xeeg_outer = xeeg_outer.squeeze()
                xeog_outer = xeog_outer.squeeze()
                if len(xeeg_outer.shape) == 3 and len(xeog_outer.shape) == 3:

                    # xeeg_sa_o = torch.einsum('b o f , f p -> b o p', xeeg_sa_o, self.eeg_proj_head)
                    # xeog_sa_o = torch.einsum('b o f , f p -> b o p', xeog_sa_o, self.eog_proj_head)

                    # # cosine similarity as logits
                    # logit_scale = self.logit_scale.exp()

                    if  'big_al' in self.args and self.args['big_al']:
                        xeeg_outer_sole_sq = einops.rearrange(xeeg_outer, "b o f -> (b o) f")
                        xeog_outer_sole_sq = einops.rearrange(xeog_outer, 'b o f -> (b o) f')

                        # normalized features
                        xeeg_outer_sole_sq_norm = xeeg_outer_sole_sq / xeeg_outer_sole_sq.norm(dim=1, keepdim=True)
                        xeog_outer_sole_sq_norm = xeog_outer_sole_sq / xeog_outer_sole_sq.norm(dim=1, keepdim=True)

                        x_match_eeg = torch.matmul(xeeg_outer_sole_sq_norm,xeog_outer_sole_sq_norm.t())
                        x_match_eog = x_match_eeg.permute(1, 0)

                    else:
                        # normalized features
                        xeeg_outer_sole_sq_norm = xeeg_outer / xeeg_outer.norm(dim=1, keepdim=True)
                        xeog_outer_sole_sq_norm = xeog_outer / xeog_outer.norm(dim=1, keepdim=True)

                        x_match_eeg = torch.einsum('b o f , b m f -> b o m', xeeg_outer_sole_sq_norm, xeog_outer_sole_sq_norm)
                        x_match_eog = x_match_eeg.permute(0, 2, 1)

                    x_match = {"stft_eeg": x_match_eeg, "stft_eog": x_match_eog}
                else:
                    x_match = torch.Tensor([0]).to(xeeg.device)

            output["matches"] = x_match

            return output
        def forward_sole(self, x, output, mod_num, **kwargs):

            x = self.inner_tf.forward_inner(x)

            x = x[:, :, :1] if mod_num==0 else x[:, :, -1:]

            if self.mod_token:
                x = self.modtype_token_outer(data=x, mod_num=mod_num)

            if self.pos:
                x = self.pos_emb_eeg.forward_outer(x) if mod_num == 0 else \
                    self.pos_emb_eog.forward_outer(x)

            if mod_num == 0:
                output["output_features"]["eeg"] = x
            else:
                output["output_features"]["eog"] = x

            return output


class SleepEnc_EarlyConcat_3pass_EEG_EOG_free_cliplike(nn.Module):
    def __init__(self, args, encs=[None]):
        """
        :param encoder_filters_small, encoder_filters_big: filters of CNN output, quantized by 20
        :param encs_small, encs_big:
        """
        super().__init__()
        self.args = args

        # self.enc_0 = encs[0]
        # self.enc_1 = encs[0]

        d_model = args.d_model  # 64*8

        self.pos = args.pos if "pos" in args else True
        self.outer_rep = args.outer_rep if "outer_rep" in args else False
        self.skip_percentile = args.skip_percentile if "skip_percentile" in args else False
        self.mod_token = args.mod_token if "mod_token" in args else False
        self.dropout = args.dropout if "dropout" in args else False

        self.inner_tf = TF_Block_SA_CA_CA(CA_flag=True, **args)

        self.outer_tf = TF_Block_SA_CA_CA(CA_flag=True, **args)

        self.cls_token_eeg = nn.Parameter(torch.randn(1, 1, 1, 1, 1, d_model), requires_grad=True)
        self.cls_token_eog = nn.Parameter(torch.randn(1, 1, 1, 1, 1, d_model), requires_grad=True)

        if self.pos:
            self.pos_emb_eeg = pos_embedding(max_pos=200, dim=d_model)
            self.pos_emb_eog = pos_embedding(max_pos=200, dim=d_model)
            # self.inner_positional_embedding = huy_pos_inner(d_model, pos=False, inner=29, outer=21, modalities=1,
            #                                                 channels=1)
            # self.outer_positional_embedding = huy_pos_outer(d_model, pos=False, inner=29, outer=21, modalities=1,
            #                                                 channels=1)

        if self.mod_token:
            self.modtype_token_inner = modtype_embedding(num_modalities=2, dim=d_model)
            self.modtype_token_outer = modtype_embedding(num_modalities=2, dim=d_model)

    def forward(self, x, skip_modality="random", **kwargs):
        xeeg = None
        # TODO: There was here a and skip_modality["stft_eeg"].sum()<=len(skip_modality["stft_eeg"]), which I dont understand
        if skip_modality != "eeg":

            xeeg = x["stft_eeg"][:, :, :, :, 1:, :].float()  # mat
            xeeg = einops.rearrange(xeeg, "b outer mod ch f inner -> b outer inner mod ch f")
            if type(skip_modality) == dict and "stft_eeg" in skip_modality:
                xeeg_shape = xeeg.shape
                xeeg = xeeg[~skip_modality["stft_eeg"].bool()]
                xeeg = einops.rearrange(xeeg, "(b outer) i m c f -> b outer i m c f", outer=xeeg_shape[1],
                                        b=int(xeeg.shape[0] / xeeg_shape[1]))
            if self.mod_token:
                xeeg = self.modtype_token_inner(data=xeeg, mod_num=0)
            if self.pos:
                xeeg = self.pos_emb_eeg.forward_inner(xeeg)

            cls_token_eeg = self.cls_token_eeg.repeat(xeeg.shape[0], xeeg.shape[1], 1, 1, xeeg.shape[3], 1).to(xeeg.device)
            xeeg = torch.cat([cls_token_eeg, xeeg], dim=2)

        xeog = None
        if skip_modality != "eog":
            xeog = x["stft_eog"][:, :, :, :, 1:, :].float()  # mat
            xeog = einops.rearrange(xeog, "b outer mod ch f inner -> b outer inner mod ch f")
            if type(skip_modality) == dict and "stft_eog" in skip_modality:
                xeog_shape = xeog.shape
                xeog = xeog[~skip_modality["stft_eog"].bool()]
                xeog = einops.rearrange(xeog, "(b outer) i m c f -> b outer i m c f", outer=xeog_shape[1],
                                        b=int(xeog.shape[0] / xeog_shape[1]))
            if self.mod_token:
                xeog = self.modtype_token_inner(data=xeog, mod_num=1)
            if self.pos:
                xeog = self.pos_emb_eog.forward_inner(xeog)
            cls_token_eog = self.cls_token_eog.repeat(xeog.shape[0], xeog.shape[1], 1, 1, xeog.shape[3], 1).to(xeeg.device)
            xeog = torch.cat([cls_token_eog, xeog], dim=2)

        output = {"output_features": {}}
        if skip_modality == "eeg":
            output = self.forward_sole(x=xeog, output=output, mod_num=1, **kwargs)
        elif skip_modality == "eog":
            output = self.forward_sole(x=xeeg, output=output, mod_num=0, **kwargs)
        else:
            output = self.forward_common(xeeg=xeeg, xeog=xeog, output=output, **kwargs)
            # output = self.forward_sole(x=xeog, output=output, mod_num=1, **kwargs)
            # output = self.forward_sole(x=xeeg, output=output, mod_num=0, **kwargs)
            # output = self.forward_match(output=output, **kwargs)
        return output

    def _keep_common(self, x, common_idx, skip_idx):

        if len(x.shape) == 6:
            output = x[einops.rearrange(~common_idx[~skip_idx.bool()], "(b outer) -> b outer", b=x.shape[0],
                                        outer=x.shape[1])]
            output = einops.rearrange(output, "(b outer) i m c f -> b outer i m c f",
                                      outer=common_idx.shape[1],
                                      b=int(output.shape[0] / common_idx.shape[1]))
        elif len(x.shape) == 3:
            output = x[einops.rearrange(~common_idx[~skip_idx.bool()], "(b outer) -> b outer", b=x.shape[0],
                                        outer=x.shape[1])]
            output = einops.rearrange(output, "(b outer) f -> b outer f",
                                      outer=common_idx.shape[1],
                                      b=int(output.shape[0] / common_idx.shape[1]))
        elif len(x.shape) == 2:
            # This assumes that batch dim has been squeezed
            x = x.unsqueeze(dim=1)
            output = x[einops.rearrange(~common_idx[~skip_idx.bool()], "(b outer) -> b outer", b=x.shape[0],
                                        outer=x.shape[1])]
            output = einops.rearrange(output, "(b outer) f -> b outer f",
                                      outer=common_idx.shape[1],
                                      b=int(output.shape[0] / common_idx.shape[1]))

        return output

    def forward_common(self, xeeg, xeog, output, skip_modality=None, **kwargs):

        if type(skip_modality) == dict and "stft_eeg" in skip_modality and "stft_eog" in skip_modality:
            common_kept_idx = torch.logical_or(skip_modality["stft_eeg"].bool(), skip_modality["stft_eog"].bool())

            xeeg = self._keep_common(xeeg, common_kept_idx, skip_modality["stft_eeg"])
            xeog = self._keep_common(xeog, common_kept_idx, skip_modality["stft_eog"])


        if xeeg.shape[0] > 0 and xeog.shape[0] > 0:
            x_common = torch.cat([xeeg, xeog], dim=2)
            # xeeg_match = self.inner_tf.forward_inner(xeeg)
            # xeog_match = self.inner_tf.forward_inner(xeog)
            x_common = self.inner_tf.forward_inner(x_common)

            eog_cls_pos = int(x_common.shape[2]/2)

            #Version to use in the hooks

            x_match = torch.cat([xeeg, xeog], dim=0)
            x_match = self.inner_tf.forward_inner(x_match)
            xeeg_match = x_match[:xeeg.shape[0], :, :, :, :, :]
            xeog_match = x_match[xeeg.shape[0]:, :, :, :, :, :]

            xeeg_match_sq = xeeg_match[:, :, :1].squeeze()
            xeog_match_sq = xeog_match[:, :, :1].squeeze()

            xeeg_o = x_common[:, :, :1]
            xeog_o = x_common[:, :, eog_cls_pos:eog_cls_pos+1]

            if self.mod_token:
                xeeg_o = self.modtype_token_outer(data=xeeg_o, mod_num=0)
                xeog_o = self.modtype_token_outer(data=xeeg_o, mod_num=1)

            if self.pos:
                xeeg_o = self.pos_emb_eeg.forward_outer(xeeg_o)
                xeog_o = self.pos_emb_eog.forward_outer(xeeg_o)

            xeeg_o_outershape = xeeg_o.shape[1]

            x_common_outer = torch.cat([xeeg_o, xeog_o], dim=1)

            x_common_outer = self.outer_tf.forward_outer(x_common_outer)
            xeeg_outer_s = self.outer_tf.forward_outer(xeeg_match[:, :, :1])
            xeog_outer_s = self.outer_tf.forward_outer(xeog_match[:, :, :1])

            xeeg_outer = x_common_outer[:, :xeeg_o_outershape]
            xeog_outer = x_common_outer[:, xeeg_o_outershape:]

            x_common = xeeg_outer + xeog_outer

            output["output_features"]["combined"] = x_common

            # These are not needed when we do 3 forwards
            output["output_features"]["eeg"] = xeeg_outer_s
            output["output_features"]["eog"] = xeog_outer_s

            # print(output["output_features"]["combined"].shape)
            # print(output["output_features"]["eog"].shape)
            # print(output["output_features"]["eeg"].shape)

            return_matches = kwargs["return_matches"] if "return_matches" in kwargs else False

            x_match = None
            if skip_modality != "eeg" and skip_modality != "eog" and return_matches and "xeeg_match_sq" in locals() and "xeog_match_sq" in locals():

                if len(xeeg_match_sq.shape) == 3 and len(xeog_match_sq.shape) == 3 and xeeg_match_sq.shape[0] > 0 and \
                        xeog_match_sq.shape[0] > 0:

                    if 'big_al' in self.args and self.args['big_al']:
                        xeeg_outer_sole_sq = einops.rearrange(xeeg_match_sq, "b o f -> (b o) f")
                        xeog_outer_sole_sq = einops.rearrange(xeog_match_sq, 'b o f -> (b o) f')

                        # normalized features
                        xeeg_outer_sole_sq_norm = xeeg_outer_sole_sq / xeeg_outer_sole_sq.norm(dim=1, keepdim=True)
                        xeog_outer_sole_sq_norm = xeog_outer_sole_sq / xeog_outer_sole_sq.norm(dim=1, keepdim=True)

                        x_match_eeg = torch.matmul(xeeg_outer_sole_sq_norm, xeog_outer_sole_sq_norm.t())
                        x_match_eog = x_match_eeg.permute(1, 0)

                    else:
                        # normalized features
                        xeeg_outer_sole_sq_norm = xeeg_match_sq / xeeg_match_sq.norm(dim=1, keepdim=True)
                        xeog_outer_sole_sq_norm = xeog_match_sq / xeog_match_sq.norm(dim=1, keepdim=True)

                        x_match_eeg = torch.einsum('b o f , b m f -> b o m', xeeg_outer_sole_sq_norm,
                                                   xeog_outer_sole_sq_norm)
                        x_match_eog = x_match_eeg.permute(0, 2, 1)

                    x_match = {"stft_eeg": x_match_eeg, "stft_eog": x_match_eog}
                    # else:
                    #     x_match = torch.Tensor([0]).to(xeeg.device)
                elif len(xeeg_match_sq.shape) == 2 and len(xeog_match_sq.shape) == 2:

                    if 'big_al' in self.args and self.args['big_al']:
                        xeeg_outer_sole_sq = einops.rearrange(xeeg_match_sq, "b o f -> (b o) f")
                        xeog_outer_sole_sq = einops.rearrange(xeog_match_sq, 'b o f -> (b o) f')

                        # normalized features
                        xeeg_outer_sole_sq_norm = xeeg_outer_sole_sq / xeeg_outer_sole_sq.norm(dim=1, keepdim=True)
                        xeog_outer_sole_sq_norm = xeog_outer_sole_sq / xeog_outer_sole_sq.norm(dim=1, keepdim=True)

                        x_match_eeg = torch.matmul(xeeg_outer_sole_sq_norm, xeog_outer_sole_sq_norm.t())
                        x_match_eog = x_match_eeg.permute(1, 0)

                    else:
                        # normalized features
                        xeeg_outer_sole_sq_norm = xeeg_match_sq / xeeg_match_sq.norm(dim=1, keepdim=True)
                        xeog_outer_sole_sq_norm = xeog_match_sq / xeog_match_sq.norm(dim=1, keepdim=True)

                        x_match_eeg = torch.einsum('b o f , b m f -> b o m', xeeg_outer_sole_sq_norm,
                                                   xeog_outer_sole_sq_norm)
                        x_match_eog = x_match_eeg.permute(0, 2, 1)

                    x_match = {"stft_eeg": x_match_eeg, "stft_eog": x_match_eog}

                else:
                    x_match = {"stft_eeg": None, "stft_eog": None}
                    # x_match = torch.Tensor([0]).to(xeeg.device)

            output["matches"] = x_match

        return output

    def forward_sole(self, x, output, mod_num, **kwargs):

        if x is not None and x.shape[0] > 0:

            x = self.inner_tf.forward_inner(x)
            x = x[:, :, :1] if mod_num == 0 else x[:, :, -1:]
            if self.mod_token:
                x = self.modtype_token_outer(data=x, mod_num=mod_num)
            if self.pos:
                x = self.pos_emb_eeg.forward_outer(x) if mod_num == 0 else \
                    self.pos_emb_eog.forward_outer(x)
            x = self.outer_tf.forward_inner(x)
            if mod_num == 0:
                output["output_features"]["eeg"] = x
            else:
                output["output_features"]["eog"] = x

        return output

    # def forward_match(self, output, skip_modality, **kwargs):
    #
    #     if output["output_features"]["combined"]:
    #         if type(skip_modality) == dict and "stft_eeg" in skip_modality and "stft_eog" in skip_modality:
    #             common_kept_idx = torch.logical_or(skip_modality["stft_eeg"].bool(), skip_modality["stft_eog"].bool())
    #
    #             xeeg = self._keep_common(xeeg, common_kept_idx, skip_modality["stft_eeg"])
    #             xeog = self._keep_common(xeog, common_kept_idx, skip_modality["stft_eog"])
    #
    #         x = self.inner_tf.forward_inner(x)
    #         x = x[:, :, :1] if mod_num == 0 else x[:, :, -1:]
    #         if self.mod_token:
    #             x = self.modtype_token_outer(data=x, mod_num=mod_num)
    #         if self.pos:
    #             x = self.pos_emb_eeg.forward_outer(x) if mod_num == 0 else \
    #                 self.pos_emb_eog.forward_outer(x)
    #         x = self.outer_tf.forward_inner(x)
    #         if mod_num == 0:
    #             output["output_features"]["eeg"] = x
    #         else:
    #             output["output_features"]["eog"] = x
    #
    #     return output
class SleepEnc_Late_EEG_EOG_free_cliplike(nn.Module):
        def __init__(self, args, encs=[None]):
            """
            :param encoder_filters_small, encoder_filters_big: filters of CNN output, quantized by 20
            :param encs_small, encs_big:
            """
            super().__init__()
            self.args = args

            # self.enc_0 = encs[0]
            # self.enc_1 = encs[0]

            d_model = args.d_model  # 64*8

            self.pos = args.pos if "pos" in args else True
            self.outer_rep = args.outer_rep if "outer_rep" in args else False
            self.skip_percentile = args.skip_percentile if "skip_percentile" in args else False
            self.mod_token = args.mod_token if "mod_token" in args else False
            self.dropout = args.dropout if "dropout" in args else False
            self.align_inner = args.align_inner if "align_inner" in args else False

            self.inner_tf_eeg = TF_Block_SA_CA_CA(CA_flag=True, **args)
            self.inner_tf_eog = TF_Block_SA_CA_CA(CA_flag=True, **args)

            self.outer_tf_eeg = TF_Block_SA_CA_CA(CA_flag=True, **args)
            self.outer_tf_eog = TF_Block_SA_CA_CA(CA_flag=True, **args)

            self.cls_token_eeg = nn.Parameter(torch.randn(1, 1, 1, 1, 1, d_model), requires_grad=True)
            self.cls_token_eog = nn.Parameter(torch.randn(1, 1, 1, 1, 1, d_model), requires_grad=True)

            self.cls_token_outer_eeg = nn.Parameter(torch.randn(1, 1, 1, 1, 1, d_model), requires_grad=True)
            self.cls_token_outer_eog = nn.Parameter(torch.randn(1, 1, 1, 1, 1, d_model), requires_grad=True)

            if self.pos:
                self.pos_emb_eeg = pos_embedding(max_pos=200, dim=d_model)
                self.pos_emb_eog = pos_embedding(max_pos=200, dim=d_model)

            if self.mod_token:
                self.modtype_token = modtype_embedding(num_modalities=2, dim=d_model)

        def forward(self, x, skip_modality=None, **kwargs):
            xeeg = None
            #TODO: There was here a and skip_modality["stft_eeg"].sum()<=len(skip_modality["stft_eeg"]), which I dont understand
            if skip_modality!="eeg":

                xeeg = x["stft_eeg"][:, :, :, :, 1:, :].float()  # mat
                xeeg = einops.rearrange(xeeg, "b outer mod ch f inner -> b outer inner mod ch f")
                if type(skip_modality) == dict and "stft_eeg" in skip_modality:
                    xeeg_shape = xeeg.shape
                    xeeg = xeeg[~skip_modality["stft_eeg"].bool()]
                    xeeg = einops.rearrange(xeeg, "(b outer) i m c f -> b outer i m c f", outer=xeeg_shape[1], b=int(xeeg.shape[0]/xeeg_shape[1]))
                if self.mod_token:
                    xeeg = self.modtype_token(data=xeeg, mod_num=0)
                if self.pos:
                    xeeg = self.pos_emb_eeg.forward_inner(xeeg)

                cls_token_eeg = self.cls_token_eeg.repeat(xeeg.shape[0], xeeg.shape[1], 1, 1, xeeg.shape[3], 1).to(xeeg.device)

                xeeg = torch.cat([cls_token_eeg, xeeg], dim=2)

            xeog = None
            if skip_modality != "eog":
                xeog = x["stft_eog"][:, :, :, :, 1:, :].float()  # mat
                xeog = einops.rearrange(xeog, "b outer mod ch f inner -> b outer inner mod ch f")
                if type(skip_modality) == dict and "stft_eog" in skip_modality:
                    xeog_shape = xeog.shape
                    xeog = xeog[~skip_modality["stft_eog"].bool()]
                    xeog = einops.rearrange(xeog, "(b outer) i m c f -> b outer i m c f", outer=xeog_shape[1],
                                            b=int(xeog.shape[0] / xeog_shape[1]))
                if self.mod_token:
                    xeog = self.modtype_token(data=xeog, mod_num=1)
                if self.pos:
                    xeog = self.pos_emb_eog.forward_inner(xeog)
                cls_token_eog = self.cls_token_eog.repeat(xeog.shape[0], xeog.shape[1], 1, 1, xeog.shape[3], 1).to(xeog.device)
                xeog = torch.cat([cls_token_eog, xeog], dim=2)

            output = {"output_features": {}}
            output = self.forward_sole(xeeg=xeeg, xeog=xeog, output=output, skip_modality=skip_modality, align_inner = self.align_inner, **kwargs)
            if xeeg is not None and xeog is not None:
                output = self.forward_common(output=output, skip_modality=skip_modality, **kwargs)

            return output
        def _keep_common(self, x, common_idx, skip_idx):

            if len(x.shape)==6:
                output = x[einops.rearrange(~common_idx[~skip_idx.bool()], "(b outer) -> b outer", b=x.shape[0],
                                            outer=x.shape[1])]
                output = einops.rearrange(output, "(b outer) i m c f -> b outer i m c f",
                                                 outer=common_idx.shape[1],
                                                 b=int(output.shape[0] / common_idx.shape[1]))
            elif len(x.shape)==3:
                output = x[einops.rearrange(~common_idx[~skip_idx.bool()], "(b outer) -> b outer", b=x.shape[0],
                                            outer=x.shape[1])]
                output = einops.rearrange(output, "(b outer) f -> b outer f",
                                                 outer=common_idx.shape[1],
                                                 b=int(output.shape[0] / common_idx.shape[1]))
            elif len(x.shape)==2:
                #This assumes that batch dim has been squeezed
                x = x.unsqueeze(dim=1)
                output = x[einops.rearrange(~common_idx[~skip_idx.bool()], "(b outer) -> b outer", b=x.shape[0],
                                            outer=x.shape[1])]
                output = einops.rearrange(output, "(b outer) f -> b outer f",
                                                 outer=common_idx.shape[1],
                                                 b=int(output.shape[0] / common_idx.shape[1]))

            return  output

        def forward_common(self, output, skip_modality=None, **kwargs):

            xeeg = output["output_features"]["eeg"]
            xeog = output["output_features"]["eog"]
            if type(skip_modality) == dict and "stft_eeg" in skip_modality and "stft_eog" in skip_modality:
                common_kept_idx = torch.logical_or(skip_modality["stft_eeg"].bool(), skip_modality["stft_eog"].bool())

                xeeg = self._keep_common(xeeg, common_kept_idx, skip_modality["stft_eeg"])
                xeog = self._keep_common(xeog, common_kept_idx, skip_modality["stft_eog"])

            output["output_features"]["combined"] = xeeg + xeog

            return output

        def forward_sole(self, xeeg, xeog, output, skip_modality=None, return_matches=False, **kwargs):

            if xeeg is not None and skip_modality!="eeg" and xeeg.shape[0]>0:
                # and type(skip_modality)==dict and skip_modality["stft_eeg"].sum()<len(skip_modality["stft_eeg"]):
                xeeg_sole = self.inner_tf_eeg.forward_inner(xeeg)
                xeeg_cls_sole = xeeg_sole[:, :, :1]
                if xeeg_cls_sole.shape[1]>1:
                    xeeg_outer_sole = self.outer_tf_eeg.forward_outer(xeeg_cls_sole, use_rpos=True)
                    output["output_features"]["eeg"] = xeeg_outer_sole
                    if "align_inner" in kwargs and kwargs["align_inner"]:
                        xeeg_match_sq = xeeg_cls_sole.squeeze()
                    else:
                        xeeg_match_sq = xeeg_outer_sole.squeeze()
                else:
                    output["output_features"]["eeg"] = xeeg_cls_sole

            if xeog is not None and skip_modality != "eog" and xeog.shape[0]>0:
                # and type(skip_modality)==dict and skip_modality["stft_eog"].sum()<len(skip_modality["stft_eog"]):

                xeog_sole = self.inner_tf_eog.forward_inner(xeog)
                xeog_cls_sole = xeog_sole[:, :, :1]
                if xeog_cls_sole.shape[1]>1:
                    xeog_outer_sole = self.outer_tf_eog.forward_outer(xeog_cls_sole, use_rpos=True)
                    output["output_features"]["eog"] = xeog_outer_sole
                    if "align_inner" in kwargs and kwargs["align_inner"]:
                        xeog_match_sq = xeog_cls_sole.squeeze()
                    else:
                        xeog_match_sq = xeog_outer_sole.squeeze()
                else:
                    output["output_features"]["eog"] = xeog_cls_sole

            x_match = None
            if skip_modality != "eeg" and skip_modality != "eog" and return_matches and "xeeg_match_sq" in locals() and "xeog_match_sq" in locals():
                if type(skip_modality) == dict and "stft_eeg" in skip_modality and "stft_eog" in skip_modality:
                    common_kept_idx = torch.logical_or(skip_modality["stft_eeg"].bool(), skip_modality["stft_eog"].bool())
                    xeeg_match_sq = self._keep_common(xeeg_match_sq, common_kept_idx, skip_modality["stft_eeg"])
                    xeog_match_sq = self._keep_common(xeog_match_sq, common_kept_idx, skip_modality["stft_eog"])

                if len(xeeg_match_sq.shape) == 3 and len(xeog_match_sq.shape) == 3 and xeeg_match_sq.shape[0]>0 and  xeog_match_sq.shape[0]>0:

                    # xeeg_sa_o = torch.einsum('b o f , f p -> b o p', xeeg_sa_o, self.eeg_proj_head)
                    # xeog_sa_o = torch.einsum('b o f , f p -> b o p', xeog_sa_o, self.eog_proj_head)

                    # # cosine similarity as logits
                    # logit_scale = self.logit_scale.exp()

                    if  'big_al' in self.args and self.args['big_al']:
                        xeeg_outer_sole_sq = einops.rearrange(xeeg_match_sq, "b o f -> (b o) f")
                        xeog_outer_sole_sq = einops.rearrange(xeog_match_sq, 'b o f -> (b o) f')

                        # normalized features
                        xeeg_outer_sole_sq_norm = xeeg_outer_sole_sq / xeeg_outer_sole_sq.norm(dim=1, keepdim=True)
                        xeog_outer_sole_sq_norm = xeog_outer_sole_sq / xeog_outer_sole_sq.norm(dim=1, keepdim=True)

                        x_match_eeg = torch.matmul(xeeg_outer_sole_sq_norm,xeog_outer_sole_sq_norm.t())
                        x_match_eog = x_match_eeg.permute(1, 0)

                    else:
                        # normalized features
                        xeeg_outer_sole_sq_norm = xeeg_match_sq / xeeg_match_sq.norm(dim=1, keepdim=True)
                        xeog_outer_sole_sq_norm = xeog_match_sq / xeog_match_sq.norm(dim=1, keepdim=True)

                        x_match_eeg = torch.einsum('b o f , b m f -> b o m', xeeg_outer_sole_sq_norm, xeog_outer_sole_sq_norm)
                        x_match_eog = x_match_eeg.permute(0, 2, 1)

                    x_match = {"stft_eeg": x_match_eeg, "stft_eog": x_match_eog}
                    # else:
                    #     x_match = torch.Tensor([0]).to(xeeg.device)
                elif len(xeeg_match_sq.shape) == 2 and len(xeog_match_sq.shape) == 2:

                    if  'big_al' in self.args and self.args['big_al']:
                        xeeg_outer_sole_sq = einops.rearrange(xeeg_match_sq, "b o f -> (b o) f")
                        xeog_outer_sole_sq = einops.rearrange(xeog_match_sq, 'b o f -> (b o) f')

                        # normalized features
                        xeeg_outer_sole_sq_norm = xeeg_outer_sole_sq / xeeg_outer_sole_sq.norm(dim=1, keepdim=True)
                        xeog_outer_sole_sq_norm = xeog_outer_sole_sq / xeog_outer_sole_sq.norm(dim=1, keepdim=True)

                        x_match_eeg = torch.matmul(xeeg_outer_sole_sq_norm,xeog_outer_sole_sq_norm.t())
                        x_match_eog = x_match_eeg.permute(1, 0)

                    else:
                        # normalized features
                        xeeg_outer_sole_sq_norm = xeeg_match_sq / xeeg_match_sq.norm(dim=1, keepdim=True)
                        xeog_outer_sole_sq_norm = xeog_match_sq / xeog_match_sq.norm(dim=1, keepdim=True)

                        x_match_eeg = torch.einsum('b o f , b m f -> b o m', xeeg_outer_sole_sq_norm, xeog_outer_sole_sq_norm)
                        x_match_eog = x_match_eeg.permute(0, 2, 1)

                    x_match = {"stft_eeg": x_match_eeg, "stft_eog": x_match_eog}

                else:
                    x_match = {"stft_eeg": None, "stft_eog": None}
                    # x_match = torch.Tensor([0]).to(xeeg.device)

            output["matches"] = x_match

            return output

class SleepEnc_Unimodal_EEG(nn.Module):
    def __init__(self, args, encs=[]):
        """
        :param encoder_filters_small, encoder_filters_big: filters of CNN output, quantized by 20
        :param encs_small, encs_big:
        """
        super().__init__()
        self.args = args

        d_model = args.d_model  # 64*8

        self.pos = args.pos if "pos" in args else False
        self.outer_rep = args.outer_rep if "outer_rep" in args else False
        self.skip_percentile = args.skip_percentile if "skip_percentile" in args else False
        self.mod_token = args.mod_token if "mod_token" in args else False
        self.dropout = args.dropout if "dropout" in args else False

        self.inner_tf = TF_Block_SA_CA_CA(CA_flag=False, **args)
        self.outer_tf = TF_Block_SA_CA_CA(CA_flag=False, **args)

        self.cls_token = nn.Parameter(torch.randn(1, 1, 1, 1, 1, d_model), requires_grad=True)

        if self.pos and self.pos == "trained":
            self.pos_emb_eeg = pos_embedding(max_pos=200, dim=d_model)
        elif self.pos and self.pos == "sinusoidal":
            self.pos_emb_eeg = pos_sinusoidal(d_model, max_pos=200)

    def forward(self, xeeg, extract_norm=False, **kwargs):
        """

        :param xeeg: dict of Tensor of [batch, outer_seq, modalities=1, ch=1, frequency_features, inner_seq]
        :param extract_norm: flag for post-hoc analysis
        :param kwargs: dict, hyper-parameters passing through the network
        :return:
        """
        xeeg = xeeg["stft_eeg"][:, :, :, :, 1:, :].float()  # mat

        xeeg = einops.rearrange(xeeg, "b outer mod ch f inner -> b outer inner mod ch f")

        if self.pos:
            xeeg = self.pos_emb_eeg.forward_inner(xeeg)

        cls_token_eeg = self.cls_token.repeat(xeeg.shape[0], xeeg.shape[1], 1, 1, xeeg.shape[3], 1).to(xeeg.device)
        xeeg = torch.cat([cls_token_eeg, xeeg], dim=2)

        xeeg = self.inner_tf.forward_inner(xeeg, extract_norm=extract_norm)

        x = xeeg[:, :, 0].unsqueeze(dim=2)

        if self.pos:
            x = self.pos_emb_eeg.forward_outer(x)

        x = self.outer_tf.forward_outer(x, extract_norm=extract_norm)

        return {"output_features":x}
class EEG_SLEEP_BLIP_GM_EEG(nn.Module):

    def __init__(self, encs=[None], args=None):
        """
        :param encoder_filters_small, encoder_filters_big: filters of CNN output, quantized by 20
        :param encs_small, encs_big:
        """
        super().__init__()

        d_model =  args.dmodel#64*8
        fc_inner = args.fc_inner
        num_classes = args.num_classes
        dropout = args.dropout if "dropout" in args else 0.1

        self.dropout_input = False
        if "dropout_input" in args:
            self.dropout_input = True
            self.dropout_features = nn.Dropout(0.15)
            self.dropout_inner = nn.Dropout(0.15)
            self.dropout_outer = nn.Dropout(0.15)
            # self.dropout_mod = nn.Dropout(0.05)

        self.args = args
        self.num_encoders = 0
        for i, enc in enumerate(encs):
            if enc != None:
                setattr(self, "enc_{}".format(i), enc)
                self.num_encoders +=1


        self.fc_out = nn.Sequential(
                        nn.Linear(d_model, fc_inner),
                        nn.ReLU(),
                        nn.Dropout(dropout),
                        nn.Linear(fc_inner, fc_inner),
                        nn.ReLU(),
                        nn.Dropout(dropout),
                        nn.Linear(fc_inner, num_classes)
                    )
    def forward(self, x, **kwargs):

        return_matches = kwargs["return_matches"] if "return_matches" in kwargs else False
        return_inter_reps = kwargs["return_inter_reps"] if "return_inter_reps" in kwargs else False
        return_final_reps = kwargs["return_final_reps"] if "return_final_reps" in kwargs else False
        return_order = kwargs["return_order"] if "return_order" in kwargs else False
        extract_norm = kwargs["extract_norm"] if "extract_norm" in kwargs else False
        return_reconstruction = kwargs["return_reconstruction"] if "return_reconstruction" in kwargs else False
        return_input = return_reconstruction

        if self.dropout_input:
            # x = x
            for i in range(len(x)):
                b, outer, mod, ch, f, inner = x[i].shape
                p = [0.2, 0.2]
                mask = torch.distributions.Bernoulli(probs=(1 - p[i])).sample(torch.Size([b*outer*inner*f]))
                mask_len = len(mask)
                mask = einops.rearrange(mask, "(b outer inner f) -> b outer inner f", b=b, outer=outer, inner=inner, f=f)
                x[i] = einops.rearrange(x[i], "b outer mod ch f inner-> (b mod ch) outer inner f")
                x[i][~mask.bool()] = torch.rand([int(mask_len - mask.sum())]).cuda()
                x[i] = einops.rearrange(x[i], "(b mod ch) outer inner f -> b outer mod ch f inner", b=b, mod=mod, ch=ch)
                # x[i] = einops.rearrange( self.dropout_outer(einops.rearrange(x[i], "b outer mod ch f inner-> (b mod ch f inner) outer")), "(b mod ch f inner) outer -> b outer mod ch f inner", b=b, outer=outer, mod=mod, ch=ch, f=f, inner=inner)
                # x[i] = einops.rearrange( self.dropout_inner(einops.rearrange(x[i], "b outer mod ch f inner-> (b outer mod ch f) inner")), "(b outer mod ch f) inner -> b outer mod ch f inner", b=b, outer=outer, mod=mod, ch=ch, f=f, inner=inner)
                # x[i] = einops.rearrange( self.dropout_features(einops.rearrange(x[i], "b outer mod ch f inner-> (b outer mod ch inner) f")), "(b outer mod ch inner) f -> b outer mod ch f inner", b=b, outer=outer, mod=mod, ch=ch, f=f, inner=inner)

            # x = [self.dropout_features(i) for i in x]
            # x = [self.dropout_inner(i) for i in x]
            # x = [self.dropout_outer(i) for i in x]
            # x = [self.dropout_mod(i) for i in x]

        for i in range(self.num_encoders):
            enc = getattr(self, "enc_{}".format(i))
            total_x = enc(x, **kwargs)
            x = total_x["output_features"]

        if len(x.shape)>2:
            x = x.flatten(start_dim=0, end_dim=1).flatten(start_dim=1)

        x = self.fc_out(x)

        output = {"preds": {"combined": x}}
        if return_matches:
            output["matches"] = total_x["matches"] if "matches" in total_x and total_x["matches"] is not None else torch.Tensor([0]).to(x.device)
        if return_inter_reps:
            output["inter_reps"] = total_x["intermediate_reps"]
        if return_final_reps:
            output["final_reps"] = total_x["output_features"]
        if return_reconstruction:
            output["reconstruction"] = total_x["reconstruction"]
            output["input"] = total_x["input"]

        return output
class SleepEnc_Unimodal_EOG(nn.Module):
    def __init__(self, args, encs=[]):
        """
        :param encoder_filters_small, encoder_filters_big: filters of CNN output, quantized by 20
        :param encs_small, encs_big:
        """
        super().__init__()
        self.args = args

        d_model = args.d_model  # 64*8

        self.pos = args.pos if "pos" in args else False
        self.outer_rep = args.outer_rep if "outer_rep" in args else False
        self.skip_percentile = args.skip_percentile if "skip_percentile" in args else False
        self.mod_token = args.mod_token if "mod_token" in args else False
        self.dropout = args.dropout if "dropout" in args else False

        self.inner_tf = TF_Block_SA_CA_CA(CA_flag=False, **args)
        self.outer_tf = TF_Block_SA_CA_CA(CA_flag=False, **args)

        self.cls_token = nn.Parameter(torch.randn(1, 1, 1, 1, 1, d_model), requires_grad=True)

        if self.pos and self.pos == "trained":
            self.pos_emb_eeg = pos_embedding(max_pos=200, dim=d_model)
        elif self.pos and self.pos == "sinusoidal":
            self.pos_emb_eeg = pos_sinusoidal(d_model, max_pos=200)

    def forward(self, xeeg, extract_norm=False, **kwargs):
        """

        :param xeeg: dict of Tensor of [batch, outer_seq, modalities=1, ch=1, frequency_features, inner_seq]
        :param extract_norm: flag for post-hoc analysis
        :param kwargs: dict, hyper-parameters passing through the network
        :return:
        """
        xeeg = xeeg["stft_eog"][:, :, :, :, 1:, :].float()  # mat
        xeeg = einops.rearrange(xeeg, "b outer mod ch f inner -> b outer inner mod ch f")
        if self.pos:
            xeeg = self.pos_emb_eeg.forward_inner(xeeg)

        cls_token_eeg = self.cls_token.repeat(xeeg.shape[0], xeeg.shape[1], 1, 1, xeeg.shape[3], 1)
        xeeg = torch.cat([cls_token_eeg, xeeg], dim=2)

        xeeg = self.inner_tf.forward_inner(xeeg, extract_norm=extract_norm)

        x = xeeg[:, :, 0].unsqueeze(dim=2)

        if self.pos:
            x = self.pos_emb_eeg.forward_outer(x)

        x = self.outer_tf.forward_outer(x, extract_norm=extract_norm)

        return {"output_features":x}


class TF_Block_SA_CA_CA(nn.Module):
    def __init__(self, d_model, num_layers=4, **kwargs):
        super().__init__()

        enc = My_TF(d_model, **kwargs)
        self.tf = My_TransformerEncoder(enc, num_layers=num_layers)

    def forward(self, x, **kwargs):
        return self.forward_sa(x,  **kwargs)

    def forward_inner(self, x, x_ca = None, x_ca_ca=None, **kwargs):
        x_shape = x.shape
        self.batch, self.outer, self.inner, self.mod, self.ch, self.features = x_shape[0], x_shape[1], x_shape[2], x_shape[3], x_shape[4], x_shape[5]

        x = einops.rearrange(x, "b outer inner mod ch k -> (inner mod ch) (b outer) k")
        if x_ca is not None:
            x_ca = einops.rearrange(x_ca, "b outer inner mod ch k -> (inner mod ch) (b outer) k")
        if x_ca_ca is not None:
            x_ca_ca = einops.rearrange(x_ca_ca, "b outer inner mod ch k -> (outer mod) (b inner ch) k")
        x = self.tf(x, src_ca = x_ca, src_ca_ca = x_ca_ca,  **kwargs)
        x = einops.rearrange(x, " (inner mod ch) (b outer) k -> b outer inner mod ch k", outer=self.outer, mod=self.mod, ch=self.ch,  b=self.batch)
        return x

    def forward_outer(self, x, x_ca = None, x_ca_ca = None,  **kwargs):
        x_shape = x.shape

        x = einops.rearrange(x, "b outer inner mod ch k -> (outer mod) (b inner ch) k")
        if x_ca is not None:
            x_ca = einops.rearrange(x_ca, "b outer inner mod ch k -> (outer mod) (b inner ch) k")
        if x_ca_ca is not None:
            x_ca_ca = einops.rearrange(x_ca_ca, "b outer inner mod ch k -> (outer mod) (b inner ch) k")
        x = self.tf(x, src_ca = x_ca, src_ca_ca=x_ca_ca, **kwargs)
        x = einops.rearrange(x, " (outer mod) (b inner ch) k -> b outer inner mod ch k", outer=x_shape[1], mod=x_shape[3], ch=x_shape[4],  b=x_shape[0])
        return x

class Positionwise_FC(nn.Module):
    def __init__(self, d_model, dim_feedforward=1024, dropout=0.1, activation="relu", **kwargs):
        super().__init__()

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout_in = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout_out = nn.Dropout(dropout)

        self.activation = nn.ReLU()

    def forward(self, src: Tensor) -> Tensor:

        src_fc = self.linear2(self.dropout_in(self.activation(self.linear1(src))))
        src_att = self.norm(src + self.dropout_out(src_fc))

        return src_att

class My_TF(nn.Module):
    def __init__(self, d_model, CA_flag=False, CA_CA_flag=False, num_heads=8, extra_attention=False, dropout=0.1, shared=True, **kwargs):
        # super(nn.TransformerEncoderLayer, self).__init__()
        super().__init__()

        self.shared = shared
        self.CA_flag = CA_flag
        self.CA_CA_flag = CA_CA_flag
        self.extra_attention = extra_attention

        if self.extra_attention:
            self.extra_self_attn = My_MultiHeadAttention(d_model, num_heads=num_heads, **kwargs)
            self.extra_norm = nn.LayerNorm(d_model)
            self.extra_dropout = nn.Dropout(dropout)
            if not self.shared:
                self.extra_self_attn_2 = My_MultiHeadAttention(d_model, num_heads=num_heads, **kwargs)
                self.extra_norm_2 = nn.LayerNorm(d_model)
                self.extra_dropout_2 = nn.Dropout(dropout)


        self.self_attn = My_MultiHeadAttention(d_model, num_heads=num_heads, **kwargs)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        if not self.shared:
            self.self_attn_2 = My_MultiHeadAttention(d_model, num_heads=num_heads, **kwargs)
            self.norm_2 = nn.LayerNorm(d_model)
            self.dropout_2 = nn.Dropout(dropout)

        if self.CA_flag:
            self.CA = My_MultiHeadAttention(d_model, num_heads=num_heads, **kwargs)
            self.norm_CA = nn.LayerNorm(d_model)
            self.dropout_CA = nn.Dropout(dropout)

        if self.CA_CA_flag:
            self.CA_CA = My_MultiHeadAttention(d_model, num_heads=num_heads, **kwargs)
            self.norm_CA_CA = nn.LayerNorm(d_model)
            self.dropout_CA_CA = nn.Dropout(dropout)

        self.fc_SA = Positionwise_FC(d_model=d_model, dropout=dropout, **kwargs)
        if not self.shared:
            self.fc_SA_2 = Positionwise_FC(d_model=d_model, dropout=dropout, **kwargs)

        self.activation = nn.ReLU()
        # self.norm_calc = BertNormOutput(num_attention_heads=nhead, hidden_size=d_model)
        self.nhead = num_heads


    def forward(self, src: Tensor, crossatt_src: Optional[Tensor] = None, crossatt_src_2: Optional[Tensor] = None, **kwargs) -> Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        x_shape = src.shape

        if self.extra_attention:
            src_extr_att, att_0= self.extra_self_attn(src, src, src, **kwargs)
            src_extr_att = src + self.extra_dropout(src_extr_att)
            src_extr_att = self.extra_norm(src_extr_att)
        else:
            src_extr_att = src

        if self.shared or (crossatt_src_2 is not None and crossatt_src is not None):
            src_att, att = self.self_attn(src_extr_att, src_extr_att, src_extr_att, **kwargs)
            src_att = self.norm(src_extr_att + self.dropout(src_att))
        else:
            src_att, att = self.self_attn_2(src_extr_att, src_extr_att, src_extr_att, **kwargs)
            src_att = self.norm_2(src_extr_att + self.dropout_2(src_att))

        if self.CA_flag and crossatt_src is not None:
            src_ca, att = self.CA(crossatt_src, src_att, src_att, **kwargs)
            src_att = self.norm_CA(src_att + self.dropout_CA(src_ca))
            # src_att = self.fc_CA(src_att)

        if self.CA_CA_flag and crossatt_src_2 is not None:
            src_ca, att = self.CA_CA(crossatt_src_2, src_att, src_att, **kwargs)
            src_att = self.norm_CA_CA(src_att + self.dropout_CA_CA(src_ca))
            # src_att = self.fc_CA_CA(src_att)

        if self.shared or (crossatt_src_2 is not None and crossatt_src is not None):
            src_att = self.fc_SA(src_att)
        else:
            src_att = self.fc_SA_2(src_att)

        return src_att

class My_TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super(My_TransformerEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src: Tensor, src_ca: Optional[Tensor]=None, src_ca_ca: Optional[Tensor]=None, mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None, return_layer="last", ca_type=None, **kwargs) -> Tensor:
        r"""Pass the input through the encoder layers in turn.

        Args:
            src: the sequence to the encoder (required).
            mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        return_layer = kwargs["return_layer"] if "return_layer" in kwargs else "last"
        ca_type = kwargs["ca_type"] if "ca_type" in kwargs else None

        output = src
        output_list = []
        for li, mod in enumerate(self.layers):
            if (src_ca is not None) and (src_ca_ca is not None):
                this_ca = src_ca[li] if ca_type=="full" else src_ca
                this_ca_ca = src_ca_ca[li] if ca_type=="full" else src_ca_ca
                output = mod(output, crossatt_src=this_ca, crossatt_src_1=this_ca_ca,  **kwargs)
            elif (src_ca is not None):
                this_ca = src_ca[li] if ca_type=="full" else src_ca
                output = mod(output, crossatt_src=this_ca, **kwargs)
            elif (src_ca_ca is not None):
                this_ca_ca = src_ca_ca[li] if ca_type=="full" else src_ca_ca
                output = mod(output, crossatt_ca_src=this_ca_ca, **kwargs)
            else:
                output = mod(output, **kwargs)
            if return_layer != "last":
                output_list.append(output)

        if return_layer=="all":
            output = torch.cat([i.unsqueeze(dim=0) for i in output_list])
        return output

class modtype_embedding(nn.Module):
    def __init__(self, num_modalities, dim):
        super().__init__()
        self.mod_tokens = nn.Parameter(torch.randn(num_modalities, dim), requires_grad=True)

    def forward(self, data, mod_num):
        return data + self.mod_tokens[mod_num].to(data.device)

class pos_embedding(nn.Module):
    def __init__(self, max_pos, dim):
        super().__init__()
        self.pos_inner_tokens = nn.Parameter(torch.randn(max_pos, dim), requires_grad=True)
        self.pos_outer_tokens = nn.Parameter(torch.randn(max_pos, dim), requires_grad=True)

    def forward_outer(self, data):
        """

        :param data: [batch, outer, inner, mod, ch, features]
        :return: [batch, outer, inner, mod, ch, features] by having added pos_inner_tokens
        """

        data_shape = data.shape
        data = einops.rearrange(data, "b outer inner mod ch k -> (b inner mod ch) outer k")
        data = data + self.pos_inner_tokens[:data.shape[1],:].squeeze()
        data = einops.rearrange(data, "(b inner mod ch) outer k -> b outer inner mod ch k",  b=data_shape[0],  inner=data_shape[2], mod=data_shape[3], ch=data_shape[4])
        return data

    def forward_inner(self, data):
        """

        :param data: [batch, outer, inner, mod, ch, features]
        :return: [batch, outer, inner, mod, ch, features] by having added pos_outer_tokens
        """
        data_shape = data.shape
        data = einops.rearrange(data, "b outer inner mod ch k -> (b outer mod ch) inner k")
        data = data + self.pos_outer_tokens[:data.shape[1],:].squeeze()
        data = einops.rearrange(data, "(b outer mod ch) inner k -> b outer inner mod ch k",  b=data_shape[0],  outer=data_shape[1], mod=data_shape[3], ch=data_shape[4])
        return data
class pos_sinusoidal(nn.Module):
    def __init__(self, dmodel, max_pos=400):
        super().__init__()
        # self.pos = PositionalEncoding_AIAYN(dmodel, n_position=max_pos)
        # Not a parameter
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(max_pos, dmodel))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        ''' Sinusoid position encoding table '''
        # TODO: make it with torch instead of numpy

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table)

    def forward_outer(self, data):
        """

        :param data: [batch, outer, inner, mod, ch, features]
        :return: [batch, outer, inner, mod, ch, features] by having added pos_inner_tokens
        """

        data_shape = data.shape
        data = einops.rearrange(data, "b outer inner mod ch k -> (b inner mod ch) outer k")
        data = data + self.pos_table[:data.shape[1],:].squeeze()
        data = einops.rearrange(data, "(b inner mod ch) outer k -> b outer inner mod ch k",  b=data_shape[0],  inner=data_shape[2], mod=data_shape[3], ch=data_shape[4])
        return data

    def forward_inner(self, data):
        """

        :param data: [batch, outer, inner, mod, ch, features]
        :return: [batch, outer, inner, mod, ch, features] by having added pos_outer_tokens
        """
        data_shape = data.shape
        data = einops.rearrange(data, "b outer inner mod ch k -> (b outer mod ch) inner k")
        data = data + self.pos_table[:data.shape[1],:].squeeze()
        data = einops.rearrange(data, "(b outer mod ch) inner k -> b outer inner mod ch k",  b=data_shape[0],  outer=data_shape[1], mod=data_shape[3], ch=data_shape[4])
        return data

class My_MultiHeadAttention(nn.Module):

    def __init__(self,
                 in_features,
                 num_heads,
                 bias=True,
                 dim_proj = 128,
                 activation= None,
                 gbiased = None,
                 rpos = False,
                 **kwargs
                 ):
        """Multi-head attention.
        :param in_features: Size of each input sample.
        :param head_num: Number of heads.
        :param bias: Whether to use the bias term.
        :param activation: The activation after each linear transformation.
        """
        super(My_MultiHeadAttention, self).__init__()
        if in_features % num_heads != 0:
            raise ValueError('`in_features`({}) should be divisible by `head_num`({})'.format(in_features, head_num))

        self.in_features = in_features
        self.head_num = num_heads
        self.activation = activation
        if self.activation=="relu":
            self.activation = nn.ReLU()

        self.bias = bias
        self.gbiased = gbiased
        self.linear_q = nn.Linear(in_features, dim_proj, bias)
        self.linear_k = nn.Linear(in_features, dim_proj, bias)
        self.linear_v = nn.Linear(in_features, dim_proj, bias)
        self.linear_o = nn.Linear(dim_proj, in_features, False)

        self.scaled_dotproduct_attention =  ScaledDotProductAttention( rpos=rpos, d_head=int(dim_proj / num_heads), head_num=num_heads)

    def forward(self, q, k, v, attn_mask=None, **kwargs):
        prev = v
        q, k, v = self.linear_q(q), self.linear_k(k), self.linear_v(v)

        if self.activation is not None:
            q = self.activation(q)
            k = self.activation(k)
            v = self.activation(v)

        q = self._reshape_to_batches(q)
        k = self._reshape_to_batches(k)
        v = self._reshape_to_batches(v)


        if attn_mask is not None:
            attn_mask = attn_mask.repeat(self.head_num, 1, 1)
        y, att = self.scaled_dotproduct_attention(q, k, v, self.gbiased, prevalue=prev, mask=attn_mask, **kwargs)

        y = self._reshape_from_batches(y)
        y = self.linear_o(y)

        if self.activation is not None:
            y = self.activation(y)

        return y, att

    @staticmethod
    def gen_history_mask(x):
        """Generate the mask that only uses history data.
        :param x: Input tensor.
        :return: The mask.
        """
        batch_size, seq_len, _ = x.size()
        return torch.tril(torch.ones(seq_len, seq_len)).view(1, seq_len, seq_len).repeat(batch_size, 1, 1)

    def _reshape_to_batches(self, x):
        seq_len, batch_size, in_feature = x.size()
        sub_dim = in_feature // self.head_num
        return einops.rearrange(x, "seq b (h sub_dim)-> seq (b h) sub_dim", h=self.head_num, sub_dim=sub_dim)


    def _reshape_from_batches(self, x):
        seq_len, batch_size, in_feature = x.size()
        batch_size //= self.head_num
        return einops.rearrange(x, "seq (b h) sub_dim -> seq b (h sub_dim)", h=self.head_num, b=batch_size)

    def extra_repr(self):
        return 'in_features={}, head_num={}, bias={}, activation={}'.format(
            self.in_features, self.head_num, self.bias, self.activation,
        )

class ScaledDotProductAttention(nn.Module):
    def __init__(self, rpos=False, d_head=16, max_len=7, head_num=8):
        super().__init__()
        self.rpos = rpos
        self.head_num = head_num
        if rpos:
            self.k_rpos = Relative_Positional_Embeddings(tokens=max_len, dim_head=d_head, heads=head_num)
            self.v_rpos = Relative_Positional_Embeddings(tokens=max_len, dim_head=d_head, heads=head_num)

    def forward(self, query, key, value, gbiased, prevalue=None, mask=None, **kwargs):
        query = einops.rearrange(query,"seq b f -> b seq f")
        key = einops.rearrange(key,"seq b f -> b seq f")
        value = einops.rearrange(value,"seq b f -> b seq f")
        # attn_output, att_weights = F._scaled_dot_product_attention(query, key, value, attn_mask=mask)
        dk = query.size()[-1]

        use_rpos = kwargs["use_rpos"] if "use_rpos" in kwargs else True
        if self.rpos and use_rpos:
            rel_key = einops.rearrange(key,"(b h) seq f -> b h seq f", b = int(key.shape[0]/self.head_num), h = self.head_num)
            rel_key = self.k_rpos(rel_key)
            rel_key = einops.rearrange(rel_key, " b h seq f -> (b h) seq f ")
            scores = (query.matmul(key.transpose(-2, -1)) + rel_key)/ math.sqrt(dk)
        else:
            scores = query.matmul(key.transpose(-2, -1)) / math.sqrt(dk)

        # if mask is not None:
        #     scores = scores.masked_fill(mask == 0, -1e9)

        if gbiased:
            attention = gbiased(scores, prevalue)
        else:
            attention = nn.functional.softmax(scores, dim=-1)

        attn_output = torch.einsum('b i j , b j d -> b i d', attention, value)
        attn_output = einops.rearrange(attn_output," b seq f -> seq b f")

        return attn_output, attention

class Relative_Positional_Embeddings(nn.Module):
    def __init__(self, tokens, dim_head, max_tokens = 2000, heads=None):
        """
        Output: [batch head tokens tokens]
        Args:
            tokens: the number of the tokens of the seq
            dim_head: the size of the last dimension of q
            heads: if None representation is shared across heads.
            else the number of heads must be provided
        """
        super().__init__()
        scale = dim_head ** -0.5
        self.shared_heads = heads if heads is not None else True
        if self.shared_heads:
            self.rel_pos_emb = nn.Parameter(torch.randn(heads, 2 * tokens - 1, dim_head) * scale, requires_grad=True)
        else:
            self.rel_pos_emb = nn.Parameter(torch.randn(2 * tokens - 1, dim_head) * scale, requires_grad=True)

        #Create an indice table to call the matrix and speed up during training
        indices = torch.arange(-tokens, tokens+1)
        self.indices_ext = torch.cat([indices[0].repeat(int((max_tokens - 2 * tokens + 1))), indices, indices[-1].repeat(int((max_tokens - 2 * tokens + 1)))], dim=0)
        self.indices_ext = self.indices_ext.unsqueeze(dim=0).repeat(max_tokens, 1)
        for i in range(max_tokens):
            self.indices_ext[i] = torch.roll(self.indices_ext[i], shifts=i)
        self.indices_ext = self.indices_ext[:max_tokens - tokens + 2, max_tokens - tokens + 1:]

    def forward(self, q):
        if self.shared_heads:
            this_indices_ext = self.indices_ext[:q.shape[2], :q.shape[2]]
            emb = torch.einsum('b h t d, h r t d -> b h t r', q, self.rel_pos_emb[:,this_indices_ext])
        else:
            this_indices_ext = self.indices_ext[:q.shape[2],q.shape[2]]
            emb = torch.einsum('b h t d, r t d -> b h t r', q, self.rel_pos_emb[this_indices_ext])

        return emb

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])