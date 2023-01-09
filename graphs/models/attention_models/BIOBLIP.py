import copy

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F

from graphs.models.custom_layers.eeg_encoders import *
# from graphs.models.minirocket import fit, transform
# from fairseq.models.lightconv import LightConvEncoderLayer
from graphs.models.attention_models.stand_alone_att_vision import *

import sys
import transformers
import collections

sys.path.insert(1, '/users/sista/kkontras/Documents/Sleep_Project/Github_Project/vilbert-multi-task')

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
class EEG_SLEEP_BLIP_GM_EEG_Threemode(nn.Module):

    def __init__(self, encs=[None], args=None):
        """
        :param encoder_filters_small, encoder_filters_big: filters of CNN output, quantized by 20
        :param encs_small, encs_big:
        """
        super().__init__()

        d_model =  args.dmodel#64*8
        fc_inner = args.fc_inner
        num_classes = args.num_classes

        self.args = args
        self.num_encoders = 0

        for i, enc in enumerate(encs):
            if enc != None:
                setattr(self, "enc_{}".format(i), enc)
                self.num_encoders +=1

        self.fc_out = nn.Sequential(
                        # nn.BatchNorm1d(d_model),
                        nn.Linear(d_model, fc_inner),
                        nn.ReLU(),
                        nn.Dropout(0.1),
                        nn.Linear(fc_inner, fc_inner),
                        nn.ReLU(),
                        nn.Dropout(0.1),

                        # nn.Dropout(0.45),
                        nn.Linear(fc_inner, num_classes)
                        # nn.Softmax(dim=1)
                    )
    def forward(self, x, **kwargs):

        return_matches = kwargs["return_matches"] if "return_matches" in kwargs else False

        for i in range(self.num_encoders):
            enc = getattr(self, "enc_{}".format(i))
            x = enc(x, **kwargs)

        output_features = x["output_features"]


        output_lens = {}
        allpred_features = []
        for pred in output_features:
            if len(output_features[pred].shape)>2:
                output_features[pred] = output_features[pred].flatten(start_dim=0, end_dim=1).flatten(start_dim=1)
            output_lens[pred] = len(output_features[pred])
            allpred_features.append(output_features[pred])
        out_x = torch.cat(allpred_features, dim=0)


        out_x = self.fc_out(out_x)

        if "skip_modality" in kwargs and "stft_eeg" in kwargs["skip_modality"]:
            kwargs["skip_modality"]["combined"] = kwargs["skip_modality"]["stft_eeg"]*1 + kwargs["skip_modality"]["stft_eog"]*2

        output = {"preds": {}}
        target_mask = {}
        count = 0
        for pred in output_lens:
            output["preds"][pred] = out_x[count:count+output_lens[pred]]
            count+= output_lens[pred]
            if "skip_modality" in kwargs and "stft_eeg" in kwargs["skip_modality"]:
                for i in kwargs["skip_modality"]:
                    if pred in i:
                        this_target_mask = kwargs["skip_modality"][i]
                target_mask[pred] = this_target_mask.flatten()
        #This target mask is used to sample the training labels that we want to backpropagate with in case we already know some broken data.
        output["target_mask"] = target_mask

        # if "skip_modality" in kwargs and "combined" in kwargs["skip_modality"] and not self.training:
        #     kwargs["skip_modality"]["combined"][kwargs["skip_modality"]["combined"]==3]=0
        #     output["preds"]["skipped"] = copy.deepcopy(output["preds"]["combined"])
        #     output["preds"]["skipped"][(kwargs["skip_modality"]["combined"]==1).flatten()] = output["preds"]["eog"][(kwargs["skip_modality"]["combined"]==1).flatten()]
        #     output["preds"]["skipped"][(kwargs["skip_modality"]["combined"]==2).flatten()] = output["preds"]["eeg"][(kwargs["skip_modality"]["combined"]==2).flatten()]


        output["preds"] = collections.OrderedDict(sorted(output["preds"].items()))

        if return_matches:
            output["matches"] = x["matches"] if "matches" in x and x["matches"] is not None else torch.Tensor([0]).to(out_x.device)

        return output
class EEG_SLEEP_LE_GM(nn.Module):

    def __init__(self, encs=[None], args=None):
        """
        :param encoder_filters_small, encoder_filters_big: filters of CNN output, quantized by 20
        :param encs_small, encs_big:
        """
        super().__init__()

        d_model =  args.dmodel#64*8
        fc_inner = args.fc_inner
        num_classes = args.num_classes

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
                        nn.Linear(d_model, num_classes)
                    )
    def forward(self, x, inits=None, return_matches=False, extract_norm=False, return_inter_reps=False, return_final_reps=False, return_order=False):

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
            total_x = enc(x, return_matches=return_matches, extract_norm=extract_norm, return_inter_reps=return_inter_reps, return_order=return_order)
            x = total_x[0]
            index_enc_output = 1
            if return_matches:
                x_match = total_x[index_enc_output]
                index_enc_output += 1
            if return_inter_reps:
                inter_views = total_x[index_enc_output]
                index_enc_output += 1


        if len(x.shape)>2:
            x = x.flatten(start_dim=0, end_dim=1).flatten(start_dim=1)
        x = self.fc_out(x)

        output = {"preds": {"combined": x}}
        if return_matches:
            output["matches"] = x_match
        if return_inter_reps:
            output["inter_reps"] = inter_views
        if return_final_reps:
            output["final_reps"] = total_x[0]

        return output

class Transformer_Sleep_GM(nn.Module):

        def __init__(self, encs=[None], args=None):
            """
            :param encoder_filters_small, encoder_filters_big: filters of CNN output, quantized by 20
            :param encs_small, encs_big:
            """
            super().__init__()

            d_model = args.dmodel  # 64*8
            fc_inner = args.fc_inner
            num_classes = args.num_classes

            if "training_type" in args and \
                args.training_type == "alignment_order" or args.training_type == "alignment_order_multisupervised" \
                and "multi_loss" in args and \
                "multi_loss_weights" in args.multi_loss and \
                args.multi_loss.multi_loss_weights["order_loss"]!=0:


                self.fc_out_order = nn.Sequential(
                    # nn.BatchNorm1d(d_model),
                    nn.Linear(d_model * 3, fc_inner),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(fc_inner, fc_inner),
                    nn.ReLU(),
                    nn.Dropout(0.1),

                    # nn.Dropout(0.45),
                    nn.Linear(fc_inner, 2),
                    nn.Softmax(dim=1)
                )


            self.args = args
            self.num_encoders = 0
            for i, enc in enumerate(encs):
                if enc != None:
                    setattr(self, "enc_{}".format(i), enc)
                    self.num_encoders += 1

            self.fc_out = nn.Sequential(
                # nn.BatchNorm1d(d_model),
                nn.Linear(d_model, fc_inner),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(fc_inner, fc_inner),
                nn.ReLU(),
                nn.Dropout(0.1),

                # nn.Dropout(0.45),
                nn.Linear(fc_inner, num_classes),
                nn.Softmax(dim=1)
            )


        def forward(self, x, return_matches=False, extract_norm=False, return_inter_reps=False, return_order=False):

            for enc_i in range(self.num_encoders):
                enc = getattr(self, "enc_{}".format(enc_i))
                # total_x -> x_features, x_matches, intermediate_views, intermediate_outer_views
                total_x = enc(x, return_matches=return_matches, extract_norm=extract_norm, return_inter_reps=return_inter_reps, return_order=return_order)
                x = total_x[0]
                index_enc_output = 1
                if return_matches:
                    x_match = total_x[index_enc_output]
                    index_enc_output +=1
                if return_inter_reps:
                    inter_views = total_x[index_enc_output]
                    index_enc_output +=1
                if return_order:
                    outer_views = total_x[index_enc_output]
                    #Unfold them to make pairs of 3.
                    outer_views = einops.rearrange(outer_views.unfold(1,3,1), "b outer inner mod ch f folds-> (b outer inner) (mod ch f folds)")
                    x_order = self.fc_out_order(outer_views)

            if len(x.shape) > 2:
                x = x.flatten(start_dim=0, end_dim=1).flatten(start_dim=1)
            x = self.fc_out(x)

            output = {"preds": {"combined": x}}
            if return_matches:
                output["matches"] = x_match
            if return_inter_reps:
                output["inter_reps"] = inter_views
            if return_order:
                output["order"] = x_order

            return output
class Transformer_Sleep_Reconstruct_GM(nn.Module):

        def __init__(self, encs=[None], args=None):
            """
            :param encoder_filters_small, encoder_filters_big: filters of CNN output, quantized by 20
            :param encs_small, encs_big:
            """
            super().__init__()

            d_model = args.dmodel  # 64*8
            fc_inner = args.fc_inner
            num_classes = args.num_classes

            if "training_type" in args and \
                args.training_type == "alignment_order" or args.training_type == "alignment_order_multisupervised" \
                and "multi_loss" in args and \
                "multi_loss_weights" in args.multi_loss and \
                args.multi_loss.multi_loss_weights["order_loss"]!=0:


                self.fc_out_order = nn.Sequential(
                    # nn.BatchNorm1d(d_model),
                    nn.Linear(d_model * 3, fc_inner),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(fc_inner, fc_inner),
                    nn.ReLU(),
                    nn.Dropout(0.1),

                    # nn.Dropout(0.45),
                    nn.Linear(fc_inner, 2),
                    nn.Softmax(dim=1)
                )


            self.args = args
            self.num_encoders = 0
            for i, enc in enumerate(encs):
                if enc != None:
                    setattr(self, "enc_{}".format(i), enc)
                    self.num_encoders += 1

            self.fc_out = nn.Sequential(
                # nn.BatchNorm1d(d_model),
                nn.Linear(d_model, fc_inner),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(fc_inner, fc_inner),
                nn.ReLU(),
                nn.Dropout(0.1),

                # nn.Dropout(0.45),
                nn.Linear(fc_inner, num_classes),
                nn.Softmax(dim=1)
            )


        def forward(self, x, return_matches=False, extract_norm=False, return_inter_reps=False, return_final_reps=False, return_order=False):

            for enc_i in range(self.num_encoders):
                enc = getattr(self, "enc_{}".format(enc_i))
                # total_x -> x_features, x_matches, intermediate_views, intermediate_outer_views
                total_x = enc(x, return_matches=return_matches, extract_norm=extract_norm, return_inter_reps=return_inter_reps, return_order=return_order)
                x = total_x[0]
                index_enc_output = 1
                if return_matches:
                    x_match = total_x[index_enc_output]
                    index_enc_output +=1
                if return_inter_reps:
                    inter_views = total_x[index_enc_output]
                    index_enc_output +=1
                if return_order:
                    outer_views = total_x[index_enc_output]
                    index_enc_output +=1
                    #Unfold them to make pairs of 3.
                    outer_views = einops.rearrange(outer_views.unfold(1,3,1), "b outer inner mod ch f folds-> (b outer inner) (mod ch f folds)")
                    x_order = self.fc_out_order(outer_views)


            if len(x.shape) > 2:
                x = x.flatten(start_dim=0, end_dim=1).flatten(start_dim=1)
            x = self.fc_out(x)

            output = {"preds": {"combined": x}}
            if return_matches:
                output["matches"] = x_match
            if return_inter_reps:
                output["inter_reps"] = inter_views
            if return_order:
                output["order"] = x_order

            return output
class Transformer_Router_GM(nn.Module):

        def __init__(self, encs=[None], args=None):
            """
            :param encoder_filters_small, encoder_filters_big: filters of CNN output, quantized by 20
            :param encs_small, encs_big:
            """
            super().__init__()

            self.enc_0 = encs[0]
            self.enc_1 = encs[1]
            self.enc_2 = encs[2]
            self.enc_3 = encs[3]


            d_model = args.dmodel  # 64*8
            fc_inner = args.fc_inner
            num_classes = 3

            self.fc_out = nn.Sequential(
                # nn.BatchNorm1d(d_model),
                nn.Linear(d_model, fc_inner),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(fc_inner, fc_inner),
                nn.ReLU(),
                nn.Dropout(0.1),

                # nn.Dropout(0.45),
                nn.Linear(fc_inner, num_classes),
                nn.Softmax(dim=1)
            )


        def forward(self, data, return_matches=False, extract_norm=False, return_inter_reps=False, return_order=False):

            pred_mm = self.enc_1(data)
            pred_eeg = self.enc_2([data[0]])
            pred_eog = self.enc_3([data[1]])

            x = self.enc_0(data)
            x = self.fc_out(x[0].flatten(start_dim=2).flatten(start_dim=0, end_dim=1)).argmax(-1).argmax(-1)

            pred_mm["preds"]["combined"][x==1] = pred_eeg[x==1]
            pred_mm["preds"]["combined"][x==2] = pred_eog[x==2]

            output = {"preds": {"combined": pred_mm}}

            return output

class Transformer_Sleep_GM_SepFC(nn.Module):

        def __init__(self, encs=[None], args=None):
            """
            :param encoder_filters_small, encoder_filters_big: filters of CNN output, quantized by 20
            :param encs_small, encs_big:
            """
            super().__init__()

            d_model = args.dmodel  # 64*8
            fc_inner = args.fc_inner
            num_classes = args.num_classes

            if "training_type" in args and \
                args.training_type == "alignment_order" \
                and "multi_loss" in args and \
                "multi_loss_weights" in args.multi_loss and \
                args.multi_loss.multi_loss_weights["order_loss"]!=0:

                self.fc_out_order = nn.Sequential(
                    # nn.BatchNorm1d(d_model),
                    nn.Linear(d_model * 3, fc_inner),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(fc_inner, fc_inner),
                    nn.ReLU(),
                    nn.Dropout(0.1),

                    # nn.Dropout(0.45),
                    nn.Linear(fc_inner, 2),
                    nn.Softmax(dim=1)
                )


            self.args = args
            self.num_encoders = 0
            for i, enc in enumerate(encs):
                if enc != None:
                    setattr(self, "enc_{}".format(i), enc)
                    self.num_encoders += 1

            self.fc_out_0 = nn.Sequential(
                # nn.BatchNorm1d(d_model),
                nn.Linear(d_model, fc_inner),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(fc_inner, fc_inner),
                nn.ReLU(),
                nn.Dropout(0.1),

                # nn.Dropout(0.45),
                nn.Linear(fc_inner, num_classes),
                nn.Softmax(dim=1)
            )

            self.fc_out_1 = nn.Sequential(
                # nn.BatchNorm1d(d_model),
                nn.Linear(d_model, fc_inner),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(fc_inner, fc_inner),
                nn.ReLU(),
                nn.Dropout(0.1),

                # nn.Dropout(0.45),
                nn.Linear(fc_inner, num_classes),
                nn.Softmax(dim=1)
            )


        def forward(self, x, return_matches=False, extract_norm=False, return_inter_reps=False, return_order=False):

            for enc_i in range(self.num_encoders):
                enc = getattr(self, "enc_{}".format(enc_i))
                # total_x -> x_features, x_matches, intermediate_views, intermediate_outer_views
                total_x = enc(x, return_matches=return_matches, extract_norm=extract_norm, return_inter_reps=return_inter_reps, return_order=return_order)
                x = total_x[0]
                index_enc_output = 1
                if return_matches:
                    x_match = total_x[index_enc_output]
                    index_enc_output +=1
                if return_inter_reps:
                    inter_views = total_x[index_enc_output]
                    index_enc_output +=1
                if return_order:
                    outer_views = total_x[index_enc_output]
                    #Unfold them to make pairs of 3.
                    outer_views = einops.rearrange(outer_views.unfold(1,3,1), "b outer inner mod ch f folds-> (b outer inner) (mod ch f folds)")
                    x_order = self.fc_out_order(outer_views)

            if len(x[0].shape) > 2:
                x[0] = x[0].flatten(start_dim=0, end_dim=1).flatten(start_dim=1)
            if len(x[1].shape) > 2:
                x[1] = x[1].flatten(start_dim=0, end_dim=1).flatten(start_dim=1)
            x[0] = self.fc_out_0(x[0])
            x[1] = self.fc_out_1(x[1])

            output = {"preds":{"eeg":x[0], "eog":x[1]}}
            if return_matches:
                output["matches"] = x_match
            if return_inter_reps:
                output["inter_reps"] = inter_views
            if return_order:
                output["order"] = x_order

            return output
class Transformer_Sleep_GM_SepFC_Comb(nn.Module):

        def __init__(self, encs=[None], args=None):
            """
            :param encoder_filters_small, encoder_filters_big: filters of CNN output, quantized by 20
            :param encs_small, encs_big:
            """
            super().__init__()

            d_model = args.dmodel  # 64*8
            fc_inner = args.fc_inner
            num_classes = args.num_classes

            if "training_type" in args and \
                args.training_type == "alignment_order" \
                and "multi_loss" in args and \
                "multi_loss_weights" in args.multi_loss and \
                args.multi_loss.multi_loss_weights["order_loss"]!=0:

                self.fc_out_order = nn.Sequential(
                    # nn.BatchNorm1d(d_model),
                    nn.Linear(d_model * 3, fc_inner),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(fc_inner, fc_inner),
                    nn.ReLU(),
                    nn.Dropout(0.1),

                    # nn.Dropout(0.45),
                    nn.Linear(fc_inner, 2)
                )


            self.args = args
            self.num_encoders = 0
            for i, enc in enumerate(encs):
                if enc != None:
                    setattr(self, "enc_{}".format(i), enc)
                    self.num_encoders += 1

            self.fc_out_0 = nn.Sequential(
                # nn.BatchNorm1d(d_model),
                nn.Linear(d_model, fc_inner),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(fc_inner, fc_inner),
                nn.ReLU(),
                nn.Dropout(0.1),

                # nn.Dropout(0.45),
                nn.Linear(fc_inner, num_classes)
            )

            self.fc_out_1 = nn.Sequential(
                # nn.BatchNorm1d(d_model),
                nn.Linear(d_model, fc_inner),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(fc_inner, fc_inner),
                nn.ReLU(),
                nn.Dropout(0.1),

                # nn.Dropout(0.45),
                nn.Linear(fc_inner, num_classes)
            )

            self.fc_out_combined = nn.Sequential(
                # nn.BatchNorm1d(d_model),
                nn.Linear(d_model*2, fc_inner),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(fc_inner, fc_inner),
                nn.ReLU(),
                nn.Dropout(0.1),

                # nn.Dropout(0.45),
                nn.Linear(fc_inner, num_classes)
                # nn.Softmax(dim=1)
            )


        def forward(self, x, return_matches=False, extract_norm=False, return_inter_reps=False, return_order=False):

            for enc_i in range(self.num_encoders):
                enc = getattr(self, "enc_{}".format(enc_i))
                # total_x -> x_features, x_matches, intermediate_views, intermediate_outer_views
                x = enc(x, return_matches=return_matches, extract_norm=extract_norm, return_inter_reps=return_inter_reps, return_order=return_order)

            if len(x["outuput_features"]["eeg"].shape) > 2:
                x["outuput_features"]["eeg"] = x["outuput_features"]["eeg"].flatten(start_dim=0, end_dim=1).flatten(start_dim=1)
            if len(x["outuput_features"]["eog"].shape) > 2:
                x["outuput_features"]["eog"] = x["outuput_features"]["eog"].flatten(start_dim=0, end_dim=1).flatten(start_dim=1)
            if len(x["outuput_features"]["combined"].shape) > 2:
                x["outuput_features"]["combined"] = x["outuput_features"]["combined"].flatten(start_dim=0, end_dim=1).flatten(start_dim=1)

            x["outuput_features"]["eeg"] = self.fc_out_0(x["outuput_features"]["eeg"])
            x["outuput_features"]["eog"] = self.fc_out_1(x["outuput_features"]["eog"])
            x["outuput_features"]["combined"] = self.fc_out_combined(x["outuput_features"]["combined"])

            output = {"preds":x["outuput_features"]}
            if return_matches:
                output["matches"] = x["matches"]
            if return_inter_reps:
                output["inter_reps"] = x["inter_views"]
            if return_order:
                x["order"] = einops.rearrange(x["order"].unfold(1, 3, 1),
                                               "b outer inner mod ch f folds-> (b outer inner) (mod ch f folds)")
                x["order"] = self.fc_out_order(x["order"])
                output["order"] = x["order"]

            return output
class SleepEnc_Simple_EEG_EOG(nn.Module):
    def __init__(self, args):
        """
        :param encoder_filters_small, encoder_filters_big: filters of CNN output, quantized by 20
        :param encs_small, encs_big:
        """
        super().__init__()
        self.args = args

        d_model = args.dmodel  # 64*8

        inner_biased = args.inner_biased if "inner_biased" in args else False
        outer_biased = args.outer_biased if "outer_biased" in args else False
        rpos = args.rpos if "rpos" in args else False
        self.outer_rep = args.outer_rep if "outer_rep" in args else False

        if inner_biased == "gaussian_learned":
            inner_biased = Gaussian_Learned_Attention_Bias(d_model, heads=8, type="mul")

        # if outer_biased == "gaussian_learned":
        #     outer_biased = Gaussian_Learned_Attention_Bias(d_model, heads=8, type="mul")

        self.inner_tf_eeg = inner_ch_att_RA(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, gbiased=inner_biased, num_layers=4)
        self.inner_tf_eog = inner_ch_att_RA(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, gbiased=inner_biased, num_layers=4)

        self.outer_tf = outer_mod_att_RA(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, gbiased=outer_biased, num_layers=4)

        self.cls_token = nn.Parameter(torch.randn(1, 1, 1, 1, 1, d_model))

        self.inner_positional_embedding = huy_pos_inner(d_model, pos=False, inner=29, outer=21, modalities=1,
                                                          channels=1)
        self.outer_positional_embedding = huy_pos_outer(d_model, pos=False, inner=29, outer=21, modalities=1,
                                                        channels=1)

    def forward(self, x, return_matches=False, extract_norm=False, return_inter_reps=False, return_order=False):
        xeeg = x[0][:, :, :, :, 1:, :]  # mat
        xeog = x[1][:, :, :, :, 1:, :]  # mat

        xeeg = einops.rearrange(xeeg, "b outer mod ch f inner -> b outer inner mod ch f")
        xeog = einops.rearrange(xeog, "b outer mod ch f inner -> b outer inner mod ch f")

        cls_token_eeg = self.cls_token.repeat(xeeg.shape[0], xeeg.shape[1], 1, 1, xeeg.shape[3], 1)
        xeeg = torch.cat([cls_token_eeg, xeeg], dim=2)

        cls_token_eog = self.cls_token.repeat(xeog.shape[0], xeog.shape[1], 1, 1, xeog.shape[3], 1)
        xeog = torch.cat([cls_token_eog, xeog], dim=2)

        xeeg_sa = self.inner_tf_eeg(xeeg)
        xeog_sa = self.inner_tf_eog(xeog)

        x = torch.cat([xeeg_sa[:, :, :1], xeeg_sa[:, :, :1]], dim=3)
        # x = self.outer_positional_embedding(x)
        x = self.outer_tf(x, extract_norm=extract_norm)

        xeeg_sa_o, xeog_sa_o = x[:,:,:,:1], x[:,:,:,1:]

        output = [x]
        if return_matches:
            x_match = torch.einsum('b o f , b m f -> b o m', xeeg_sa_o.squeeze(), xeog_sa_o.squeeze())
            output.append(x_match)
        if return_inter_reps:
            output.append([xeeg_sa, xeog_sa])
        if return_order:
            x_sa_o = torch.cat([xeeg_sa_o, xeog_sa_o], dim=3)
            output.append(x_sa_o)

        return output
class SleepEnc_BLIP_EEG_EOG(nn.Module):
    def __init__(self, args, encs=[None]):
        """
        :param encoder_filters_small, encoder_filters_big: filters of CNN output, quantized by 20
        :param encs_small, encs_big:
        """
        super().__init__()
        self.args = args

        d_model = args.dmodel  # 64*8

        inner_biased = args.inner_biased if "inner_biased" in args else False
        outer_biased = args.outer_biased if "outer_biased" in args else False
        rpos = args.rpos if "rpos" in args else False
        self.pos = args.pos if "pos" in args else True
        dim_proj = args.dim_proj if "dim_proj" in args else 128
        self.outer_rep = args.outer_rep if "outer_rep" in args else False

        if inner_biased == "gaussian_learned":
            inner_biased = Gaussian_Learned_Attention_Bias(d_model, heads=8, type="mul")

        # if outer_biased == "gaussian_learned":
        #     outer_biased = Gaussian_Learned_Attention_Bias(d_model, heads=8, type="mul")

        self.inner_tf_eeg = inner_ch_SA_CA(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, dim_proj=dim_proj, gbiased=inner_biased, num_layers=4)
        self.inner_tf_eog = inner_ch_SA_CA(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, dim_proj=dim_proj, gbiased=inner_biased, num_layers=4)

        if self.outer_rep:
            self.outer_tf_eeg = outer_mod_att_RA(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, dim_proj=dim_proj, gbiased=outer_biased, num_layers=4)
            self.outer_tf_eog = outer_mod_att_RA(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, dim_proj=dim_proj,gbiased=outer_biased, num_layers=4)

        self.outer_tf = outer_mod_att_RA(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, dim_proj=dim_proj, gbiased=outer_biased, num_layers=4)

        self.cls_token = nn.Parameter(torch.randn(1, 1, 1, 1, 1, d_model))

        if self.pos:
            self.inner_positional_embedding = huy_pos_inner(d_model, pos=False, inner=29, outer=21, modalities=1,
                                                              channels=1)
            self.outer_positional_embedding = huy_pos_outer(d_model, pos=False, inner=29, outer=21, modalities=1,
                                                            channels=1)

    def forward(self, x, return_matches=False, extract_norm=False, return_inter_reps=False, return_order=False):
        xeeg = x[0][:, :, :, :, 1:, :]  # mat
        xeog = x[1][:, :, :, :, 1:, :]  # mat

        xeeg = einops.rearrange(xeeg, "b outer mod ch f inner -> b outer inner mod ch f")
        xeog = einops.rearrange(xeog, "b outer mod ch f inner -> b outer inner mod ch f")

        if self.pos:
            xeeg = self.inner_positional_embedding(xeeg)
            xeog = self.inner_positional_embedding(xeog)

        cls_token_eeg = self.cls_token.repeat(xeeg.shape[0], xeeg.shape[1], 1, 1, xeeg.shape[3], 1)
        xeeg = torch.cat([cls_token_eeg, xeeg], dim=2)

        cls_token_eog = self.cls_token.repeat(xeog.shape[0], xeog.shape[1], 1, 1, xeog.shape[3], 1)
        xeog = torch.cat([cls_token_eog, xeog], dim=2)

        xeeg_sa = self.inner_tf_eeg.forward_sa(xeeg)
        xeog_sa = self.inner_tf_eog.forward_sa(xeog)

        xeeg_sa_o, xeog_sa_o = xeeg_sa[:, :, :1], xeog_sa[:, :, :1]
        if self.outer_rep:
            # xeeg_sa_o = self.outer_positional_embedding(xeeg_sa_o)
            # xeog_sa_o = self.outer_positional_embedding(xeog_sa_o)
            xeeg_sa_o = self.outer_tf_eeg(xeeg_sa_o, extract_norm=extract_norm)
            xeog_sa_o = self.outer_tf_eog(xeog_sa_o, extract_norm=extract_norm)

        xeeg_ca = self.inner_tf_eeg.forward_ca(xeeg, xeog_sa)
        xeog_ca = self.inner_tf_eog.forward_ca(xeog, xeeg_sa)

        x = torch.cat([xeeg_ca[:, :, :1], xeog_ca[:, :, :1]], dim=3)
        if self.pos:
            x = self.outer_positional_embedding(x)
        x = self.outer_tf(x, extract_norm=extract_norm)

        output = [x]
        if return_matches:
            x_match = torch.einsum('b o f , b m f -> b o m', xeeg_sa_o.squeeze(), xeog_sa_o.squeeze())
            output.append(x_match)
        if return_inter_reps:
            output.append([xeeg_sa_o, xeeg_sa_o])
        if return_order:
            x_sa_o = torch.cat([xeeg_sa_o, xeog_sa_o], dim=3)
            output.append(x_sa_o)

        return output
class SleepEnc_Early_Summation_EEG_EOG(nn.Module):
    def __init__(self, args, encs=[None]):
        """
        :param encoder_filters_small, encoder_filters_big: filters of CNN output, quantized by 20
        :param encs_small, encs_big:
        """
        super().__init__()
        self.args = args

        d_model = args.dmodel  # 64*8

        inner_biased = args.inner_biased if "inner_biased" in args else False
        outer_biased = args.outer_biased if "outer_biased" in args else False
        rpos = args.rpos if "rpos" in args else False
        self.pos = args.pos if "pos" in args else True
        dim_proj = args.dim_proj if "dim_proj" in args else 128
        self.outer_rep = args.outer_rep if "outer_rep" in args else False

        if inner_biased == "gaussian_learned":
            inner_biased = Gaussian_Learned_Attention_Bias(d_model, heads=8, type="mul")

        # if outer_biased == "gaussian_learned":
        #     outer_biased = Gaussian_Learned_Attention_Bias(d_model, heads=8, type="mul")

        self.inner_tf = inner_ch_att_RA(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, dim_proj=dim_proj, gbiased=inner_biased, num_layers=4)

        self.outer_tf = outer_mod_att_RA(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, dim_proj=dim_proj, gbiased=outer_biased, num_layers=4)

        self.cls_token = nn.Parameter(torch.randn(1, 1, 1, 1, 1, d_model))

        if self.pos:
            self.inner_positional_embedding = huy_pos_inner(d_model, pos=False, inner=29, outer=21, modalities=1,
                                                              channels=1)
            self.outer_positional_embedding = huy_pos_outer(d_model, pos=False, inner=29, outer=21, modalities=1,
                                                            channels=1)

    def forward(self, x, return_matches=False, extract_norm=False, return_inter_reps=False, return_order=False):
        xeeg = x[0][:, :, :, :, 1:, :]  # mat
        xeog = x[1][:, :, :, :, 1:, :]  # mat

        xeeg = einops.rearrange(xeeg, "b outer mod ch f inner -> b outer inner mod ch f")
        xeog = einops.rearrange(xeog, "b outer mod ch f inner -> b outer inner mod ch f")

        x = xeeg + xeog
        if self.pos:
            x = self.inner_positional_embedding(x)

        cls_token = self.cls_token.repeat(x.shape[0], x.shape[1], 1, 1, x.shape[3], 1)
        x = torch.cat([cls_token, x], dim=2)

        x = self.inner_tf(x)

        x = x[:, :, :1]

        if self.pos:
            x = self.outer_positional_embedding(x)
        x = self.outer_tf(x, extract_norm=extract_norm)

        output = {"output_features":x}
        if return_matches:
            raise NotImplementedError
        if return_inter_reps:
            raise NotImplementedError
        if return_order:
            raise NotImplementedError

        return output
class SleepEnc_Early_Concat_EEG_EOG_modtype(nn.Module):
    def __init__(self, args, encs=[None]):
        """
        :param encoder_filters_small, encoder_filters_big: filters of CNN output, quantized by 20
        :param encs_small, encs_big:
        """
        super().__init__()
        self.args = args

        d_model = args.dmodel  # 64*8

        inner_biased = args.inner_biased if "inner_biased" in args else False
        outer_biased = args.outer_biased if "outer_biased" in args else False
        rpos = args.rpos if "rpos" in args else False
        self.pos = args.pos if "pos" in args else True
        dim_proj = args.dim_proj if "dim_proj" in args else 128
        self.outer_rep = args.outer_rep if "outer_rep" in args else False

        if inner_biased == "gaussian_learned":
            inner_biased = Gaussian_Learned_Attention_Bias(d_model, heads=8, type="mul")

        # if outer_biased == "gaussian_learned":
        #     outer_biased = Gaussian_Learned_Attention_Bias(d_model, heads=8, type="mul")

        self.inner_tf = inner_mod_ch_att_RA(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, dim_proj=dim_proj, gbiased=inner_biased, num_layers=4)

        self.outer_tf = outer_mod_att_RA(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, dim_proj=dim_proj, gbiased=outer_biased, num_layers=4)

        self.cls_token = nn.Parameter(torch.randn(1, 1, 1, 1, 1, d_model))

        self.modtype_token = modtype_embedding(num_modalities=2, dim=d_model)

        if self.pos:
            self.inner_positional_embedding = huy_pos_inner(d_model, pos=False, inner=29, outer=21, modalities=1, channels=1)
            self.outer_positional_embedding = huy_pos_outer(d_model, pos=False, inner=29, outer=21, modalities=1, channels=1)

    def forward(self, x, **kwargs):
        xeeg = x["stft_eeg"][:, :, :, :, 1:, :]  # mat
        xeog = x["stft_eog"][:, :, :, :, 1:, :]  # mat

        xeeg = einops.rearrange(xeeg, "b outer mod ch f inner -> b outer inner mod ch f")
        xeog = einops.rearrange(xeog, "b outer mod ch f inner -> b outer inner mod ch f")

        xeeg = self.modtype_token(data=xeeg, mod_num=0)
        xeog = self.modtype_token(data=xeog, mod_num=1)

        if self.pos:
            xeeg = self.inner_positional_embedding(xeeg)
            xeog = self.inner_positional_embedding(xeog)

        x = torch.cat([xeeg, xeog], dim=2)

        cls_token = self.cls_token.repeat(x.shape[0], x.shape[1], 1, x.shape[3], x.shape[4], 1)
        x = torch.cat([cls_token, x], dim=2)

        x = self.inner_tf(x)

        x = x[:, :, :1]

        if self.pos:
            x = self.outer_positional_embedding(x)
        x = self.outer_tf(x, **kwargs)

        output = {"output_features":x}
        return output
class SleepEnc_Early_Concat_EEG_EOG_modtype_onlyi(nn.Module):
    def __init__(self, args, encs=[None]):
        """
        :param encoder_filters_small, encoder_filters_big: filters of CNN output, quantized by 20
        :param encs_small, encs_big:
        """
        super().__init__()
        self.args = args

        d_model = args.dmodel  # 64*8

        inner_biased = args.inner_biased if "inner_biased" in args else False
        outer_biased = args.outer_biased if "outer_biased" in args else False
        rpos = args.rpos if "rpos" in args else False
        self.pos = args.pos if "pos" in args else True
        dim_proj = args.dim_proj if "dim_proj" in args else 128
        self.outer_rep = args.outer_rep if "outer_rep" in args else False

        if inner_biased == "gaussian_learned":
            inner_biased = Gaussian_Learned_Attention_Bias(d_model, heads=8, type="mul")

        # if outer_biased == "gaussian_learned":
        #     outer_biased = Gaussian_Learned_Attention_Bias(d_model, heads=8, type="mul")

        self.inner_tf = inner_att_RA(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, dim_proj=dim_proj, gbiased=inner_biased, num_layers=4)

        self.outer_tf = outer_mod_att_RA(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, dim_proj=dim_proj, gbiased=outer_biased, num_layers=4)

        self.cls_token_i = nn.Parameter(torch.randn(1, 1, 1, 1, 1, d_model))
        self.cls_token_o = nn.Parameter(torch.randn(1, 1, 1, 1, 1, d_model))

        self.modtype_token = modtype_embedding(num_modalities=2, dim=d_model)

        if self.pos:
            self.inner_positional_embedding = huy_pos_inner(d_model, pos=False, inner=29, outer=21, modalities=1, channels=1)
            self.outer_positional_embedding = huy_pos_outer(d_model, pos=False, inner=29, outer=21, modalities=1, channels=1)

    def forward(self, x, skip_modality=None, **kwargs):
        xeeg = x["stft_eeg"][:, :, :, :, 1:, :]  # mat
        xeog = x["stft_eog"][:, :, :, :, 1:, :]  # mat

        xeeg = einops.rearrange(xeeg, "b outer mod ch f inner -> b outer inner mod ch f")
        xeog = einops.rearrange(xeog, "b outer mod ch f inner -> b outer inner mod ch f")

        xeeg = self.modtype_token(data=xeeg, mod_num=0)
        xeog = self.modtype_token(data=xeog, mod_num=1)

        if self.pos:
            xeeg = self.inner_positional_embedding(xeeg)
            xeog = self.inner_positional_embedding(xeog)

        xeeg_common = xeeg
        xeog_common = xeog

        skip_modality = self.calculate_skip_modality( num_batches=xeeg.shape[0], outer_size=xeeg.shape[1], skip_modality=skip_modality)
        if skip_modality is not None:
            xeeg_common = xeeg[skip_modality==0] #Keep epochs where both mods are available
            xeog_common = xeog[skip_modality==0]

            xeeg_sole = xeeg[skip_modality==2] #Keep epochs where eog is not available
            xeog_sole = xeog[skip_modality==1] #Keep epochs where eeg is not available

            [xeeg_common, xeog_common, xeeg_sole, xeog_sole] = self.reshape_batchouter([xeeg_common, xeog_common, xeeg_sole, xeog_sole], outer_size=xeeg.shape[1])

        else:
            skip_modality = torch.zeros(xeeg.shape[0])

        x = torch.empty(xeeg.shape)[:, :, :1].to(xeeg.device)
        output = {"output_features": x}
        if xeeg_common.shape[0]>0 and xeog_common.shape[0]>0:
            output = self.forward_common(xeeg=xeeg_common, xeog=xeog_common, skip_modality=skip_modality, output=output, **kwargs)
        if "xeeg_sole" in locals() and xeeg_sole.shape[0]>0:
            output = self.forward_eeg(xeeg=xeeg_sole, skip_modality=skip_modality, output=output)
        if "xeog_sole" in locals() and xeog_sole.shape[0]>0:
            output = self.forward_eog(xeog=xeog_sole, skip_modality=skip_modality, output=output)
        return output

    def calculate_skip_modality(self, num_batches, outer_size, skip_modality):
        if  self.training:
            skip_modality = torch.rand(num_batches)
            skip_modality[skip_modality > 0.75] = 2
            skip_modality[skip_modality < 0.25] = 1
            skip_modality = skip_modality.int()
        elif skip_modality is not None:
            if skip_modality=="full":
                skip_modality = None
            elif skip_modality=="random":
                skip_modality = torch.rand(num_batches)
                skip_modality[skip_modality > 0.66] = 2
                skip_modality[skip_modality < 0.33] = 1
                skip_modality = skip_modality.int()
            elif skip_modality=="eeg":
                skip_modality = torch.ones(num_batches)*1
            elif skip_modality=="eog":
                skip_modality = torch.ones(num_batches)*2
            elif "percentile" in skip_modality:
                keep_eeg = skip_modality.split("_")[0]
                keep_eog = skip_modality.split("_")[1]

                skip_modality_eeg = torch.rand(num_batches)
                skip_modality_eog = torch.rand(num_batches)

                skip_modality_eeg[skip_modality_eeg > keep_eeg] = 0
                skip_modality_eeg[skip_modality_eeg <= keep_eeg] = 1

                kept_eeg = len(skip_modality_eeg[skip_modality_eeg == 0])

                keep_eog = keep_eog/(kept_eeg/len(skip_modality_eeg))

                skip_modality_eog[skip_modality_eog > keep_eeg] = 0
                skip_modality_eog[skip_modality_eog <= keep_eeg] = 2

                skip_modality_eog = torch.rand(num_batches)

                skip_modality = torch.ones(num_batches)*2

            elif skip_modality is "vae":
                raise not NotImplementedError
            else:
                skip_modality_temp = torch.zeros(num_batches, outer_size)
                skip_modality_temp[skip_modality["stft_eeg"] == 1] = 1
                skip_modality_temp[skip_modality["stft_eog"] == 1] = 2
                skip_modality = skip_modality_temp
                # with torch.no_grad():
                #     vae_eeg_output = self.enc_0([x[0]])
                #     output_losses = self.enc_0.module.loss_function(vae_eeg_output[0], vae_eeg_output[1],
                #                                                     vae_eeg_output[2], vae_eeg_output[3],
                #                                                     reduction="none")
                #     eeg_routing = output_losses["total"].flatten(start_dim=1).mean(dim=1)
                #     # print(eeg_routing)
                #     eeg_routing[eeg_routing < 3] = 0
                #     eeg_routing[eeg_routing > 3] = 1
                #
                #     vae_eog_output = self.enc_1([x[1]])
                #     output_losses = self.enc_1.module.loss_function(vae_eog_output[0], vae_eog_output[1],
                #                                                     vae_eog_output[2], vae_eog_output[3],
                #                                                     reduction="none")
                #     eog_routing = output_losses["total"].flatten(start_dim=1).mean(dim=1)
                #     # print(eog_routing)
                #     eog_routing[eog_routing < 3] = 0
                #     eog_routing[eog_routing > 3] = 2
                #
                #     skip_modality = eeg_routing + eog_routing
                #     skip_modality[skip_modality == 3] = 0
        return skip_modality

    def forward_common(self, xeeg, xeog, skip_modality, output, **kwargs):

        x = torch.cat([xeeg, xeog], dim=4)

        cls_token = self.cls_token_i.repeat(x.shape[0], x.shape[1], 1, x.shape[3], x.shape[4], 1)
        x = torch.cat([cls_token, x], dim=2)

        x = self.inner_tf(x)

        x = x[:, :, :1]

        cls_token = self.cls_token_o.repeat(x.shape[0], x.shape[1], 1, 1, 1, 1)
        x = torch.cat([cls_token, x], dim=4)

        if self.pos:
            x = self.outer_positional_embedding(x)
        x = self.outer_tf(x, **kwargs)[:, :, :, :, :1]

        output["output_features"][skip_modality==0] = x
        return output
    def forward_eeg(self, xeeg, skip_modality, output):

        x = xeeg

        cls_token = self.cls_token_i.repeat(x.shape[0], x.shape[1], 1, x.shape[3], x.shape[4], 1)
        x = torch.cat([cls_token, x], dim=2)

        x = self.inner_tf(x)

        x = x[:, :, :1]

        cls_token = self.cls_token_o.repeat(x.shape[0], x.shape[1], 1, 1, 1, 1)
        x = torch.cat([cls_token, x], dim=4)

        if self.pos:
            x = self.outer_positional_embedding(x)
        x = self.outer_tf(x)[:, :, :, :, :1]
        output["output_features"][skip_modality == 2] = x.flatten(start_dim=0, end_dim=1)
        return output
    def forward_eog(self, xeog, skip_modality, output):
        x = xeog
        cls_token = self.cls_token_i.repeat(x.shape[0], x.shape[1], 1, x.shape[3], x.shape[4], 1)
        x = torch.cat([cls_token, x], dim=2)

        x = self.inner_tf(x)

        x = x[:, :, :1]

        cls_token = self.cls_token_o.repeat(x.shape[0], x.shape[1], 1, 1, 1, 1)
        x = torch.cat([cls_token, x], dim=4)

        if self.pos:
            x = self.outer_positional_embedding(x)
        x = self.outer_tf(x)[:, :, :, :, :1]
        output["output_features"][skip_modality == 1] = x.flatten(start_dim=0, end_dim=1)
        return output
    def reshape_batchouter(self, list_of_tensors, outer_size):
        output_tens = []
        for tens in list_of_tensors:
            if tens.shape[0]>0:
                batch_size = tens.shape[0]/outer_size
                if batch_size%1 !=0: raise ValueError("split_modalities should be the same for every outer in each batch sample. (b o) shape was {}".format(tens.shape[0]))
                tens = einops.rearrange(tens, "(b o) i m c f -> b o i m c f", b=int(batch_size), o=outer_size)
            output_tens.append(tens)
        return output_tens


class SleepEnc_Early_Concat_EEG_EOG_modtype_seponlyi(nn.Module):
    def __init__(self, args, encs=[None]):
        """
        :param encoder_filters_small, encoder_filters_big: filters of CNN output, quantized by 20
        :param encs_small, encs_big:
        """
        super().__init__()
        self.args = args

        d_model = args.dmodel  # 64*8

        inner_biased = args.inner_biased if "inner_biased" in args else False
        outer_biased = args.outer_biased if "outer_biased" in args else False
        rpos = args.rpos if "rpos" in args else False
        self.pos = args.pos if "pos" in args else True
        dim_proj = args.dim_proj if "dim_proj" in args else 128
        self.outer_rep = args.outer_rep if "outer_rep" in args else False

        if inner_biased == "gaussian_learned":
            inner_biased = Gaussian_Learned_Attention_Bias(d_model, heads=8, type="mul")

        # if outer_biased == "gaussian_learned":
        #     outer_biased = Gaussian_Learned_Attention_Bias(d_model, heads=8, type="mul")

        self.inner_tf_eeg = inner_att_RA(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, dim_proj=dim_proj, gbiased=inner_biased, num_layers=4)
        self.inner_tf_eog = inner_att_RA(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, dim_proj=dim_proj, gbiased=inner_biased, num_layers=4)

        self.outer_tf = outer_mod_att_RA(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, dim_proj=dim_proj, gbiased=outer_biased, num_layers=4)

        self.cls_token_i_eeg = nn.Parameter(torch.randn(1, 1, 1, 1, 1, d_model))
        self.cls_token_i_eog = nn.Parameter(torch.randn(1, 1, 1, 1, 1, d_model))
        self.cls_token_o = nn.Parameter(torch.randn(1, 1, 1, 1, 1, d_model))

        self.modtype_token = modtype_embedding(num_modalities=2, dim=d_model)

        if self.pos:
            self.inner_positional_embedding = huy_pos_inner(d_model, pos=False, inner=29, outer=21, modalities=1, channels=1)
            self.outer_positional_embedding = huy_pos_outer(d_model, pos=False, inner=29, outer=21, modalities=1, channels=1)

    def forward(self, x, **kwargs):
        xeeg = x["stft_eeg"][:, :, :, :, 1:, :]  # mat
        xeog = x["stft_eog"][:, :, :, :, 1:, :]  # mat

        xeeg = einops.rearrange(xeeg, "b outer mod ch f inner -> b outer inner mod ch f")
        xeog = einops.rearrange(xeog, "b outer mod ch f inner -> b outer inner mod ch f")

        if self.pos:
            xeeg = self.inner_positional_embedding(xeeg)
            xeog = self.inner_positional_embedding(xeog)


        cls_token = self.cls_token_i_eeg.repeat(xeeg.shape[0], xeeg.shape[1], 1, 1, xeeg.shape[4], 1)
        xeeg = torch.cat([cls_token, xeeg], dim=2)

        cls_token = self.cls_token_i_eog.repeat(xeog.shape[0], xeog.shape[1], 1, 1, xeog.shape[4], 1)
        xeog = torch.cat([cls_token, xeog], dim=2)

        xeeg = self.inner_tf_eeg(xeeg)[:, :, :1]
        xeog = self.inner_tf_eog(xeog)[:, :, :1]


        xeeg = self.modtype_token(data=xeeg, mod_num=0)
        xeog = self.modtype_token(data=xeog, mod_num=1)
        x = torch.cat([xeeg, xeog], dim=4)

        if self.pos:
            x = self.outer_positional_embedding(x)

        cls_token = self.cls_token_o.repeat(x.shape[0], x.shape[1], 1, 1, 1, 1)
        x = torch.cat([cls_token, x], dim=4)

        x = self.outer_tf(x, **kwargs)[:, :, :, :, :1]

        output = {"output_features":x}
        return output
class SleepEnc_BLIP_EEG_EOG_twomode(nn.Module):
    def __init__(self, args, encs=[None]):
        """
        :param encoder_filters_small, encoder_filters_big: filters of CNN output, quantized by 20
        :param encs_small, encs_big:
        """
        super().__init__()
        self.args = args

        self.enc_0 = encs[0]
        self.enc_1 = encs[0]

        d_model = args.dmodel  # 64*8

        inner_biased = args.inner_biased if "inner_biased" in args else False
        outer_biased = args.outer_biased if "outer_biased" in args else False
        rpos = args.rpos if "rpos" in args else False
        self.pos = args.pos if "pos" in args else True
        dim_proj = args.dim_proj if "dim_proj" in args else 128
        self.outer_rep = args.outer_rep if "outer_rep" in args else False
        self.skip_random_mod = args.skip_random_mod if "skip_random_mod" in args else False

        if inner_biased == "gaussian_learned":
            inner_biased = Gaussian_Learned_Attention_Bias(d_model, heads=8, type="mul")

        # if outer_biased == "gaussian_learned":
        #     outer_biased = Gaussian_Learned_Attention_Bias(d_model, heads=8, type="mul")

        self.inner_tf_eeg = inner_ch_SA_CA(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, dim_proj=dim_proj, gbiased=inner_biased, num_layers=4)
        self.inner_tf_eog = inner_ch_SA_CA(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, dim_proj=dim_proj, gbiased=inner_biased, num_layers=4)

        if self.outer_rep:
            self.outer_tf_eeg = outer_mod_att_RA(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, dim_proj=dim_proj, gbiased=outer_biased, num_layers=4)
            self.outer_tf_eog = outer_mod_att_RA(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, dim_proj=dim_proj,gbiased=outer_biased, num_layers=4)

        self.outer_tf = outer_mod_att_RA(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, dim_proj=dim_proj, gbiased=outer_biased, num_layers=4)

        self.cls_token_eeg = nn.Parameter(torch.randn(1, 1, 1, 1, 1, d_model))
        self.cls_token_eog = nn.Parameter(torch.randn(1, 1, 1, 1, 1, d_model))
        self.cls_token_outer = nn.Parameter(torch.randn(1, 1, 1, 1, 1, d_model))

        # if self.skip_random_mod:
        #     self.skip_dropout = nn.Dropout()
        if self.pos:
            self.inner_positional_embedding = huy_pos_inner(d_model, pos=False, inner=29, outer=21, modalities=1,
                                                              channels=1)
            self.outer_positional_embedding = huy_pos_outer(d_model, pos=False, inner=29, outer=21, modalities=1,
                                                            channels=1)

    def forward(self, x, return_matches=False, extract_norm=False, return_inter_reps=False, return_order=False, skip_modality="random"):

        xeeg = x[0][:, :, :, :, 1:, :]  # mat
        xeog = x[1][:, :, :, :, 1:, :]  # mat

        xeeg = einops.rearrange(xeeg, "b outer mod ch f inner -> b outer inner mod ch f")
        xeog = einops.rearrange(xeog, "b outer mod ch f inner -> b outer inner mod ch f")

        if self.pos:
            xeeg = self.inner_positional_embedding(xeeg)
            xeog = self.inner_positional_embedding(xeog)

        cls_token_eeg = self.cls_token_eeg.repeat(xeeg.shape[0], xeeg.shape[1], 1, 1, xeeg.shape[3], 1)
        xeeg = torch.cat([cls_token_eeg, xeeg], dim=2)

        cls_token_eog = self.cls_token_eog.repeat(xeog.shape[0], xeog.shape[1], 1, 1, xeog.shape[3], 1)
        xeog = torch.cat([cls_token_eog, xeog], dim=2)

        xeeg_common_init = xeeg
        xeog_common_init = xeog

        xeeg_common = xeeg
        xeog_common = xeog

        if self.skip_random_mod:
            if  self.training:
                # skip_modality_eeg = torch.bernoulli(torch.ones(xeeg.shape[0])*0.5)
                # skip_modality_eog = torch.bernoulli(torch.ones(xeog.shape[0])*0.5)
                # skip_modality = skip_modality_eeg*2 + skip_modality_eog
                # skip_modality[skip_modality==3] = 0

                skip_modality = torch.rand(xeeg.shape[0])
                skip_modality[skip_modality>0.66] = 2
                skip_modality[skip_modality<0.33] = 1
                skip_modality = skip_modality.int()
                # skip_modality[skip_modality==0]=1

            else:
                if skip_modality=="full":
                    skip_modality = None
                elif skip_modality=="random":
                    skip_modality = torch.rand(xeeg.shape[0])
                    skip_modality[skip_modality > 0.66] = 2
                    skip_modality[skip_modality < 0.33] = 1
                    skip_modality = skip_modality.int()
                    # skip_modality[skip_modality == 0] = 1
                elif skip_modality=="eeg":
                    skip_modality = torch.ones(xeeg.shape[0])*1
                elif skip_modality=="eog":
                    skip_modality = torch.ones(xeeg.shape[0])*2
                elif skip_modality is None:
                    with torch.no_grad():
                        vae_eeg_output = self.enc_0([x[0]])
                        output_losses = self.enc_0.module.loss_function(vae_eeg_output[0], vae_eeg_output[1],
                                                                        vae_eeg_output[2], vae_eeg_output[3],
                                                                        reduction="none")
                        eeg_routing = output_losses["total"].flatten(start_dim=1).mean(dim=1)
                        # print(eeg_routing)
                        eeg_routing[eeg_routing < 3] = 0
                        eeg_routing[eeg_routing > 3] = 1

                        vae_eog_output = self.enc_1([x[1]])
                        output_losses = self.enc_1.module.loss_function(vae_eog_output[0], vae_eog_output[1],
                                                                        vae_eog_output[2], vae_eog_output[3],
                                                                        reduction="none")
                        eog_routing = output_losses["total"].flatten(start_dim=1).mean(dim=1)
                        # print(eog_routing)
                        eog_routing[eog_routing < 3] = 0
                        eog_routing[eog_routing > 3] = 2

                        skip_modality = eeg_routing + eog_routing
                        skip_modality[skip_modality == 3] = 0

            if skip_modality is not None:
                # xeeg_common = xeeg_sa[np.logical_or(skip_modality==0, skip_modality==1)] #Process EEG with EOG if you are not skipping EOG
                xeeg_common = xeeg[skip_modality==0] #Process EEG with EOG if you are not skipping EOG
                xeeg_common_init = xeeg[skip_modality==0] #Process EEG with EOG if you are not skipping EOG
                # xeeg_undir_sole = xeeg_sa[skip_modality==1] #If you skip EOG process EEG on its own
                xeeg_dir_sole = xeeg[skip_modality==2] #If you skip EOG process EEG on its own
                xeog_common = xeog[skip_modality==0] #Process EOG with EEG if you are not skipping EEG
                xeog_common_init = xeog[skip_modality==0] #Process EOG with EEG if you are not skipping EEG
                # xeog_undir_sole = xeog_sa[skip_modality==2] #If you skip EEG process EOG on its own
                xeog_dir_sole = xeog[skip_modality==1] #If you skip EEG process EOG on its own
            # print(skip_modality)

        if xeeg_common.shape[0]>0 and xeog_common.shape[0]>0:
            xeeg_common = self.inner_tf_eeg.forward_sa(xeeg_common)
            xeog_common = self.inner_tf_eog.forward_sa(xeog_common)

            if self.outer_rep:
                # xeeg_sa_o = self.outer_positional_embedding(xeeg_sa_o)
                # xeog_sa_o = self.outer_positional_embedding(xeog_sa_o)
                xeeg_common = self.outer_tf_eeg(xeeg_common, extract_norm=extract_norm)
                xeog_common = self.outer_tf_eog(xeog_common, extract_norm=extract_norm)

            xeeg_sa_o, xeog_sa_o = xeeg_common[:, :, :1].squeeze(), xeog_common[:, :, :1].squeeze()

            xeeg_ca_common = self.inner_tf_eeg.forward_ca(xeeg_common_init, xeog_common)[:, :, :1]
            xeog_ca_common = self.inner_tf_eog.forward_ca(xeog_common_init, xeeg_common)[:, :, :1]
            x_common = torch.cat([xeeg_ca_common, xeog_ca_common], dim=3)
            if self.pos:
                x_common = self.outer_positional_embedding(x_common)
            cls_token_outer = self.cls_token_outer.repeat(x_common.shape[0], x_common.shape[1], 1, 1, 1, 1)
            x_common = torch.cat([cls_token_outer, x_common], dim=3)
            x_common = self.outer_tf(x_common, extract_norm=extract_norm)[:, :, :, :1]

        if "xeeg_dir_sole" in locals() and xeeg_dir_sole.shape[0]>0:
            xeeg_ca_sole = self.inner_tf_eeg.forward_ca(xeeg_dir_sole)[:, :, :1]
            if self.pos:
                xeeg_ca_sole = self.outer_positional_embedding(xeeg_ca_sole)
            # cls_token_outer_eeg = self.cls_token_outer.repeat(xeeg_ca_sole.shape[0], xeeg_ca_sole.shape[1], 1, 1, 1, 1)
            # xeeg_ca_sole = torch.cat([cls_token_outer_eeg, xeeg_ca_sole], dim=3)
            xeeg_ca_sole = self.outer_tf(xeeg_ca_sole, extract_norm=extract_norm)[:, :, :, :1]

        if "xeog_dir_sole" in locals() and xeog_dir_sole.shape[0]>0:
            xeog_ca_sole = self.inner_tf_eeg.forward_ca(xeog_dir_sole)[:, :, :1]
            if self.pos:
                xeog_ca_sole = self.outer_positional_embedding(xeog_ca_sole)
            # cls_token_outer_eog = self.cls_token_outer.repeat(xeog_ca_sole.shape[0], xeog_ca_sole.shape[1], 1, 1, 1, 1)
            # xeog_ca_sole = torch.cat([cls_token_outer_eog, xeog_ca_sole], dim=3)
            xeog_ca_sole = self.outer_tf(xeog_ca_sole, extract_norm=extract_norm)[:, :, :, :1]

        if "xeog_dir_sole" in locals() and "xeog_dir_sole" in locals() and (xeog_dir_sole.shape[0]>0 or xeeg_dir_sole.shape[0]>0):
            x = []
            counter = [0, 0, 0]
            for i in range(len(skip_modality)):
                if skip_modality[i] == 0:
                    x.append(x_common[counter[int(skip_modality[i].item())]:counter[int(skip_modality[i].item())] + 1])
                elif skip_modality[i] == 1:
                    x.append(xeog_ca_sole[
                             counter[int(skip_modality[i].item())]:counter[int(skip_modality[i].item())] + 1])
                elif skip_modality[i] == 2:
                    x.append(xeeg_ca_sole[
                             counter[int(skip_modality[i].item())]:counter[int(skip_modality[i].item())] + 1])
                counter[int(skip_modality[i].item())] += 1
            x = torch.cat(x, dim=0)
        else: x = x_common

        output={"output_features": x}

        if return_matches:
            if ("xeeg_sa_o" in locals() and xeeg_sa_o.shape[0]>0 and len(xeeg_sa_o.shape)==3) and ("xeog_sa_o" in locals() and xeog_sa_o.shape[0]>0 and len(xeog_sa_o.shape)==3):
                x_match = torch.einsum('b o f , b m f -> b o m', xeeg_sa_o, xeog_sa_o)
                output["matches"] = x_match
            else:
                output["matches"] = None
        if return_inter_reps:
            output["intermediate_reps"] = [xeeg_sa_o, xeeg_sa_o]
        if return_order:
            x_sa_o = torch.cat([xeeg_sa_o, xeog_sa_o], dim=3)
            output["order"] =x_sa_o

        return output
class SleepEnc_BLIP_EEG_EOG_twomode_sepouter(nn.Module):
    def __init__(self, args, encs=[None]):
        """
        :param encoder_filters_small, encoder_filters_big: filters of CNN output, quantized by 20
        :param encs_small, encs_big:
        """
        super().__init__()
        self.args = args

        self.enc_0 = encs[0]
        self.enc_1 = encs[0]

        d_model = args.dmodel  # 64*8

        inner_biased = args.inner_biased if "inner_biased" in args else False
        outer_biased = args.outer_biased if "outer_biased" in args else False
        rpos = args.rpos if "rpos" in args else False
        self.pos = args.pos if "pos" in args else True
        dim_proj = args.dim_proj if "dim_proj" in args else 128
        self.outer_rep = args.outer_rep if "outer_rep" in args else False
        self.skip_random_mod = args.skip_random_mod if "skip_random_mod" in args else False

        if inner_biased == "gaussian_learned":
            inner_biased = Gaussian_Learned_Attention_Bias(d_model, heads=8, type="mul")

        # if outer_biased == "gaussian_learned":
        #     outer_biased = Gaussian_Learned_Attention_Bias(d_model, heads=8, type="mul")

        self.inner_tf_eeg = inner_ch_SA_CA(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, dim_proj=dim_proj, gbiased=inner_biased, num_layers=4)
        self.inner_tf_eog = inner_ch_SA_CA(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, dim_proj=dim_proj, gbiased=inner_biased, num_layers=4)

        if self.outer_rep:
            self.outer_tf_eeg = outer_mod_att_RA(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, dim_proj=dim_proj, gbiased=outer_biased, num_layers=4)
            self.outer_tf_eog = outer_mod_att_RA(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, dim_proj=dim_proj,gbiased=outer_biased, num_layers=4)

        self.outer_tf_eeg_out = outer_mod_att_RA(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1,
                                             dim_proj=dim_proj, gbiased=outer_biased, num_layers=4)
        self.outer_tf_eοg_out = outer_mod_att_RA(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1,
                                             dim_proj=dim_proj, gbiased=outer_biased, num_layers=4)

        self.cls_token_eeg = nn.Parameter(torch.randn(1, 1, 1, 1, 1, d_model))
        self.cls_token_eog = nn.Parameter(torch.randn(1, 1, 1, 1, 1, d_model))
        self.cls_token_outer_eeg = nn.Parameter(torch.randn(1, 1, 1, 1, 1, d_model))
        self.cls_token_outer_eog = nn.Parameter(torch.randn(1, 1, 1, 1, 1, d_model))

        # if self.skip_random_mod:
        #     self.skip_dropout = nn.Dropout()
        if self.pos:
            self.inner_positional_embedding = huy_pos_inner(d_model, pos=False, inner=29, outer=21, modalities=1,
                                                              channels=1)
            self.outer_positional_embedding = huy_pos_outer(d_model, pos=False, inner=29, outer=21, modalities=1,
                                                            channels=1)

    def forward(self, x, return_matches=False, extract_norm=False, return_inter_reps=False, return_order=False, skip_modality="random"):

        xeeg = x[0][:, :, :, :, 1:, :]  # mat
        xeog = x[1][:, :, :, :, 1:, :]  # mat

        xeeg = einops.rearrange(xeeg, "b outer mod ch f inner -> b outer inner mod ch f")
        xeog = einops.rearrange(xeog, "b outer mod ch f inner -> b outer inner mod ch f")

        if self.pos:
            xeeg = self.inner_positional_embedding(xeeg)
            xeog = self.inner_positional_embedding(xeog)

        cls_token_eeg = self.cls_token_eeg.repeat(xeeg.shape[0], xeeg.shape[1], 1, 1, xeeg.shape[3], 1)
        xeeg = torch.cat([cls_token_eeg, xeeg], dim=2)

        cls_token_eog = self.cls_token_eog.repeat(xeog.shape[0], xeog.shape[1], 1, 1, xeog.shape[3], 1)
        xeog = torch.cat([cls_token_eog, xeog], dim=2)

        xeeg_common_init = xeeg
        xeog_common_init = xeog

        xeeg_common = xeeg
        xeog_common = xeog

        if self.skip_random_mod:
            if  self.training:
                # skip_modality_eeg = torch.bernoulli(torch.ones(xeeg.shape[0])*0.5)
                # skip_modality_eog = torch.bernoulli(torch.ones(xeog.shape[0])*0.5)
                # skip_modality = skip_modality_eeg*2 + skip_modality_eog
                # skip_modality[skip_modality==3] = 0

                skip_modality = torch.rand(xeeg.shape[0])
                skip_modality[skip_modality>0.66] = 2
                skip_modality[skip_modality<0.33] = 1
                skip_modality = skip_modality.int()
                # skip_modality[skip_modality==0]=1

            else:
                if skip_modality=="full":
                    skip_modality = None
                elif skip_modality=="random":
                    skip_modality = torch.rand(xeeg.shape[0])
                    skip_modality[skip_modality > 0.66] = 2
                    skip_modality[skip_modality < 0.33] = 1
                    skip_modality = skip_modality.int()
                    # skip_modality[skip_modality == 0] = 1

                elif skip_modality=="eeg":
                    skip_modality = torch.ones(xeeg.shape[0])*1
                elif skip_modality=="eog":
                    skip_modality = torch.ones(xeeg.shape[0])*2
                elif skip_modality is None:
                    with torch.no_grad():
                        vae_eeg_output = self.enc_0([x[0]])
                        output_losses = self.enc_0.module.loss_function(vae_eeg_output[0], vae_eeg_output[1],
                                                                        vae_eeg_output[2], vae_eeg_output[3],
                                                                        reduction="none")
                        eeg_routing = output_losses["total"].flatten(start_dim=1).mean(dim=1)
                        # print(eeg_routing)
                        eeg_routing[eeg_routing < 3] = 0
                        eeg_routing[eeg_routing > 3] = 1

                        vae_eog_output = self.enc_1([x[1]])
                        output_losses = self.enc_1.module.loss_function(vae_eog_output[0], vae_eog_output[1],
                                                                        vae_eog_output[2], vae_eog_output[3],
                                                                        reduction="none")
                        eog_routing = output_losses["total"].flatten(start_dim=1).mean(dim=1)
                        # print(eog_routing)
                        eog_routing[eog_routing < 3] = 0
                        eog_routing[eog_routing > 3] = 2

                        skip_modality = eeg_routing + eog_routing
                        skip_modality[skip_modality == 3] = 0

            if skip_modality is not None:
                # xeeg_common = xeeg_sa[np.logical_or(skip_modality==0, skip_modality==1)] #Process EEG with EOG if you are not skipping EOG
                xeeg_common = xeeg[skip_modality==0] #Process EEG with EOG if you are not skipping EOG
                xeeg_common_init = xeeg[skip_modality==0] #Process EEG with EOG if you are not skipping EOG
                # xeeg_undir_sole = xeeg_sa[skip_modality==1] #If you skip EOG process EEG on its own
                xeeg_dir_sole = xeeg[skip_modality==2] #If you skip EOG process EEG on its own
                xeog_common = xeog[skip_modality==0] #Process EOG with EEG if you are not skipping EEG
                xeog_common_init = xeog[skip_modality==0] #Process EOG with EEG if you are not skipping EEG
                # xeog_undir_sole = xeog_sa[skip_modality==2] #If you skip EEG process EOG on its own
                xeog_dir_sole = xeog[skip_modality==1] #If you skip EEG process EOG on its own
            # print(skip_modality)

        if xeeg_common.shape[0]>0 and xeog_common.shape[0]>0:
            xeeg_common = self.inner_tf_eeg.forward_sa(xeeg_common)
            xeog_common = self.inner_tf_eog.forward_sa(xeog_common)

            if self.outer_rep:
                # xeeg_sa_o = self.outer_positional_embedding(xeeg_sa_o)
                # xeog_sa_o = self.outer_positional_embedding(xeog_sa_o)
                xeeg_common = self.outer_tf_eeg(xeeg_common, extract_norm=extract_norm)
                xeog_common = self.outer_tf_eog(xeog_common, extract_norm=extract_norm)

            xeeg_sa_o, xeog_sa_o = xeeg_common[:, :, :1].squeeze(), xeog_common[:, :, :1].squeeze()

            xeeg_ca_common_outer = self.inner_tf_eeg.forward_ca(xeeg_common_init, xeog_common)[:, :, :1]
            xeog_ca_common = self.inner_tf_eog.forward_ca(xeog_common_init, xeeg_common)[:, :, :1]
            if self.pos:
                xeeg_ca_common_outer = self.outer_positional_embedding(xeeg_ca_common_outer)
                xeog_ca_common_outer = self.outer_positional_embedding(xeog_ca_common)

            cls_token_outer_eeg = self.cls_token_outer_eeg.repeat(xeeg_ca_common_outer.shape[0], xeeg_ca_common_outer.shape[1], 1, 1, 1, 1)
            cls_token_outer_eog = self.cls_token_outer_eog.repeat(xeog_ca_common.shape[0], xeog_ca_common.shape[1], 1, 1, 1, 1)

            xeeg_ca_common = torch.cat([cls_token_outer_eeg, xeog_ca_common], dim=3)
            xeog_ca_common = torch.cat([cls_token_outer_eog, xeog_ca_common], dim=3)

            xeeg_ca_common = self.outer_tf_eeg_out(xeeg_ca_common, extract_norm=extract_norm)[:, :, :, :1]
            xeog_ca_common = self.outer_tf_eοg_out(xeog_ca_common, extract_norm=extract_norm)[:, :, :, :1]
            x_common = xeeg_ca_common + xeog_ca_common

        if "xeeg_dir_sole" in locals() and xeeg_dir_sole.shape[0]>0:
            xeeg_ca_sole = self.inner_tf_eeg.forward_ca(xeeg_dir_sole)[:, :, :1]
            if self.pos:
                xeeg_ca_sole = self.outer_positional_embedding(xeeg_ca_sole)
            # cls_token_outer_eeg = self.cls_token_outer.repeat(xeeg_ca_sole.shape[0], xeeg_ca_sole.shape[1], 1, 1, 1, 1)
            # xeeg_ca_sole = torch.cat([cls_token_outer_eeg, xeeg_ca_sole], dim=3)
            xeeg_ca_sole = self.outer_tf_eeg_out(xeeg_ca_sole, extract_norm=extract_norm)[:, :, :, :1]

        if "xeog_dir_sole" in locals() and xeog_dir_sole.shape[0]>0:
            xeog_ca_sole = self.inner_tf_eeg.forward_ca(xeog_dir_sole)[:, :, :1]
            if self.pos:
                xeog_ca_sole = self.outer_positional_embedding(xeog_ca_sole)
            # cls_token_outer_eog = self.cls_token_outer.repeat(xeog_ca_sole.shape[0], xeog_ca_sole.shape[1], 1, 1, 1, 1)
            # xeog_ca_sole = torch.cat([cls_token_outer_eog, xeog_ca_sole], dim=3)
            xeog_ca_sole = self.outer_tf_eοg_out(xeog_ca_sole, extract_norm=extract_norm)[:, :, :, :1]

        if "xeog_dir_sole" in locals() and "xeog_dir_sole" in locals() and (xeog_dir_sole.shape[0]>0 or xeeg_dir_sole.shape[0]>0):
            x = []
            counter = [0, 0, 0]
            for i in range(len(skip_modality)):
                if skip_modality[i] == 0:
                    x.append(x_common[counter[int(skip_modality[i].item())]:counter[int(skip_modality[i].item())] + 1])
                elif skip_modality[i] == 1:
                    x.append(xeog_ca_sole[
                             counter[int(skip_modality[i].item())]:counter[int(skip_modality[i].item())] + 1])
                elif skip_modality[i] == 2:
                    x.append(xeeg_ca_sole[
                             counter[int(skip_modality[i].item())]:counter[int(skip_modality[i].item())] + 1])
                counter[int(skip_modality[i].item())] += 1
            x = torch.cat(x, dim=0)
        else:
            x = x_common

        output={"output_features": x}

        if return_matches:
            if ("xeeg_sa_o" in locals() and xeeg_sa_o.shape[0]>0 and len(xeeg_sa_o.shape)==3) and ("xeog_sa_o" in locals() and xeog_sa_o.shape[0]>0 and len(xeog_sa_o.shape)==3):
                x_match = torch.einsum('b o f , b m f -> b o m', xeeg_sa_o, xeog_sa_o)
                output["matches"] = x_match
            else:
                output["matches"] = None
        if return_inter_reps:
            output["intermediate_reps"] = [xeeg_sa_o, xeeg_sa_o]
        if return_order:
            x_sa_o = torch.cat([xeeg_sa_o, xeog_sa_o], dim=3)
            output["order"] =x_sa_o

        return output
class SleepEnc_BLIP_EEG_EOG_twomode_caouter(nn.Module):
    def __init__(self, args, encs=[None]):
        """
        :param encoder_filters_small, encoder_filters_big: filters of CNN output, quantized by 20
        :param encs_small, encs_big:
        """
        super().__init__()
        self.args = args

        self.enc_0 = encs[0]
        self.enc_1 = encs[0]

        d_model = args.dmodel  # 64*8

        inner_biased = args.inner_biased if "inner_biased" in args else False
        outer_biased = args.outer_biased if "outer_biased" in args else False
        rpos = args.rpos if "rpos" in args else False
        self.pos = args.pos if "pos" in args else True
        dim_proj = args.dim_proj if "dim_proj" in args else 128
        self.outer_rep = args.outer_rep if "outer_rep" in args else False
        self.skip_random_mod = args.skip_random_mod if "skip_random_mod" in args else False

        if inner_biased == "gaussian_learned":
            inner_biased = Gaussian_Learned_Attention_Bias(d_model, heads=8, type="mul")

        # if outer_biased == "gaussian_learned":
        #     outer_biased = Gaussian_Learned_Attention_Bias(d_model, heads=8, type="mul")

        self.inner_tf_eeg = inner_ch_SA_CA(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, dim_proj=dim_proj, gbiased=inner_biased, num_layers=4)
        self.inner_tf_eog = inner_ch_SA_CA(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, dim_proj=dim_proj, gbiased=inner_biased, num_layers=4)

        self.outer_tf_eeg_out = outer_mod_ch_SA_CA(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, dim_proj=dim_proj, gbiased=outer_biased, num_layers=4)
        self.outer_tf_eog_out = outer_mod_ch_SA_CA(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, dim_proj=dim_proj, gbiased=outer_biased, num_layers=4)

        self.cls_token_eeg = nn.Parameter(torch.randn(1, 1, 1, 1, 1, d_model))
        self.cls_token_eog = nn.Parameter(torch.randn(1, 1, 1, 1, 1, d_model))
        self.cls_token_outer_eeg = nn.Parameter(torch.randn(1, 1, 1, 1, 1, d_model))
        self.cls_token_outer_eog = nn.Parameter(torch.randn(1, 1, 1, 1, 1, d_model))

        # if self.skip_random_mod:
        #     self.skip_dropout = nn.Dropout()
        if self.pos:
            self.inner_positional_embedding = huy_pos_inner(d_model, pos=False, inner=29, outer=21, modalities=1,
                                                              channels=1)
            self.outer_positional_embedding = huy_pos_outer(d_model, pos=False, inner=29, outer=21, modalities=1,
                                                            channels=1)

    def forward(self, x, return_matches=False, extract_norm=False, return_inter_reps=False, return_order=False, skip_modality="random"):

        xeeg = x[0][:, :, :, :, 1:, :]  # mat
        xeog = x[1][:, :, :, :, 1:, :]  # mat

        xeeg = einops.rearrange(xeeg, "b outer mod ch f inner -> b outer inner mod ch f")
        xeog = einops.rearrange(xeog, "b outer mod ch f inner -> b outer inner mod ch f")

        if self.pos:
            xeeg = self.inner_positional_embedding(xeeg)
            xeog = self.inner_positional_embedding(xeog)

        cls_token_eeg = self.cls_token_eeg.repeat(xeeg.shape[0], xeeg.shape[1], 1, 1, xeeg.shape[3], 1)
        xeeg = torch.cat([cls_token_eeg, xeeg], dim=2)

        cls_token_eog = self.cls_token_eog.repeat(xeog.shape[0], xeog.shape[1], 1, 1, xeog.shape[3], 1)
        xeog = torch.cat([cls_token_eog, xeog], dim=2)

        xeeg_common_init = xeeg
        xeog_common_init = xeog

        xeeg_common = xeeg
        xeog_common = xeog

        if self.skip_random_mod:
            if  self.training:
                # skip_modality_eeg = torch.bernoulli(torch.ones(xeeg.shape[0])*0.5)
                # skip_modality_eog = torch.bernoulli(torch.ones(xeog.shape[0])*0.5)
                # skip_modality = skip_modality_eeg*2 + skip_modality_eog
                # skip_modality[skip_modality==3] = 0

                skip_modality = torch.rand(xeeg.shape[0])
                # skip_modality[skip_modality>0.5] = 2
                # skip_modality[skip_modality<=0.5] = 1
                skip_modality[skip_modality > 0.66] = 2
                skip_modality[skip_modality < 0.33] = 1
                skip_modality = skip_modality.int()
                # skip_modality[skip_modality < 1.5 ]=0
                # skip_modality = None

            else:
                if skip_modality=="full":
                    skip_modality = None
                elif skip_modality=="random":
                    skip_modality = torch.rand(xeeg.shape[0])
                    # skip_modality[skip_modality > 0.5] = 2
                    # skip_modality[skip_modality <= 0.5] = 1
                    skip_modality[skip_modality > 0.66] = 2
                    skip_modality[skip_modality < 0.33] = 1
                    skip_modality = skip_modality.int()
                    # skip_modality[skip_modality == 0] = 1

                elif skip_modality=="eeg":
                    skip_modality = torch.ones(xeeg.shape[0])*1
                elif skip_modality=="eog":
                    skip_modality = torch.ones(xeeg.shape[0])*2
                elif skip_modality is "vae":
                    with torch.no_grad():
                        vae_eeg_output = self.enc_0([x[0]])
                        output_losses = self.enc_0.module.loss_function(vae_eeg_output[0], vae_eeg_output[1],
                                                                        vae_eeg_output[2], vae_eeg_output[3],
                                                                        reduction="none")
                        eeg_routing = output_losses["total"].flatten(start_dim=1).mean(dim=1)
                        # print(eeg_routing)
                        eeg_routing[eeg_routing < 3] = 0
                        eeg_routing[eeg_routing > 3] = 1

                        vae_eog_output = self.enc_1([x[1]])
                        output_losses = self.enc_1.module.loss_function(vae_eog_output[0], vae_eog_output[1],
                                                                        vae_eog_output[2], vae_eog_output[3],
                                                                        reduction="none")
                        eog_routing = output_losses["total"].flatten(start_dim=1).mean(dim=1)
                        # print(eog_routing)
                        eog_routing[eog_routing < 3] = 0
                        eog_routing[eog_routing > 3] = 2

                        skip_modality = eeg_routing + eog_routing
                        skip_modality[skip_modality == 3] = 0

            if skip_modality is not None:
                # xeeg_common = xeeg_sa[np.logical_or(skip_modality==0, skip_modality==1)] #Process EEG with EOG if you are not skipping EOG
                xeeg_common = xeeg[skip_modality==0] #Process EEG with EOG if you are not skipping EOG
                xeeg_common_init = xeeg[skip_modality==0] #Process EEG with EOG if you are not skipping EOG
                # xeeg_undir_sole = xeeg_sa[skip_modality==1] #If you skip EOG process EEG on its own
                xeeg_dir_sole = xeeg[skip_modality==2] #If you skip EOG process EEG on its own
                xeog_common = xeog[skip_modality==0] #Process EOG with EEG if you are not skipping EEG
                xeog_common_init = xeog[skip_modality==0] #Process EOG with EEG if you are not skipping EEG
                # xeog_undir_sole = xeog_sa[skip_modality==2] #If you skip EEG process EOG on its own
                xeog_dir_sole = xeog[skip_modality==1] #If you skip EEG process EOG on its own
            # print(skip_modality)

        if xeeg_common.shape[0]>0 and xeog_common.shape[0]>0:
            xeeg_common = self.inner_tf_eeg.forward_sa(xeeg_common)
            xeog_common = self.inner_tf_eog.forward_sa(xeog_common)

            # xeeg_sa_o = self.outer_positional_embedding(xeeg_sa_o)
            # xeog_sa_o = self.outer_positional_embedding(xeog_sa_o)
            xeeg_common_outer = self.outer_tf_eeg_out.forward_sa(xeeg_common, extract_norm=extract_norm)[:, :, :1]
            xeog_common_outer = self.outer_tf_eog_out.forward_sa(xeog_common, extract_norm=extract_norm)[:, :, :1]

            xeeg_sa_o, xeog_sa_o = xeeg_common_outer.squeeze(), xeog_common_outer.squeeze()

            xeeg_ca_common = self.inner_tf_eeg.forward_ca(xeeg_common_init, xeog_common)[:, :, :1]
            xeog_ca_common = self.inner_tf_eog.forward_ca(xeog_common_init, xeeg_common)[:, :, :1]

            if self.pos:
                xeeg_ca_common = self.outer_positional_embedding(xeeg_ca_common)
                xeog_ca_common = self.outer_positional_embedding(xeog_ca_common)

            # cls_token_outer_eeg = self.cls_token_outer_eeg.repeat(xeeg_ca_common.shape[0], xeeg_ca_common.shape[1], 1, 1, 1, 1)
            # cls_token_outer_eog = self.cls_token_outer_eog.repeat(xeog_ca_common.shape[0], xeog_ca_common.shape[1], 1, 1, 1, 1)
            #
            # xeeg_ca_common = torch.cat([cls_token_outer_eeg, xeog_ca_common], dim=3)
            # xeog_ca_common = torch.cat([cls_token_outer_eog, xeog_ca_common], dim=3)

            xeeg_ca_common_outer = self.outer_tf_eeg_out.forward_ca(xeeg_ca_common, xeog_common_outer, extract_norm=extract_norm)
            xeog_ca_common_outer = self.outer_tf_eog_out.forward_ca(xeog_ca_common, xeeg_common_outer, extract_norm=extract_norm)

            x_common = xeeg_ca_common_outer + xeog_ca_common_outer

        if "xeeg_dir_sole" in locals() and xeeg_dir_sole.shape[0]>0:
            xeeg_ca_sole = self.inner_tf_eeg.forward_ca(xeeg_dir_sole)[:, :, :1]
            if self.pos:
                xeeg_ca_sole = self.outer_positional_embedding(xeeg_ca_sole)
            # cls_token_outer_eeg = self.cls_token_outer.repeat(xeeg_ca_sole.shape[0], xeeg_ca_sole.shape[1], 1, 1, 1, 1)
            # xeeg_ca_sole = torch.cat([cls_token_outer_eeg, xeeg_ca_sole], dim=3)
            xeeg_ca_sole = self.outer_tf_eeg_out.forward_ca(xeeg_ca_sole, extract_norm=extract_norm)[:, :, :, :1]

        if "xeog_dir_sole" in locals() and xeog_dir_sole.shape[0]>0:
            xeog_ca_sole = self.inner_tf_eeg.forward_ca(xeog_dir_sole)[:, :, :1]
            if self.pos:
                xeog_ca_sole = self.outer_positional_embedding(xeog_ca_sole)
            # cls_token_outer_eog = self.cls_token_outer.repeat(xeog_ca_sole.shape[0], xeog_ca_sole.shape[1], 1, 1, 1, 1)
            # xeog_ca_sole = torch.cat([cls_token_outer_eog, xeog_ca_sole], dim=3)
            xeog_ca_sole = self.outer_tf_eog_out.forward_ca(xeog_ca_sole, extract_norm=extract_norm)[:, :, :, :1]

        if "xeog_dir_sole" in locals() and "xeog_dir_sole" in locals() and (xeog_dir_sole.shape[0]>0 or xeeg_dir_sole.shape[0]>0):
            x = []
            counter = [0, 0, 0]
            for i in range(len(skip_modality)):
                if skip_modality[i] == 0:
                    x.append(x_common[counter[int(skip_modality[i].item())]:counter[int(skip_modality[i].item())] + 1])
                elif skip_modality[i] == 1:
                    x.append(xeog_ca_sole[
                             counter[int(skip_modality[i].item())]:counter[int(skip_modality[i].item())] + 1])
                elif skip_modality[i] == 2:
                    x.append(xeeg_ca_sole[
                             counter[int(skip_modality[i].item())]:counter[int(skip_modality[i].item())] + 1])
                counter[int(skip_modality[i].item())] += 1
            x = torch.cat(x, dim=0)
        else:
            x = x_common

        output={"output_features": x}

        if return_matches:
            if ("xeeg_sa_o" in locals() and xeeg_sa_o.shape[0]>0 and len(xeeg_sa_o.shape)==3) and ("xeog_sa_o" in locals() and xeog_sa_o.shape[0]>0 and len(xeog_sa_o.shape)==3):
                x_match = torch.einsum('b o f , b m f -> b o m', xeeg_sa_o, xeog_sa_o)
                output["matches"] = x_match
            else:
                output["matches"] = None
        if return_inter_reps:
            output["intermediate_reps"] = [xeeg_sa_o, xeeg_sa_o]
        if return_order:
            x_sa_o = torch.cat([xeeg_sa_o, xeog_sa_o], dim=3)
            output["order"] =x_sa_o

        return output
class SleepEnc_BLIP_EEG_EOG_twomode_caouter_shared(nn.Module):
    def __init__(self, args, encs=[None]):
        """
        :param encoder_filters_small, encoder_filters_big: filters of CNN output, quantized by 20
        :param encs_small, encs_big:
        """
        super().__init__()
        self.args = args

        # self.enc_0 = encs[0]
        # self.enc_1 = encs[0]

        d_model = args.dmodel  # 64*8

        inner_biased = args.inner_biased if "inner_biased" in args else False
        outer_biased = args.outer_biased if "outer_biased" in args else False
        rpos = args.rpos if "rpos" in args else False
        self.pos = args.pos if "pos" in args else True
        dim_proj = args.dim_proj if "dim_proj" in args else 128
        self.outer_rep = args.outer_rep if "outer_rep" in args else False
        self.skip_percentile = args.skip_percentile if "skip_percentile" in args else False
        self.mod_token = args.mod_token if "mod_token" in args else False
        self.default_skip = args.default_skip if "default_skip" in args else False

        if inner_biased == "gaussian_learned":
            inner_biased = Gaussian_Learned_Attention_Bias(d_model, heads=8, type="mul")

        # if outer_biased == "gaussian_learned":
        #     outer_biased = Gaussian_Learned_Attention_Bias(d_model, heads=8, type="mul")

        self.inner_tf_eeg = inner_ch_SA_CA_shared_v2(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, dim_proj=dim_proj, gbiased=inner_biased, num_layers=4)
        self.inner_tf_eog = inner_ch_SA_CA_shared_v2(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, dim_proj=dim_proj, gbiased=inner_biased, num_layers=4)

        self.outer_tf_eeg_out = outer_mod_ch_SA_CA_shared_v2(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, dim_proj=dim_proj, gbiased=outer_biased, num_layers=4)
        self.outer_tf_eog_out = outer_mod_ch_SA_CA_shared_v2(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, dim_proj=dim_proj, gbiased=outer_biased, num_layers=4)

        self.cls_token_eeg = nn.Parameter(torch.randn(1, 1, 1, 1, 1, d_model))
        self.cls_token_eog = nn.Parameter(torch.randn(1, 1, 1, 1, 1, d_model))

        self.cls_token_outer_eeg = nn.Parameter(torch.randn(1, 1, 1, 1, 1, d_model))
        self.cls_token_outer_eog = nn.Parameter(torch.randn(1, 1, 1, 1, 1, d_model))

        # if self.skip_random_mod:
        #     self.skip_dropout = nn.Dropout()
        if self.pos:
            self.inner_positional_embedding = huy_pos_inner(d_model, pos=False, inner=29, outer=21, modalities=1, channels=1)
            self.outer_positional_embedding = huy_pos_outer(d_model, pos=False, inner=29, outer=21, modalities=1, channels=1)

        if self.mod_token:
            self.modtype_token = modtype_embedding(num_modalities=2, dim=d_model)

    def forward(self, x, skip_modality="random", **kwargs):

        xeeg = x["stft_eeg"][:, :, :, :, 1:, :]  # mat
        xeog = x["stft_eog"][:, :, :, :, 1:, :]  # mat
        # xeog = x["stft_eeg"][:, :, :, :, 1:, :]  # mat

        xeeg = einops.rearrange(xeeg, "b outer mod ch f inner -> b outer inner mod ch f")
        xeog = einops.rearrange(xeog, "b outer mod ch f inner -> b outer inner mod ch f")

        if self.mod_token:
            xeeg = self.modtype_token(data=xeeg, mod_num=0)
            xeog = self.modtype_token(data=xeog, mod_num=1)

        if self.pos:
            xeeg = self.inner_positional_embedding(xeeg)
            xeog = self.inner_positional_embedding(xeog)

        cls_token_eeg = self.cls_token_eeg.repeat(xeeg.shape[0], xeeg.shape[1], 1, 1, xeeg.shape[3], 1)
        xeeg = torch.cat([cls_token_eeg, xeeg], dim=2)

        cls_token_eog = self.cls_token_eog.repeat(xeog.shape[0], xeog.shape[1], 1, 1, xeog.shape[3], 1)
        xeog = torch.cat([cls_token_eog, xeog], dim=2)

        xeeg_common = xeeg
        xeog_common = xeog

        skip_modality = self.calculate_skip_modality( num_batches=xeeg.shape[0], outer_size=xeeg.shape[1], skip_modality=skip_modality)
        if skip_modality is not None:
            # print(skip_modality.shape)
            # print(xeeg.shape)
            xeeg_common = xeeg[skip_modality==0] #Keep epochs where both mods are available
            xeog_common = xeog[skip_modality==0]

            xeeg_sole = xeeg[skip_modality==2] #Keep epochs where eog is not available
            xeog_sole = xeog[skip_modality==1] #Keep epochs where eeg is not available

            [xeeg_common, xeog_common, xeeg_sole, xeog_sole] = self.reshape_batchouter([xeeg_common, xeog_common, xeeg_sole, xeog_sole], outer_size=xeeg.shape[1])
        else:
            skip_modality = torch.zeros(xeeg.shape[0])


        x = torch.empty(xeeg.shape)[:, :, :1].to(xeeg.device)
        output = {"output_features": x}
        if xeeg_common.shape[0]>0 and xeog_common.shape[0]>0:

            output = self.forward_common(xeeg=xeeg_common, xeog=xeog_common, skip_modality=skip_modality, output=output, **kwargs)
        if "xeeg_sole" in locals() and xeeg_sole.shape[0]>0:
            output = self.forward_eeg(xeeg=xeeg_sole, skip_modality=skip_modality, output=output)
        if "xeog_sole" in locals() and xeog_sole.shape[0]>0:
            output = self.forward_eog(xeog=xeog_sole, skip_modality=skip_modality, output=output)

        return output

    def calculate_skip_modality(self, num_batches, outer_size, skip_modality):
        # if  self.training:
        #
        #     skip_modality = torch.rand(num_batches)
        #     skip_modality[skip_modality > 1 - self.skip_percentile["eeg"]] = 2
        #     skip_modality[skip_modality < self.skip_percentile["eog"]] = 1
        #     skip_modality = skip_modality.int()
        # else:

        if self.default_skip:
            skip_modality = self.default_skip

        if skip_modality=="full":
            skip_modality = None
        elif skip_modality=="random":
            skip_modality = torch.rand(num_batches, outer_size)
            skip_modality[skip_modality > self.skip_percentile["eeg"]] = 2
            skip_modality[skip_modality < self.skip_percentile["eog"]] = 1
            skip_modality = skip_modality.int()
        elif skip_modality=="eeg":
            skip_modality = torch.ones(num_batches, outer_size)*1
        elif skip_modality=="eog":
            skip_modality = torch.ones(num_batches, outer_size)*2
        elif "percentile" in skip_modality:
            skip_eeg = skip_modality.split("_")[0]
            skip_eog = skip_modality.split("_")[1]

            skip_modality = torch.rand(num_batches, outer_size)
            skip_modality[skip_modality > 1 - skip_eeg] = 2
            skip_modality[skip_modality < skip_eog] = 1
            skip_modality = skip_modality.int()
        elif type(skip_modality)==dict:
            skip_modality = skip_modality["stft_eeg"] + skip_modality["stft_eog"]*2
            skip_modality[skip_modality==3] = 0

        elif skip_modality is "vae":
                raise not NotImplementedError
                # with torch.no_grad():
                #     vae_eeg_output = self.enc_0([x[0]])
                #     output_losses = self.enc_0.module.loss_function(vae_eeg_output[0], vae_eeg_output[1],
                #                                                     vae_eeg_output[2], vae_eeg_output[3],
                #                                                     reduction="none")
                #     eeg_routing = output_losses["total"].flatten(start_dim=1).mean(dim=1)
                #     # print(eeg_routing)
                #     eeg_routing[eeg_routing < 3] = 0
                #     eeg_routing[eeg_routing > 3] = 1
                #
                #     vae_eog_output = self.enc_1([x[1]])
                #     output_losses = self.enc_1.module.loss_function(vae_eog_output[0], vae_eog_output[1],
                #                                                     vae_eog_output[2], vae_eog_output[3],
                #                                                     reduction="none")
                #     eog_routing = output_losses["total"].flatten(start_dim=1).mean(dim=1)
                #     # print(eog_routing)
                #     eog_routing[eog_routing < 3] = 0
                #     eog_routing[eog_routing > 3] = 2
                #
                #     skip_modality = eeg_routing + eog_routing
                #     skip_modality[skip_modality == 3] = 0
        return skip_modality

    def forward_common(self, xeeg, xeog, skip_modality, output, **kwargs):

        return_matches = kwargs["return_matches"] if "return_matches" in kwargs else False
        return_inter_reps = kwargs["return_inter_reps"] if "return_inter_reps" in kwargs else False
        return_order = kwargs["return_order"] if "return_order" in kwargs else False

        xeeg_common_i = self.inner_tf_eeg.forward_sa(xeeg)
        xeog_common_i = self.inner_tf_eog.forward_sa(xeog)

        # xeeg_sa_o = self.outer_positional_embedding(xeeg_sa_o)
        # xeog_sa_o = self.outer_positional_embedding(xeog_sa_o)
        xeeg_common_outer = self.outer_tf_eeg_out.forward_sa(xeeg_common_i)[:, :, :1]
        xeog_common_outer = self.outer_tf_eog_out.forward_sa(xeog_common_i)[:, :, :1]

        xeeg_sa_o, xeog_sa_o = xeeg_common_outer.squeeze(), xeog_common_outer.squeeze()

        if return_matches:
            if ("xeeg_sa_o" in locals() and xeeg_sa_o.shape[0] > 0 and len(xeeg_sa_o.shape) == 3) and (
                    "xeog_sa_o" in locals() and xeog_sa_o.shape[0] > 0 and len(xeog_sa_o.shape) == 3):
                x_match = torch.einsum('b o f , b m f -> b o m', xeeg_sa_o, xeog_sa_o)
            else:
                x_match = torch.Tensor([0]).to(xeeg.device)
        else:
            x_match = None

        xeeg_ca_common = self.inner_tf_eeg.forward_ca(xeeg, xeog_common_i)[:, :, :1]
        xeog_ca_common = self.inner_tf_eog.forward_ca(xeog, xeeg_common_i)[:, :, :1]

        if self.pos:
            xeeg_ca_common = self.outer_positional_embedding(xeeg_ca_common)
            xeog_ca_common = self.outer_positional_embedding(xeog_ca_common)

        # cls_token_outer_eeg = self.cls_token_outer_eeg.repeat(xeeg_ca_common.shape[0], xeeg_ca_common.shape[1], 1, 1, 1, 1)
        # cls_token_outer_eog = self.cls_token_outer_eog.repeat(xeog_ca_common.shape[0], xeog_ca_common.shape[1], 1, 1, 1, 1)
        #
        # xeeg_ca_common = torch.cat([cls_token_outer_eeg, xeog_ca_common], dim=3)
        # xeog_ca_common = torch.cat([cls_token_outer_eog, xeog_ca_common], dim=3)

        xeeg_ca_common_outer = self.outer_tf_eeg_out.forward_ca(xeeg_ca_common, xeog_common_outer)
        xeog_ca_common_outer = self.outer_tf_eog_out.forward_ca(xeog_ca_common, xeeg_common_outer)

        x_common = xeeg_ca_common_outer + xeog_ca_common_outer
        output["output_features"][skip_modality == 0] = x_common

        if return_matches: output["matches"] = x_match
        if return_inter_reps: output["intermediate_reps"] = [xeeg_sa_o, xeeg_sa_o]
        if return_order: output["order"] = torch.cat([xeeg_sa_o, xeog_sa_o], dim=3)
        return output

    def forward_eeg(self, xeeg, skip_modality, output):
        xeeg_ca_sole = self.inner_tf_eeg.forward_ca(xeeg)[:, :, :1]
        if self.pos:
            xeeg_ca_sole = self.outer_positional_embedding(xeeg_ca_sole)
        # cls_token_outer_eeg = self.cls_token_outer.repeat(xeeg_ca_sole.shape[0], xeeg_ca_sole.shape[1], 1, 1, 1, 1)
        # xeeg_ca_sole = torch.cat([cls_token_outer_eeg, xeeg_ca_sole], dim=3)
        xeeg_ca_sole = self.outer_tf_eeg_out.forward_ca(xeeg_ca_sole)[:, :, :, :1]
        output["output_features"][skip_modality == 2] = xeeg_ca_sole.flatten(start_dim=0, end_dim=1)
        return output

    def forward_eog(self, xeog, skip_modality, output):
        xeog_ca_sole = self.inner_tf_eog.forward_ca(xeog)[:, :, :1]
        if self.pos:
            xeog_ca_sole = self.outer_positional_embedding(xeog_ca_sole)
        # cls_token_outer_eog = self.cls_token_outer.repeat(xeog_ca_sole.shape[0], xeog_ca_sole.shape[1], 1, 1, 1, 1)
        # xeog_ca_sole = torch.cat([cls_token_outer_eog, xeog_ca_sole], dim=3)
        xeog_ca_sole = self.outer_tf_eog_out.forward_ca(xeog_ca_sole)[:, :, :, :1]
        output["output_features"][skip_modality == 1] = xeog_ca_sole.flatten(start_dim=0, end_dim=1)
        return output

    def reshape_batchouter(self, list_of_tensors, outer_size):
        output_tens = []
        for tens in list_of_tensors:
            if tens.shape[0]>0:
                batch_size = tens.shape[0]/outer_size
                if batch_size%1 !=0: raise ValueError("split_modalities should be the same for every outer in each batch sample. (b o) shape was {}".format(tens.shape[0]))
                tens = einops.rearrange(tens, "(b o) i m c f -> b o i m c f", b=int(batch_size), o=outer_size)
            output_tens.append(tens)
        return output_tens


class SleepEnc_BLIP_EEG_EOG_twomode_caouter_shared_free(nn.Module):
    def __init__(self, args, encs=[None]):
        """
        :param encoder_filters_small, encoder_filters_big: filters of CNN output, quantized by 20
        :param encs_small, encs_big:
        """
        super().__init__()
        self.args = args

        # self.enc_0 = encs[0]
        # self.enc_1 = encs[0]

        d_model = args.dmodel  # 64*8

        inner_biased = args.inner_biased if "inner_biased" in args else False
        outer_biased = args.outer_biased if "outer_biased" in args else False
        rpos = args.rpos if "rpos" in args else False
        self.pos = args.pos if "pos" in args else True
        dim_proj = args.dim_proj if "dim_proj" in args else 128
        self.outer_rep = args.outer_rep if "outer_rep" in args else False
        self.skip_percentile = args.skip_percentile if "skip_percentile" in args else False
        self.mod_token = args.mod_token if "mod_token" in args else False

        if inner_biased == "gaussian_learned":
            inner_biased = Gaussian_Learned_Attention_Bias(d_model, heads=8, type="mul")

        # if outer_biased == "gaussian_learned":
        #     outer_biased = Gaussian_Learned_Attention_Bias(d_model, heads=8, type="mul")

        self.inner_tf_eeg = inner_ch_SA_CA_shared_v2(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, dim_proj=dim_proj, gbiased=inner_biased, num_layers=4)
        self.inner_tf_eog = inner_ch_SA_CA_shared_v2(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, dim_proj=dim_proj, gbiased=inner_biased, num_layers=4)

        self.outer_tf_eeg_out = outer_mod_ch_SA_CA_shared_v2(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, dim_proj=dim_proj, gbiased=outer_biased, num_layers=4)
        self.outer_tf_eog_out = outer_mod_ch_SA_CA_shared_v2(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, dim_proj=dim_proj, gbiased=outer_biased, num_layers=4)

        self.cls_token_eeg = nn.Parameter(torch.randn(1, 1, 1, 1, 1, d_model))
        self.cls_token_eog = nn.Parameter(torch.randn(1, 1, 1, 1, 1, d_model))

        self.cls_token_outer_eeg = nn.Parameter(torch.randn(1, 1, 1, 1, 1, d_model))
        self.cls_token_outer_eog = nn.Parameter(torch.randn(1, 1, 1, 1, 1, d_model))

        # if self.skip_random_mod:
        #     self.skip_dropout = nn.Dropout()
        if self.pos:
            self.inner_positional_embedding = huy_pos_inner(d_model, pos=False, inner=29, outer=21, modalities=1, channels=1)
            self.outer_positional_embedding = huy_pos_outer(d_model, pos=False, inner=29, outer=21, modalities=1, channels=1)

        if self.mod_token:
            self.modtype_token = modtype_embedding(num_modalities=2, dim=d_model)

    def forward(self, x, skip_modality="random", **kwargs):

        xeeg = x["stft_eeg"][:, :, :, :, 1:, :]  # mat
        xeog = x["stft_eog"][:, :, :, :, 1:, :]  # mat

        xeeg = einops.rearrange(xeeg, "b outer mod ch f inner -> b outer inner mod ch f")
        xeog = einops.rearrange(xeog, "b outer mod ch f inner -> b outer inner mod ch f")

        if self.mod_token:
            xeeg = self.modtype_token(data=xeeg, mod_num=0)
            xeog = self.modtype_token(data=xeog, mod_num=1)

        if self.pos:
            xeeg = self.inner_positional_embedding(xeeg)
            xeog = self.inner_positional_embedding(xeog)

        cls_token_eeg = self.cls_token_eeg.repeat(xeeg.shape[0], xeeg.shape[1], 1, 1, xeeg.shape[3], 1)
        xeeg = torch.cat([cls_token_eeg, xeeg], dim=2)

        cls_token_eog = self.cls_token_eog.repeat(xeog.shape[0], xeog.shape[1], 1, 1, xeog.shape[3], 1)
        xeog = torch.cat([cls_token_eog, xeog], dim=2)

        skip_modality = self.calculate_skip_modality( num_batches=xeeg.shape[0], outer_size=xeeg.shape[1], skip_modality=skip_modality)
        if skip_modality is not None:
            print(skip_modality.shape)
        #     # print(xeeg.shape)
        #     xeeg_common = xeeg[skip_modality==0] #Keep epochs where both mods are available
        #     xeog_common = xeog[skip_modality==0]
        #
        #     xeeg_sole = xeeg[skip_modality==2] #Keep epochs where eog is not available
        #     xeog_sole = xeog[skip_modality==1] #Keep epochs where eeg is not available
        #
        #     [xeeg_common, xeog_common, xeeg_sole, xeog_sole] = self.reshape_batchouter([xeeg_common, xeog_common, xeeg_sole, xeog_sole], outer_size=xeeg.shape[1])
        else:
            skip_modality = torch.zeros(xeeg.shape[0])

        output = {"output_features": {}}
        output["output_features"]["skipped"] = torch.empty(xeeg.shape[0],xeeg.shape[1], 1,1,1,128).to(xeeg.device)
        output = self.forward_sole(xeeg=xeeg, xeog=xeog, output=output, skip_modality=skip_modality, **kwargs)
        output = self.forward_common(xeeg=xeeg, xeog=xeog, output=output, skip_modality=skip_modality, **kwargs)

        return output

    def calculate_skip_modality(self, num_batches, outer_size, skip_modality):
        # if  self.training:
        #
        #     skip_modality = torch.rand(num_batches)
        #     skip_modality[skip_modality > 1 - self.skip_percentile["eeg"]] = 2
        #     skip_modality[skip_modality < self.skip_percentile["eog"]] = 1
        #     skip_modality = skip_modality.int()
        # else:

        # if self.default_skip:
        #     skip_modality = self.default_skip

        if skip_modality=="full":
            skip_modality = None
        elif skip_modality=="random":
            skip_modality = torch.rand(num_batches, outer_size)
            skip_modality[skip_modality > self.skip_percentile["eeg"]] = 2
            skip_modality[skip_modality < self.skip_percentile["eog"]] = 1
            skip_modality = skip_modality.int()
        elif skip_modality=="eeg":
            skip_modality = torch.ones(num_batches, outer_size)*1
        elif skip_modality=="eog":
            skip_modality = torch.ones(num_batches, outer_size)*2
        elif "percentile" in skip_modality:
            skip_eeg = skip_modality.split("_")[0]
            skip_eog = skip_modality.split("_")[1]

            skip_modality = torch.rand(num_batches, outer_size)
            skip_modality[skip_modality > 1 - skip_eeg] = 2
            skip_modality[skip_modality < skip_eog] = 1
            skip_modality = skip_modality.int()
        elif type(skip_modality)==dict:
            skip_modality = skip_modality["stft_eeg"] + skip_modality["stft_eog"]*2
            skip_modality[skip_modality==3] = 0

        elif skip_modality is "vae":
                raise not NotImplementedError
                # with torch.no_grad():
                #     vae_eeg_output = self.enc_0([x[0]])
                #     output_losses = self.enc_0.module.loss_function(vae_eeg_output[0], vae_eeg_output[1],
                #                                                     vae_eeg_output[2], vae_eeg_output[3],
                #                                                     reduction="none")
                #     eeg_routing = output_losses["total"].flatten(start_dim=1).mean(dim=1)
                #     # print(eeg_routing)
                #     eeg_routing[eeg_routing < 3] = 0
                #     eeg_routing[eeg_routing > 3] = 1
                #
                #     vae_eog_output = self.enc_1([x[1]])
                #     output_losses = self.enc_1.module.loss_function(vae_eog_output[0], vae_eog_output[1],
                #                                                     vae_eog_output[2], vae_eog_output[3],
                #                                                     reduction="none")
                #     eog_routing = output_losses["total"].flatten(start_dim=1).mean(dim=1)
                #     # print(eog_routing)
                #     eog_routing[eog_routing < 3] = 0
                #     eog_routing[eog_routing > 3] = 2
                #
                #     skip_modality = eeg_routing + eog_routing
                #     skip_modality[skip_modality == 3] = 0
        return skip_modality

    def forward_common(self, xeeg, xeog, output, skip_modality, **kwargs):

        xeeg_common_i = output["output_features"]["inner_eeg"]
        xeog_common_i = output["output_features"]["inner_eog"]

        output["output_features"].pop("inner_eeg")
        output["output_features"].pop("inner_eog")

        xeeg_common_outer = output["output_features"]["eeg"]
        xeog_common_outer = output["output_features"]["eog"]

        xeeg_ca_common = self.inner_tf_eeg.forward_ca(xeeg, xeog_common_i)[:, :, :1]
        xeog_ca_common = self.inner_tf_eog.forward_ca(xeog, xeeg_common_i)[:, :, :1]

        if self.pos:
            xeeg_ca_common = self.outer_positional_embedding(xeeg_ca_common)
            xeog_ca_common = self.outer_positional_embedding(xeog_ca_common)

        xeeg_ca_common_outer = self.outer_tf_eeg_out.forward_ca(xeeg_ca_common, xeog_common_outer)
        xeog_ca_common_outer = self.outer_tf_eog_out.forward_ca(xeog_ca_common, xeeg_common_outer)

        x_common = xeeg_ca_common_outer + xeog_ca_common_outer
        output["output_features"]["combined"] = x_common

        output["output_features"]["skipped"][skip_modality == 0] = x_common[skip_modality == 0]

        return output

    def forward_sole(self, xeeg, xeog, output, skip_modality, **kwargs):

        return_matches = kwargs["return_matches"] if "return_matches" in kwargs else False

        xeeg_sole = self.inner_tf_eeg.forward_ca(xeeg)[:, :, :1]
        xeog_sole = self.inner_tf_eog.forward_ca(xeog)[:, :, :1]

        xeeg_cls_sole = xeeg_sole[:, :, :1]
        xeog_cls_sole = xeog_sole[:, :, :1]

        if self.pos:
            xeeg_cls_sole = self.outer_positional_embedding(xeeg_cls_sole)
            xeog_cls_sole = self.outer_positional_embedding(xeog_cls_sole)

        xeeg_outer_sole = self.outer_tf_eeg_out.forward_ca(xeeg_cls_sole)
        xeog_outer_sole = self.outer_tf_eog_out.forward_ca(xeog_cls_sole)

        output["output_features"]["eeg"] = xeeg_outer_sole
        output["output_features"]["inner_eeg"] = xeeg_sole
        output["output_features"]["eog"] = xeog_outer_sole
        output["output_features"]["inner_eog"] = xeog_sole

        xeeg_sa_o, xeog_sa_o = xeeg_outer_sole.squeeze(), xeog_outer_sole.squeeze()

        x_match = None
        if return_matches:
            if len(xeeg_sa_o.shape) == 3 and len(xeog_sa_o.shape) == 3:
                x_match = torch.einsum('b o f , b m f -> b o m', xeeg_sa_o, xeog_sa_o)
            else:
                x_match = torch.Tensor([0]).to(xeeg.device)

        # output["output_features"]["skipped"] = xeeg_outer_sole

        output["output_features"]["skipped"][skip_modality == 2] = xeeg_outer_sole[skip_modality == 2]
        output["output_features"]["skipped"][skip_modality == 1] = xeog_outer_sole[skip_modality == 1]

        output["matches"] = x_match

        return output
class SleepEnc_BLIP_EEG_EOG_twomode_caouter_shared_free_cliplike(nn.Module):
    def __init__(self, args, encs=[None]):
        """
        :param encoder_filters_small, encoder_filters_big: filters of CNN output, quantized by 20
        :param encs_small, encs_big:
        """
        super().__init__()
        self.args = args

        # self.enc_0 = encs[0]
        # self.enc_1 = encs[0]

        d_model = args.dmodel  # 64*8

        inner_biased = args.inner_biased if "inner_biased" in args else False
        outer_biased = args.outer_biased if "outer_biased" in args else False
        rpos = args.rpos if "rpos" in args else False
        self.pos = args.pos if "pos" in args else True
        dim_proj = args.dim_proj if "dim_proj" in args else 128
        self.outer_rep = args.outer_rep if "outer_rep" in args else False
        self.skip_percentile = args.skip_percentile if "skip_percentile" in args else False
        self.mod_token = args.mod_token if "mod_token" in args else False

        if inner_biased == "gaussian_learned":
            inner_biased = Gaussian_Learned_Attention_Bias(d_model, heads=8, type="mul")

        # if outer_biased == "gaussian_learned":
        #     outer_biased = Gaussian_Learned_Attention_Bias(d_model, heads=8, type="mul")

        self.inner_tf_eeg = inner_ch_SA_CA_shared_v2(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, dim_proj=dim_proj, gbiased=inner_biased, num_layers=4)
        self.inner_tf_eog = inner_ch_SA_CA_shared_v2(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, dim_proj=dim_proj, gbiased=inner_biased, num_layers=4)

        self.outer_tf_eeg_out = outer_mod_ch_SA_CA_shared_v2(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, dim_proj=dim_proj, gbiased=outer_biased, num_layers=4)
        self.outer_tf_eog_out = outer_mod_ch_SA_CA_shared_v2(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, dim_proj=dim_proj, gbiased=outer_biased, num_layers=4)

        self.cls_token_eeg = nn.Parameter(torch.randn(1, 1, 1, 1, 1, d_model))
        self.cls_token_eog = nn.Parameter(torch.randn(1, 1, 1, 1, 1, d_model))

        self.cls_token_outer_eeg = nn.Parameter(torch.randn(1, 1, 1, 1, 1, d_model))
        self.cls_token_outer_eog = nn.Parameter(torch.randn(1, 1, 1, 1, 1, d_model))

        self.eeg_proj_head = nn.Parameter(torch.randn(d_model, d_model))
        self.eog_proj_head = nn.Parameter(torch.randn(d_model, d_model))

        if self.eog_proj_head is not None:
            nn.init.normal_(self.eeg_proj_head, std= d_model ** -0.5)
            nn.init.normal_(self.eog_proj_head, std= d_model ** -0.5)

        self.ln_pre_eeg = nn.LayerNorm(d_model)
        self.ln_pre_eog = nn.LayerNorm(d_model)

        self.ln_post_eeg = nn.LayerNorm(d_model)
        self.ln_post_eog = nn.LayerNorm(d_model)

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        if self.pos:
            self.inner_positional_embedding = huy_pos_inner(d_model, pos=False, inner=29, outer=21, modalities=1, channels=1)
            self.outer_positional_embedding = huy_pos_outer(d_model, pos=False, inner=29, outer=21, modalities=1, channels=1)

        if self.mod_token:
            self.modtype_token = modtype_embedding(num_modalities=2, dim=d_model)

    def forward(self, x, skip_modality="random", **kwargs):

        xeeg = x["stft_eeg"][:, :, :, :, 1:, :]  # mat
        xeog = x["stft_eog"][:, :, :, :, 1:, :]  # mat

        xeeg = einops.rearrange(xeeg, "b outer mod ch f inner -> b outer inner mod ch f")
        xeog = einops.rearrange(xeog, "b outer mod ch f inner -> b outer inner mod ch f")

        if self.mod_token:
            xeeg = self.modtype_token(data=xeeg, mod_num=0)
            xeog = self.modtype_token(data=xeog, mod_num=1)

        if self.pos:
            xeeg = self.inner_positional_embedding(xeeg)
            xeog = self.inner_positional_embedding(xeog)

        cls_token_eeg = self.cls_token_eeg.repeat(xeeg.shape[0], xeeg.shape[1], 1, 1, xeeg.shape[3], 1)
        xeeg = torch.cat([cls_token_eeg, xeeg], dim=2)

        cls_token_eog = self.cls_token_eog.repeat(xeog.shape[0], xeog.shape[1], 1, 1, xeog.shape[3], 1)
        xeog = torch.cat([cls_token_eog, xeog], dim=2)

        # xeeg = self.ln_pre_eeg(xeeg)
        # xeog = self.ln_pre_eog(xeog)

        output = {"output_features": {}}
        output = self.forward_sole(xeeg=xeeg, xeog=xeog, output=output, **kwargs)
        output = self.forward_common(xeeg=xeeg, xeog=xeog, output=output, **kwargs)

        return output

    def forward_common(self, xeeg, xeog, output, **kwargs):

        xeeg_common_i = output["output_features"]["inner_eeg"]
        xeog_common_i = output["output_features"]["inner_eog"]

        output["output_features"].pop("inner_eeg")
        output["output_features"].pop("inner_eog")

        xeeg_common_outer = output["output_features"]["eeg"]
        xeog_common_outer = output["output_features"]["eog"]

        xeeg_ca_common = self.inner_tf_eeg.forward_ca(xeeg, xeog_common_i)[:, :, :1]
        xeog_ca_common = self.inner_tf_eog.forward_ca(xeog, xeeg_common_i)[:, :, :1]

        if self.pos:
            xeeg_ca_common = self.outer_positional_embedding(xeeg_ca_common)
            xeog_ca_common = self.outer_positional_embedding(xeog_ca_common)

        xeeg_ca_common_outer = self.outer_tf_eeg_out.forward_ca(xeeg_ca_common, xeog_common_outer)
        xeog_ca_common_outer = self.outer_tf_eog_out.forward_ca(xeog_ca_common, xeeg_common_outer)

        # xeeg_ca_common_outer = self.ln_post_eeg(xeeg_ca_common_outer)
        # xeog_ca_common_outer = self.ln_post_eog(xeog_ca_common_outer)

        x_common = xeeg_ca_common_outer + xeog_ca_common_outer
        output["output_features"]["combined"] = x_common

        return output

    def forward_sole(self, xeeg, xeog, output, **kwargs):

        return_matches = kwargs["return_matches"] if "return_matches" in kwargs else False

        xeeg_sole = self.inner_tf_eeg.forward_ca(xeeg)[:, :, :1]
        xeog_sole = self.inner_tf_eog.forward_ca(xeog)[:, :, :1]

        xeeg_cls_sole = xeeg_sole[:, :, :1]
        xeog_cls_sole = xeog_sole[:, :, :1]

        # if self.pos:
        #     xeeg_cls_sole = self.outer_positional_embedding(xeeg_cls_sole)
        #     xeog_cls_sole = self.outer_positional_embedding(xeog_cls_sole)

        xeeg_outer_sole = self.outer_tf_eeg_out.forward_ca(xeeg_cls_sole)
        xeog_outer_sole = self.outer_tf_eog_out.forward_ca(xeog_cls_sole)

        # xeeg_outer_sole = self.ln_post_eeg(xeeg_outer_sole)
        # xeog_outer_sole = self.ln_post_eog(xeog_outer_sole)

        output["output_features"]["eeg"] = xeeg_outer_sole
        output["output_features"]["inner_eeg"] = xeeg_sole
        output["output_features"]["eog"] = xeog_outer_sole
        output["output_features"]["inner_eog"] = xeog_sole

        xeeg_sa_o, xeog_sa_o = xeeg_outer_sole.squeeze(), xeog_outer_sole.squeeze()

        x_match = None
        if return_matches:
            if len(xeeg_sa_o.shape) == 3 and len(xeog_sa_o.shape) == 3:

                # xeeg_sa_o = torch.einsum('b o f , f p -> b o p', xeeg_sa_o, self.eeg_proj_head)
                # xeog_sa_o = torch.einsum('b o f , f p -> b o p', xeog_sa_o, self.eog_proj_head)

                # normalized features
                xeeg_sa_o = xeeg_sa_o / xeeg_sa_o.norm(dim=1, keepdim=True)
                xeog_sa_o = xeog_sa_o / xeog_sa_o.norm(dim=1, keepdim=True)

                # cosine similarity as logits
                logit_scale = self.logit_scale.exp()

                x_match_eeg = torch.einsum('b o f , b m f -> b o m', xeeg_sa_o, xeog_sa_o)
                x_match_eog = torch.einsum('b o f , b m f -> b o m', xeeg_sa_o, xeog_sa_o).permute(0,2,1)

                x_match = {"stft_eeg":x_match_eeg, "stft_eog": x_match_eog}
            else:
                x_match = torch.Tensor([0]).to(xeeg.device)

        output["matches"] = x_match

        return output

class SleepEnc_BLIP_EEG_EOG_twomode_caouter_shared_type1(nn.Module):
    def __init__(self, args, encs=[None]):
        """
        :param encoder_filters_small, encoder_filters_big: filters of CNN output, quantized by 20
        :param encs_small, encs_big:
        """
        super().__init__()
        self.args = args

        # self.enc_0 = encs[0]
        # self.enc_1 = encs[0]

        d_model = args.dmodel  # 64*8

        inner_biased = args.inner_biased if "inner_biased" in args else False
        outer_biased = args.outer_biased if "outer_biased" in args else False
        rpos = args.rpos if "rpos" in args else False
        self.pos = args.pos if "pos" in args else True
        dim_proj = args.dim_proj if "dim_proj" in args else 128
        self.outer_rep = args.outer_rep if "outer_rep" in args else False
        self.skip_percentile = args.skip_percentile if "skip_percentile" in args else False
        self.mod_token = args.mod_token if "mod_token" in args else False

        if inner_biased == "gaussian_learned":
            inner_biased = Gaussian_Learned_Attention_Bias(d_model, heads=8, type="mul")

        # if outer_biased == "gaussian_learned":
        #     outer_biased = Gaussian_Learned_Attention_Bias(d_model, heads=8, type="mul")

        self.inner_tf_eeg = inner_ch_SA_CA_shared_v2(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, dim_proj=dim_proj, gbiased=inner_biased, num_layers=4)
        self.inner_tf_eog = inner_ch_SA_CA_shared_v2(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, dim_proj=dim_proj, gbiased=inner_biased, num_layers=4)

        self.outer_tf_eeg_out = outer_mod_ch_SA_CA_shared_v2(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, dim_proj=dim_proj, gbiased=outer_biased, num_layers=4)
        self.outer_tf_eog_out = outer_mod_ch_SA_CA_shared_v2(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, dim_proj=dim_proj, gbiased=outer_biased, num_layers=4)

        self.cls_token_eeg = nn.Parameter(torch.randn(1, 1, 1, 1, 1, d_model))
        self.cls_token_eog = nn.Parameter(torch.randn(1, 1, 1, 1, 1, d_model))

        self.cls_token_outer_eeg = nn.Parameter(torch.randn(1, 1, 1, 1, 1, d_model))
        self.cls_token_outer_eog = nn.Parameter(torch.randn(1, 1, 1, 1, 1, d_model))

        # if self.skip_random_mod:
        #     self.skip_dropout = nn.Dropout()
        if self.pos:
            self.inner_positional_embedding = huy_pos_inner(d_model, pos=False, inner=29, outer=21, modalities=1, channels=1)
            self.outer_positional_embedding = huy_pos_outer(d_model, pos=False, inner=29, outer=21, modalities=1, channels=1)

        if self.mod_token:
            self.modtype_token = modtype_embedding(num_modalities=2, dim=d_model)

    def forward(self, x, skip_modality="random", **kwargs):

        xeeg = x["stft_eeg"][:, :, :, :, 1:, :]  # mat
        xeog = x["stft_eog"][:, :, :, :, 1:, :]  # mat

        xeeg = einops.rearrange(xeeg, "b outer mod ch f inner -> b outer inner mod ch f")
        xeog = einops.rearrange(xeog, "b outer mod ch f inner -> b outer inner mod ch f")

        if self.mod_token:
            xeeg = self.modtype_token(data=xeeg, mod_num=0)
            xeog = self.modtype_token(data=xeog, mod_num=1)

        if self.pos:
            xeeg = self.inner_positional_embedding(xeeg)
            xeog = self.inner_positional_embedding(xeog)

        cls_token_eeg = self.cls_token_eeg.repeat(xeeg.shape[0], xeeg.shape[1], 1, 1, xeeg.shape[3], 1)
        xeeg = torch.cat([cls_token_eeg, xeeg], dim=2)

        cls_token_eog = self.cls_token_eog.repeat(xeog.shape[0], xeog.shape[1], 1, 1, xeog.shape[3], 1)
        xeog = torch.cat([cls_token_eog, xeog], dim=2)

        xeeg_common = xeeg
        xeog_common = xeog

        skip_modality = self.calculate_skip_modality( num_batches=xeeg.shape[0], skip_modality=skip_modality)
        if skip_modality is not None:
            xeeg_common = xeeg[skip_modality==0] #Keep epochs where both mods are available
            xeog_common = xeog[skip_modality==0]

            xeeg_sole = xeeg[skip_modality==2] #Keep epochs where eog is not available
            xeog_sole = xeog[skip_modality==1] #Keep epochs where eeg is not available
        else:
            skip_modality = torch.zeros(xeeg.shape[0])

        output = {}
        if xeeg_common.shape[0]>0 and xeog_common.shape[0]>0:
            x_common, x_match = self.forward_common(xeeg=xeeg_common, xeog=xeog_common, skip_modality=skip_modality, output=output, **kwargs)
        if "xeeg_sole" in locals() and xeeg_sole.shape[0]>0:
            xeeg_sole = self.forward_eeg(xeeg=xeeg_sole, skip_modality=skip_modality, output=output)
        if "xeog_sole" in locals() and xeog_sole.shape[0]>0:
            xeog_sole = self.forward_eog(xeog=xeog_sole, skip_modality=skip_modality, output=output)

        if ("xeeg_sole" in locals() and xeeg_sole.shape[0]>0) or ( "xeog_sole" in locals() and xeog_sole.shape[0]>0):
            x = []
            counter = [0, 0, 0]
            for i in range(len(skip_modality)):
                if skip_modality[i] == 0:
                    x.append(x_common[counter[int(skip_modality[i].item())]:counter[int(skip_modality[i].item())] + 1])
                elif skip_modality[i] == 1:
                    x.append(xeog_sole[
                             counter[int(skip_modality[i].item())]:counter[int(skip_modality[i].item())] + 1])
                elif skip_modality[i] == 2:
                    x.append(xeeg_sole[
                             counter[int(skip_modality[i].item())]:counter[int(skip_modality[i].item())] + 1])
                counter[int(skip_modality[i].item())] += 1
            x = torch.cat(x, dim=0)
        else:
            x = x_common
        output["output_features"]= x
        return_matches = kwargs["return_matches"] if "return_matches" in kwargs else False
        if return_matches and "x_match" in locals():
            output["matches"] = x_match
        print(output["output_features"].shape)
        return output

    def calculate_skip_modality(self, num_batches,skip_modality):
        if  self.training:

            skip_modality = torch.rand(num_batches)
            skip_modality[skip_modality > 1 - self.skip_percentile["eeg"]] = 2
            skip_modality[skip_modality < self.skip_percentile["eog"]] = 1
            skip_modality = skip_modality.int()
        else:
            if skip_modality=="full":
                skip_modality = None
            elif skip_modality=="random":
                skip_modality = torch.rand(num_batches)
                skip_modality[skip_modality > 0.66] = 2
                skip_modality[skip_modality < 0.33] = 1
                skip_modality = skip_modality.int()
            elif skip_modality=="eeg":
                skip_modality = torch.ones(num_batches)*1
            elif skip_modality=="eog":
                skip_modality = torch.ones(num_batches)*2
            elif "percentile" in skip_modality:
                skip_eeg = skip_modality.split("_")[0]
                skip_eog = skip_modality.split("_")[1]

                skip_modality = torch.rand(num_batches)
                skip_modality[skip_modality > 1 - skip_eeg] = 2
                skip_modality[skip_modality < skip_eog] = 1
                skip_modality = skip_modality.int()

            elif skip_modality is "vae":
                raise not NotImplementedError
                # with torch.no_grad():
                #     vae_eeg_output = self.enc_0([x[0]])
                #     output_losses = self.enc_0.module.loss_function(vae_eeg_output[0], vae_eeg_output[1],
                #                                                     vae_eeg_output[2], vae_eeg_output[3],
                #                                                     reduction="none")
                #     eeg_routing = output_losses["total"].flatten(start_dim=1).mean(dim=1)
                #     # print(eeg_routing)
                #     eeg_routing[eeg_routing < 3] = 0
                #     eeg_routing[eeg_routing > 3] = 1
                #
                #     vae_eog_output = self.enc_1([x[1]])
                #     output_losses = self.enc_1.module.loss_function(vae_eog_output[0], vae_eog_output[1],
                #                                                     vae_eog_output[2], vae_eog_output[3],
                #                                                     reduction="none")
                #     eog_routing = output_losses["total"].flatten(start_dim=1).mean(dim=1)
                #     # print(eog_routing)
                #     eog_routing[eog_routing < 3] = 0
                #     eog_routing[eog_routing > 3] = 2
                #
                #     skip_modality = eeg_routing + eog_routing
                #     skip_modality[skip_modality == 3] = 0
        return skip_modality

    def forward_common(self, xeeg, xeog, skip_modality, output, **kwargs):

        return_matches = kwargs["return_matches"] if "return_matches" in kwargs else False
        return_inter_reps = kwargs["return_inter_reps"] if "return_inter_reps" in kwargs else False
        return_order = kwargs["return_order"] if "return_order" in kwargs else False

        xeeg_common_i = self.inner_tf_eeg.forward_sa(xeeg)
        xeog_common_i = self.inner_tf_eog.forward_sa(xeog)

        # xeeg_sa_o = self.outer_positional_embedding(xeeg_sa_o)
        # xeog_sa_o = self.outer_positional_embedding(xeog_sa_o)
        xeeg_common_outer = self.outer_tf_eeg_out.forward_sa(xeeg_common_i)[:, :, :1]
        xeog_common_outer = self.outer_tf_eog_out.forward_sa(xeog_common_i)[:, :, :1]

        xeeg_sa_o, xeog_sa_o = xeeg_common_outer.squeeze(), xeog_common_outer.squeeze()

        if return_matches:
            if ("xeeg_sa_o" in locals() and xeeg_sa_o.shape[0] > 0 and len(xeeg_sa_o.shape) == 3) and (
                    "xeog_sa_o" in locals() and xeog_sa_o.shape[0] > 0 and len(xeog_sa_o.shape) == 3):
                x_match = torch.einsum('b o f , b m f -> b o m', xeeg_sa_o, xeog_sa_o)
            else:
                x_match = torch.Tensor([0]).to(xeeg.device)
        else:
            x_match = None

        xeeg_ca_common = self.inner_tf_eeg.forward_ca(xeeg, xeog_common_i)[:, :, :1]
        xeog_ca_common = self.inner_tf_eog.forward_ca(xeog, xeeg_common_i)[:, :, :1]

        if self.pos:
            xeeg_ca_common = self.outer_positional_embedding(xeeg_ca_common)
            xeog_ca_common = self.outer_positional_embedding(xeog_ca_common)

        # cls_token_outer_eeg = self.cls_token_outer_eeg.repeat(xeeg_ca_common.shape[0], xeeg_ca_common.shape[1], 1, 1, 1, 1)
        # cls_token_outer_eog = self.cls_token_outer_eog.repeat(xeog_ca_common.shape[0], xeog_ca_common.shape[1], 1, 1, 1, 1)
        #
        # xeeg_ca_common = torch.cat([cls_token_outer_eeg, xeog_ca_common], dim=3)
        # xeog_ca_common = torch.cat([cls_token_outer_eog, xeog_ca_common], dim=3)

        xeeg_ca_common_outer = self.outer_tf_eeg_out.forward_ca(xeeg_ca_common, xeog_common_outer)
        xeog_ca_common_outer = self.outer_tf_eog_out.forward_ca(xeog_ca_common, xeeg_common_outer)

        x_common = xeeg_ca_common_outer + xeog_ca_common_outer
        # output["output_features"][skip_modality == 0] = x_common
        #
        # if return_matches: output["matches"] = x_match
        # if return_inter_reps: output["intermediate_reps"] = [xeeg_sa_o, xeeg_sa_o]
        # if return_order: output["order"] = torch.cat([xeeg_sa_o, xeog_sa_o], dim=3)
        return x_common, x_match

    def forward_eeg(self, xeeg, skip_modality, output):
        xeeg_ca_sole = self.inner_tf_eeg.forward_ca(xeeg)[:, :, :1]
        if self.pos:
            xeeg_ca_sole = self.outer_positional_embedding(xeeg_ca_sole)
        # cls_token_outer_eeg = self.cls_token_outer.repeat(xeeg_ca_sole.shape[0], xeeg_ca_sole.shape[1], 1, 1, 1, 1)
        # xeeg_ca_sole = torch.cat([cls_token_outer_eeg, xeeg_ca_sole], dim=3)
        xeeg_ca_sole = self.outer_tf_eeg_out.forward_ca(xeeg_ca_sole)[:, :, :, :1]
        # output["output_features"][skip_modality == 2] = xeeg_ca_sole
        return xeeg_ca_sole

    def forward_eog(self, xeog, skip_modality, output):
        xeog_ca_sole = self.inner_tf_eog.forward_ca(xeog)[:, :, :1]
        if self.pos:
            xeog_ca_sole = self.outer_positional_embedding(xeog_ca_sole)
        # cls_token_outer_eog = self.cls_token_outer.repeat(xeog_ca_sole.shape[0], xeog_ca_sole.shape[1], 1, 1, 1, 1)
        # xeog_ca_sole = torch.cat([cls_token_outer_eog, xeog_ca_sole], dim=3)
        xeog_ca_sole = self.outer_tf_eog_out.forward_ca(xeog_ca_sole)[:, :, :, :1]
        # output["output_features"][skip_modality == 1] = xeog_ca_sole
        return xeog_ca_sole

class SleepEnc_BLIP_EEG_EOG_EMG_twomode_caouter_shared(nn.Module):
    def __init__(self, args, encs=[None]):
        """
        :param encoder_filters_small, encoder_filters_big: filters of CNN output, quantized by 20
        :param encs_small, encs_big:
        """
        super().__init__()
        self.args = args

        # self.enc_0 = encs[0]
        # self.enc_1 = encs[0]

        d_model = args.dmodel  # 64*8

        inner_biased = args.inner_biased if "inner_biased" in args else False
        outer_biased = args.outer_biased if "outer_biased" in args else False
        rpos = args.rpos if "rpos" in args else False
        self.pos = args.pos if "pos" in args else True
        dim_proj = args.dim_proj if "dim_proj" in args else 128
        self.outer_rep = args.outer_rep if "outer_rep" in args else False
        self.skip_random_mod = args.skip_random_mod if "skip_random_mod" in args else False

        if inner_biased == "gaussian_learned":
            inner_biased = Gaussian_Learned_Attention_Bias(d_model, heads=8, type="mul")

        # if outer_biased == "gaussian_learned":
        #     outer_biased = Gaussian_Learned_Attention_Bias(d_model, heads=8, type="mul")

        self.inner_tf_eeg = inner_ch_SA_CA_CA_shared_v2(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, dim_proj=dim_proj, gbiased=inner_biased, num_layers=4)
        self.inner_tf_eog = inner_ch_SA_CA_CA_shared_v2(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, dim_proj=dim_proj, gbiased=inner_biased, num_layers=4)
        self.inner_tf_emg = inner_ch_SA_CA_CA_shared_v2(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, dim_proj=dim_proj, gbiased=inner_biased, num_layers=4)

        self.outer_tf_eeg_out = outer_mod_ch_SA_CA_CA_shared_v2(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, dim_proj=dim_proj, gbiased=outer_biased, num_layers=4)
        self.outer_tf_eog_out = outer_mod_ch_SA_CA_CA_shared_v2(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, dim_proj=dim_proj, gbiased=outer_biased, num_layers=4)
        self.outer_tf_emg_out = outer_mod_ch_SA_CA_CA_shared_v2(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, dim_proj=dim_proj, gbiased=outer_biased, num_layers=4)

        self.cls_token_eeg = nn.Parameter(torch.randn(1, 1, 1, 1, 1, d_model))
        self.cls_token_eog = nn.Parameter(torch.randn(1, 1, 1, 1, 1, d_model))
        self.cls_token_emg = nn.Parameter(torch.randn(1, 1, 1, 1, 1, d_model))
        self.cls_token_outer_eeg = nn.Parameter(torch.randn(1, 1, 1, 1, 1, d_model))
        self.cls_token_outer_eog = nn.Parameter(torch.randn(1, 1, 1, 1, 1, d_model))
        self.cls_token_outer_emg = nn.Parameter(torch.randn(1, 1, 1, 1, 1, d_model))

        # if self.skip_random_mod:
        #     self.skip_dropout = nn.Dropout()
        if self.pos:
            self.inner_positional_embedding = huy_pos_inner(d_model, pos=False, inner=29, outer=21, modalities=1,
                                                              channels=1)
            self.outer_positional_embedding = huy_pos_outer(d_model, pos=False, inner=29, outer=21, modalities=1,
                                                            channels=1)

    def forward(self, x, return_matches=False, extract_norm=False, return_inter_reps=False, return_order=False, return_reconstruction = False, skip_modality="random"):

        xeeg = x["stft_eeg"][:, :, :, :, 1:, :]  # mat
        xeog = x["stft_eog"][:, :, :, :, 1:, :]  # mat
        xemg = x["stft_emg"][:, :, :, :, 1:, :]  # mat

        xeeg = einops.rearrange(xeeg, "b outer mod ch f inner -> b outer inner mod ch f")
        xeog = einops.rearrange(xeog, "b outer mod ch f inner -> b outer inner mod ch f")
        xemg = einops.rearrange(xemg, "b outer mod ch f inner -> b outer inner mod ch f")

        if self.pos:
            xeeg = self.inner_positional_embedding(xeeg)
            xeog = self.inner_positional_embedding(xeog)
            xemg = self.inner_positional_embedding(xemg)

        cls_token_eeg = self.cls_token_eeg.repeat(xeeg.shape[0], xeeg.shape[1], 1, 1, xeeg.shape[3], 1)
        xeeg = torch.cat([cls_token_eeg, xeeg], dim=2)

        cls_token_eog = self.cls_token_eog.repeat(xeog.shape[0], xeog.shape[1], 1, 1, xeog.shape[3], 1)
        xeog = torch.cat([cls_token_eog, xeog], dim=2)

        cls_token_emg = self.cls_token_emg.repeat(xemg.shape[0], xemg.shape[1], 1, 1, xemg.shape[3], 1)
        xemg = torch.cat([cls_token_emg, xemg], dim=2)

        xeeg_common_init = xeeg
        xeog_common_init = xeog
        xemg_common_init = xemg

        xeeg_common = xeeg
        xeog_common = xeog
        xemg_common = xemg

        if self.skip_random_mod:
            if  self.training:
                # skip_modality_eeg = torch.bernoulli(torch.ones(xeeg.shape[0])*0.5)
                # skip_modality_eog = torch.bernoulli(torch.ones(xeog.shape[0])*0.5)
                # skip_modality = skip_modality_eeg*2 + skip_modality_eog
                # skip_modality[skip_modality==3] = 0

                skip_modality = torch.rand(xeeg.shape[0])
                # skip_modality[skip_modality>0.5] = 2
                # skip_modality[skip_modality<=0.5] = 1
                skip_modality[skip_modality < 0.25] = 3
                skip_modality[skip_modality < 0.5] = 2
                skip_modality[skip_modality < 0.75] = 1
                skip_modality[skip_modality <= 1.1] = 0
                skip_modality = skip_modality.int()
                # skip_modality[skip_modality < 1.5 ]=0
                # skip_modality = None

            else:
                if skip_modality=="full":
                    skip_modality = None
                elif skip_modality=="random":
                    skip_modality = torch.rand(xeeg.shape[0])
                    # skip_modality[skip_modality > 0.5] = 2
                    # skip_modality[skip_modality <= 0.5] = 1
                    skip_modality[skip_modality < 0.25] = 3
                    skip_modality[skip_modality < 0.5] = 2
                    skip_modality[skip_modality < 0.75] = 1
                    skip_modality[skip_modality <= 1.1] = 0
                    skip_modality = skip_modality.int()
                    # skip_modality[skip_modality == 0] = 1

                elif skip_modality=="eeg":
                    skip_modality = torch.ones(xeeg.shape[0])*1
                elif skip_modality=="eog":
                    skip_modality = torch.ones(xeeg.shape[0])*2
                elif skip_modality=="emg":
                    skip_modality = torch.ones(xeeg.shape[0])*3
                elif skip_modality is "vae":
                    raise NotImplementedError
                    with torch.no_grad():
                        vae_eeg_output = self.enc_0([x[0]])
                        output_losses = self.enc_0.module.loss_function(vae_eeg_output[0], vae_eeg_output[1],
                                                                        vae_eeg_output[2], vae_eeg_output[3],
                                                                        reduction="none")
                        eeg_routing = output_losses["total"].flatten(start_dim=1).mean(dim=1)
                        # print(eeg_routing)
                        eeg_routing[eeg_routing < 3] = 0
                        eeg_routing[eeg_routing > 3] = 1

                        vae_eog_output = self.enc_1([x[1]])
                        output_losses = self.enc_1.module.loss_function(vae_eog_output[0], vae_eog_output[1],
                                                                        vae_eog_output[2], vae_eog_output[3],
                                                                        reduction="none")
                        eog_routing = output_losses["total"].flatten(start_dim=1).mean(dim=1)
                        # print(eog_routing)
                        eog_routing[eog_routing < 3] = 0
                        eog_routing[eog_routing > 3] = 2

                        skip_modality = eeg_routing + eog_routing
                        skip_modality[skip_modality == 3] = 0

            if skip_modality is not None:
                # xeeg_common = xeeg_sa[np.logical_or(skip_modality==0, skip_modality==1)] #Process EEG with EOG if you are not skipping EOG
                xeeg_common = xeeg[skip_modality==0] #Process EEG with EOG if you are not skipping EOG
                xeeg_common_init = xeeg[skip_modality==0] #Process EEG with EOG if you are not skipping EOG
                # xeeg_undir_sole = xeeg_sa[skip_modality==1] #If you skip EOG process EEG on its own
                xeeg_dir_sole = xeeg[skip_modality==2] #If you skip EOG process EEG on its own
                xeog_common = xeog[skip_modality==0] #Process EOG with EEG if you are not skipping EEG
                xeog_common_init = xeog[skip_modality==0] #Process EOG with EEG if you are not skipping EEG
                # xeog_undir_sole = xeog_sa[skip_modality==2] #If you skip EEG process EOG on its own
                xeog_dir_sole = xeog[skip_modality==1] #If you skip EEG process EOG on its own

                xemg_common = xemg[skip_modality==0] #Process EOG with EEG if you are not skipping EEG
                xemg_common_init = xemg[skip_modality==0] #Process EOG with EEG if you are not skipping EEG
                # xeog_undir_sole = xeog_sa[skip_modality==2] #If you skip EEG process EOG on its own
                xemg_dir_sole = xemg[skip_modality==3] #If you skip EEG process EOG on its own

            # print(skip_modality)

        if xeeg_common.shape[0]>0 and xeog_common.shape[0]>0 and xemg_common.shape[0]>0 :
            xeeg_common = self.inner_tf_eeg.forward_sa(xeeg_common)
            xeog_common = self.inner_tf_eog.forward_sa(xeog_common)
            xemg_common = self.inner_tf_emg.forward_sa(xemg_common)

            # xeeg_sa_o = self.outer_positional_embedding(xeeg_sa_o)
            # xeog_sa_o = self.outer_positional_embedding(xeog_sa_o)
            xeeg_common_outer = self.outer_tf_eeg_out.forward_sa(xeeg_common, extract_norm=extract_norm)[:, :, :1]
            xeog_common_outer = self.outer_tf_eog_out.forward_sa(xeog_common, extract_norm=extract_norm)[:, :, :1]
            xemg_common_outer = self.outer_tf_emg_out.forward_sa(xemg_common, extract_norm=extract_norm)[:, :, :1]

            xeeg_sa_o, xeog_sa_o, xemg_sa_o = xeeg_common_outer.squeeze(), xeog_common_outer.squeeze(), xemg_common_outer.squeeze()

            xeeg_ca_common = self.inner_tf_eeg.forward_ca(xeeg_common_init, xeog_common, xemg_common)[:, :, :1]
            xeog_ca_common = self.inner_tf_eog.forward_ca(xeog_common_init, xeeg_common, xemg_common)[:, :, :1]
            xemg_ca_common = self.inner_tf_eog.forward_ca(xeog_common_init, xeeg_common, xemg_common)[:, :, :1]

            if self.pos:
                xeeg_ca_common = self.outer_positional_embedding(xeeg_ca_common)
                xeog_ca_common = self.outer_positional_embedding(xeog_ca_common)
                xemg_ca_common = self.outer_positional_embedding(xemg_ca_common)

            # cls_token_outer_eeg = self.cls_token_outer_eeg.repeat(xeeg_ca_common.shape[0], xeeg_ca_common.shape[1], 1, 1, 1, 1)
            # cls_token_outer_eog = self.cls_token_outer_eog.repeat(xeog_ca_common.shape[0], xeog_ca_common.shape[1], 1, 1, 1, 1)
            #
            # xeeg_ca_common = torch.cat([cls_token_outer_eeg, xeog_ca_common], dim=3)
            # xeog_ca_common = torch.cat([cls_token_outer_eog, xeog_ca_common], dim=3)

            xeeg_ca_common_outer = self.outer_tf_eeg_out.forward_ca(xeeg_ca_common, xeog_common_outer, xemg_common_outer, extract_norm=extract_norm)
            xeog_ca_common_outer = self.outer_tf_eog_out.forward_ca(xeog_ca_common, xeeg_common_outer, xemg_common_outer, extract_norm=extract_norm)
            xemg_ca_common_outer = self.outer_tf_eog_out.forward_ca(xemg_ca_common, xeeg_common_outer, xeog_common_outer, extract_norm=extract_norm)

            x_common = xeeg_ca_common_outer + xeog_ca_common_outer + xemg_ca_common_outer

        if "xeeg_dir_sole" in locals() and xeeg_dir_sole.shape[0]>0:
            xeeg_ca_sole = self.inner_tf_eeg.forward_ca(xeeg_dir_sole)[:, :, :1]

            if self.pos:
                xeeg_ca_sole = self.outer_positional_embedding(xeeg_ca_sole)
            # cls_token_outer_eeg = self.cls_token_outer.repeat(xeeg_ca_sole.shape[0], xeeg_ca_sole.shape[1], 1, 1, 1, 1)
            # xeeg_ca_sole = torch.cat([cls_token_outer_eeg, xeeg_ca_sole], dim=3)
            xeeg_ca_sole = self.outer_tf_eeg_out.forward_ca(xeeg_ca_sole, extract_norm=extract_norm)[:, :, :, :1]

        if "xeog_dir_sole" in locals() and xeog_dir_sole.shape[0]>0:
            xeog_ca_sole = self.inner_tf_eog.forward_ca(xeog_dir_sole)[:, :, :1]
            if self.pos:
                xeog_ca_sole = self.outer_positional_embedding(xeog_ca_sole)
            # cls_token_outer_eog = self.cls_token_outer.repeat(xeog_ca_sole.shape[0], xeog_ca_sole.shape[1], 1, 1, 1, 1)
            # xeog_ca_sole = torch.cat([cls_token_outer_eog, xeog_ca_sole], dim=3)
            xeog_ca_sole = self.outer_tf_eog_out.forward_ca(xeog_ca_sole, extract_norm=extract_norm)[:, :, :, :1]

        if "xemg_dir_sole" in locals() and xemg_dir_sole.shape[0]>0:
            xemg_ca_sole = self.inner_tf_emg.forward_ca(xemg_dir_sole)[:, :, :1]
            if self.pos:
                xemg_ca_sole = self.outer_positional_embedding(xemg_ca_sole)
            # cls_token_outer_eog = self.cls_token_outer.repeat(xeog_ca_sole.shape[0], xeog_ca_sole.shape[1], 1, 1, 1, 1)
            # xeog_ca_sole = torch.cat([cls_token_outer_eog, xeog_ca_sole], dim=3)
            xemg_ca_sole = self.outer_tf_emg_out.forward_ca(xemg_ca_sole, extract_norm=extract_norm)[:, :, :, :1]


        if ("xeeg_dir_sole" in locals() and xeeg_dir_sole.shape[0]>0 ) or ( "xeog_dir_sole" in locals() and xeog_dir_sole.shape[0]>0 ) or ("xemg_dir_sole" in locals() and xemg_dir_sole.shape[0]>0 ):
            x = []
            counter = [0, 0, 0, 0]
            for i in range(len(skip_modality)):
                if skip_modality[i] == 0:
                    x.append(x_common[counter[int(skip_modality[i].item())]:counter[int(skip_modality[i].item())] + 1])
                elif skip_modality[i] == 1:
                    x.append(xeog_ca_sole[counter[int(skip_modality[i].item())]:counter[int(skip_modality[i].item())] + 1])
                elif skip_modality[i] == 2:
                    x.append(xeeg_ca_sole[counter[int(skip_modality[i].item())]:counter[int(skip_modality[i].item())] + 1])
                elif skip_modality[i] == 3:
                    x.append(xemg_ca_sole[counter[int(skip_modality[i].item())]:counter[int(skip_modality[i].item())] + 1])
                counter[int(skip_modality[i].item())] += 1
            x = torch.cat(x, dim=0)
        else:
            x = x_common


        output={"output_features": x}

        if return_matches:
            if ("xeeg_sa_o" in locals() and xeeg_sa_o.shape[0]>0 and len(xeeg_sa_o.shape)==3) and ("xeog_sa_o" in locals() and xeog_sa_o.shape[0]>0 and len(xeog_sa_o.shape)==3):
                x_match = torch.einsum('b o f , b m f -> b o m', xeeg_sa_o, xeog_sa_o)
                output["matches"] = x_match
            else:
                output["matches"] = None
        if return_inter_reps:
            output["intermediate_reps"] = [xeeg_sa_o, xeeg_sa_o]
        if return_order:
            x_sa_o = torch.cat([xeeg_sa_o, xeog_sa_o], dim=3)
            output["order"] =x_sa_o

        return output
class SleepEnc_BLIP_EEG_EOG_twomode_caouter_shared_VAE(nn.Module):
    def __init__(self, args, encs=[None]):
        """
        :param encoder_filters_small, encoder_filters_big: filters of CNN output, quantized by 20
        :param encs_small, encs_big:
        """
        super().__init__()
        self.args = args

        d_model = args.dmodel  # 64*8

        inner_biased = args.inner_biased if "inner_biased" in args else False
        outer_biased = args.outer_biased if "outer_biased" in args else False
        rpos = args.rpos if "rpos" in args else False
        self.pos = args.pos if "pos" in args else True
        dim_proj = args.dim_proj if "dim_proj" in args else 128
        self.outer_rep = args.outer_rep if "outer_rep" in args else False
        self.skip_random_mod = args.skip_random_mod if "skip_random_mod" in args else False

        if inner_biased == "gaussian_learned":
            inner_biased = Gaussian_Learned_Attention_Bias(d_model, heads=8, type="mul")

        # if outer_biased == "gaussian_learned":
        #     outer_biased = Gaussian_Learned_Attention_Bias(d_model, heads=8, type="mul")

        self.inner_tf_eeg = inner_ch_SA_CA_shared_v2(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, dim_proj=dim_proj, gbiased=inner_biased, num_layers=4)
        self.inner_tf_eog = inner_ch_SA_CA_shared_v2(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, dim_proj=dim_proj, gbiased=inner_biased, num_layers=4)

        self.outer_tf_eeg_out = outer_mod_ch_SA_CA_shared_v2(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, dim_proj=dim_proj, gbiased=outer_biased, num_layers=4)
        self.outer_tf_eog_out = outer_mod_ch_SA_CA_shared_v2(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, dim_proj=dim_proj, gbiased=outer_biased, num_layers=4)

        self.cls_token_eeg = nn.Parameter(torch.randn(1, 1, 1, 1, 1, d_model))
        self.cls_token_eog = nn.Parameter(torch.randn(1, 1, 1, 1, 1, d_model))
        self.cls_token_outer_eeg = nn.Parameter(torch.randn(1, 1, 1, 1, 1, d_model))
        self.cls_token_outer_eog = nn.Parameter(torch.randn(1, 1, 1, 1, 1, d_model))

        # if self.skip_random_mod:
        #     self.skip_dropout = nn.Dropout()
        if self.pos:
            self.inner_positional_embedding = huy_pos_inner(d_model, pos=False, inner=29, outer=21, modalities=1,
                                                              channels=1)
            self.outer_positional_embedding = huy_pos_outer(d_model, pos=False, inner=29, outer=21, modalities=1,
                                                            channels=1)

        self.token_to_latent_eeg = nn.Linear(d_model, d_model)
        self.token_to_latent_eog = nn.Linear(d_model, d_model)
        self.tanh = nn.Tanh()
        self.mask_token_eeg = nn.Parameter(torch.randn(1, 1, 1, 1, 1, d_model))
        self.mask_token_eog = nn.Parameter(torch.randn(1, 1, 1, 1, 1, d_model))
        self.deconstruct_inner_tf_eeg = inner_ch_att_RA(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, dim_proj=dim_proj, gbiased=inner_biased, num_layers=4)
        self.deconstruct_inner_tf_eog = inner_ch_att_RA(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, dim_proj=dim_proj, gbiased=inner_biased, num_layers=4)

    def forward(self, x, return_matches=False, extract_norm=False, return_inter_reps=False, return_order=False, return_reconstruction = False, skip_modality="random"):

        xeeg = x[0][:, :, :, :, 1:, :]  # mat
        xeog = x[1][:, :, :, :, 1:, :]  # mat

        xeeg = einops.rearrange(xeeg, "b outer mod ch f inner -> b outer inner mod ch f")
        xeog = einops.rearrange(xeog, "b outer mod ch f inner -> b outer inner mod ch f")

        if self.pos:
            xeeg = self.inner_positional_embedding(xeeg)
            xeog = self.inner_positional_embedding(xeog)

        cls_token_eeg = self.cls_token_eeg.repeat(xeeg.shape[0], xeeg.shape[1], 1, 1, xeeg.shape[3], 1)
        xeeg = torch.cat([cls_token_eeg, xeeg], dim=2)

        cls_token_eog = self.cls_token_eog.repeat(xeog.shape[0], xeog.shape[1], 1, 1, xeog.shape[3], 1)
        xeog = torch.cat([cls_token_eog, xeog], dim=2)

        xeeg_common_init = xeeg
        xeog_common_init = xeog

        xeeg_common = xeeg
        xeog_common = xeog

        if self.skip_random_mod:
            if  self.training:
                # skip_modality_eeg = torch.bernoulli(torch.ones(xeeg.shape[0])*0.5)
                # skip_modality_eog = torch.bernoulli(torch.ones(xeog.shape[0])*0.5)
                # skip_modality = skip_modality_eeg*2 + skip_modality_eog
                # skip_modality[skip_modality==3] = 0

                skip_modality = torch.rand(xeeg.shape[0])
                # skip_modality[skip_modality>0.5] = 2
                # skip_modality[skip_modality<=0.5] = 1
                skip_modality[skip_modality > 0.66] = 2
                skip_modality[skip_modality < 0.33] = 1
                skip_modality = skip_modality.int()
                # skip_modality[skip_modality < 1.5 ]=0
                # skip_modality = None

            else:
                if skip_modality=="full":
                    skip_modality = None
                elif skip_modality=="random":
                    skip_modality = torch.rand(xeeg.shape[0])
                    # skip_modality[skip_modality > 0.5] = 2
                    # skip_modality[skip_modality <= 0.5] = 1
                    skip_modality[skip_modality > 0.66] = 2
                    skip_modality[skip_modality < 0.33] = 1
                    skip_modality = skip_modality.int()
                    # skip_modality[skip_modality == 0] = 1

                elif skip_modality=="eeg":
                    skip_modality = torch.ones(xeeg.shape[0])*1
                elif skip_modality=="eog":
                    skip_modality = torch.ones(xeeg.shape[0])*2
                elif skip_modality is "vae":
                    with torch.no_grad():
                        vae_eeg_output = self.enc_0([x[0]])
                        output_losses = self.enc_0.module.loss_function(vae_eeg_output[0], vae_eeg_output[1],
                                                                        vae_eeg_output[2], vae_eeg_output[3],
                                                                        reduction="none")
                        eeg_routing = output_losses["total"].flatten(start_dim=1).mean(dim=1)
                        # print(eeg_routing)
                        eeg_routing[eeg_routing < 3] = 0
                        eeg_routing[eeg_routing > 3] = 1

                        vae_eog_output = self.enc_1([x[1]])
                        output_losses = self.enc_1.module.loss_function(vae_eog_output[0], vae_eog_output[1],
                                                                        vae_eog_output[2], vae_eog_output[3],
                                                                        reduction="none")
                        eog_routing = output_losses["total"].flatten(start_dim=1).mean(dim=1)
                        # print(eog_routing)
                        eog_routing[eog_routing < 3] = 0
                        eog_routing[eog_routing > 3] = 2

                        skip_modality = eeg_routing + eog_routing
                        skip_modality[skip_modality == 3] = 0

            if skip_modality is not None:
                # xeeg_common = xeeg_sa[np.logical_or(skip_modality==0, skip_modality==1)] #Process EEG with EOG if you are not skipping EOG
                xeeg_common = xeeg[skip_modality==0] #Process EEG with EOG if you are not skipping EOG
                xeeg_common_init = xeeg[skip_modality==0] #Process EEG with EOG if you are not skipping EOG
                # xeeg_undir_sole = xeeg_sa[skip_modality==1] #If you skip EOG process EEG on its own
                xeeg_dir_sole = xeeg[skip_modality==2] #If you skip EOG process EEG on its own
                xeog_common = xeog[skip_modality==0] #Process EOG with EEG if you are not skipping EEG
                xeog_common_init = xeog[skip_modality==0] #Process EOG with EEG if you are not skipping EEG
                # xeog_undir_sole = xeog_sa[skip_modality==2] #If you skip EEG process EOG on its own
                xeog_dir_sole = xeog[skip_modality==1] #If you skip EEG process EOG on its own
            # print(skip_modality)

        if xeeg_common.shape[0]>0 and xeog_common.shape[0]>0:
            xeeg_common = self.inner_tf_eeg.forward_sa(xeeg_common)
            xeog_common = self.inner_tf_eog.forward_sa(xeog_common)

            # xeeg_sa_o = self.outer_positional_embedding(xeeg_sa_o)
            # xeog_sa_o = self.outer_positional_embedding(xeog_sa_o)
            xeeg_common_outer = self.outer_tf_eeg_out.forward_sa(xeeg_common, extract_norm=extract_norm)[:, :, :1]
            xeog_common_outer = self.outer_tf_eog_out.forward_sa(xeog_common, extract_norm=extract_norm)[:, :, :1]


            if return_reconstruction:
                xeeg_latent =self.tanh(self.token_to_latent_eeg(xeeg_common_outer))
                xeog_latent =self.tanh(self.token_to_latent_eog(xeog_common_outer))

                #Deconstruct
                xeeg_mask = self.mask_token_eeg.repeat(xeeg_common.shape[0], xeeg_common.shape[1], xeeg_common.shape[2] - 1, 1, xeeg_common.shape[3], 1)
                xeeg_deconstruct_comb = torch.cat([xeeg_latent, xeeg_mask], dim=2)
                xeeg_deconstruct_comb = self.deconstruct_inner_tf_eeg(xeeg_deconstruct_comb)

                xeog_mask = self.mask_token_eeg.repeat(xeog_common.shape[0], xeog_common.shape[1], xeog_common.shape[2] - 1, 1, xeog_common.shape[3], 1)
                xeog_deconstruct_comb = torch.cat([xeog_latent, xeog_mask], dim=2)
                xeog_deconstruct_comb = self.deconstruct_inner_tf_eeg(xeog_deconstruct_comb)


            xeeg_sa_o, xeog_sa_o = xeeg_common_outer.squeeze(), xeog_common_outer.squeeze()

            xeeg_ca_common = self.inner_tf_eeg.forward_ca(xeeg_common_init, xeog_common)[:, :, :1]
            xeog_ca_common = self.inner_tf_eog.forward_ca(xeog_common_init, xeeg_common)[:, :, :1]

            if self.pos:
                xeeg_ca_common = self.outer_positional_embedding(xeeg_ca_common)
                xeog_ca_common = self.outer_positional_embedding(xeog_ca_common)

            # cls_token_outer_eeg = self.cls_token_outer_eeg.repeat(xeeg_ca_common.shape[0], xeeg_ca_common.shape[1], 1, 1, 1, 1)
            # cls_token_outer_eog = self.cls_token_outer_eog.repeat(xeog_ca_common.shape[0], xeog_ca_common.shape[1], 1, 1, 1, 1)
            #
            # xeeg_ca_common = torch.cat([cls_token_outer_eeg, xeog_ca_common], dim=3)
            # xeog_ca_common = torch.cat([cls_token_outer_eog, xeog_ca_common], dim=3)

            xeeg_ca_common_outer = self.outer_tf_eeg_out.forward_ca(xeeg_ca_common, xeog_common_outer, extract_norm=extract_norm)
            xeog_ca_common_outer = self.outer_tf_eog_out.forward_ca(xeog_ca_common, xeeg_common_outer, extract_norm=extract_norm)

            x_common = xeeg_ca_common_outer + xeog_ca_common_outer

        if "xeeg_dir_sole" in locals() and xeeg_dir_sole.shape[0]>0:
            xeeg_ca_sole = self.inner_tf_eeg.forward_ca(xeeg_dir_sole)[:, :, :1]

            if self.pos:
                xeeg_ca_sole = self.outer_positional_embedding(xeeg_ca_sole)
            # cls_token_outer_eeg = self.cls_token_outer.repeat(xeeg_ca_sole.shape[0], xeeg_ca_sole.shape[1], 1, 1, 1, 1)
            # xeeg_ca_sole = torch.cat([cls_token_outer_eeg, xeeg_ca_sole], dim=3)
            xeeg_ca_sole = self.outer_tf_eeg_out.forward_ca(xeeg_ca_sole, extract_norm=extract_norm)[:, :, :, :1]

            if return_reconstruction:
                xeeg_latent =self.tanh(self.token_to_latent_eeg(xeeg_ca_sole))
                #Deconstruct
                xeeg_mask = self.mask_token_eeg.repeat(xeeg_dir_sole.shape[0], xeeg_dir_sole.shape[1], xeeg_dir_sole.shape[2] - 1, 1, xeeg_dir_sole.shape[3], 1)
                xeeg_deconstruct = torch.cat([xeeg_latent, xeeg_mask], dim=2)
                xeeg_deconstruct = self.deconstruct_inner_tf_eeg(xeeg_deconstruct)

        if "xeog_dir_sole" in locals() and xeog_dir_sole.shape[0]>0:
            xeog_ca_sole = self.inner_tf_eog.forward_ca(xeog_dir_sole)[:, :, :1]
            if self.pos:
                xeog_ca_sole = self.outer_positional_embedding(xeog_ca_sole)
            # cls_token_outer_eog = self.cls_token_outer.repeat(xeog_ca_sole.shape[0], xeog_ca_sole.shape[1], 1, 1, 1, 1)
            # xeog_ca_sole = torch.cat([cls_token_outer_eog, xeog_ca_sole], dim=3)
            xeog_ca_sole = self.outer_tf_eog_out.forward_ca(xeog_ca_sole, extract_norm=extract_norm)[:, :, :, :1]
            if return_reconstruction:
                xeog_latent =self.tanh(self.token_to_latent_eeg(xeog_ca_sole))
                #Deconstruct
                xeog_mask = self.mask_token_eeg.repeat(xeog_dir_sole.shape[0], xeog_dir_sole.shape[1], xeog_dir_sole.shape[2] - 1, 1, xeog_dir_sole.shape[3], 1)
                xeog_deconstruct = torch.cat([xeog_latent, xeog_mask], dim=2)
                xeog_deconstruct = self.deconstruct_inner_tf_eeg(xeog_deconstruct)

        if "xeog_dir_sole" in locals() and "xeog_dir_sole" in locals() and (xeog_dir_sole.shape[0]>0 or xeeg_dir_sole.shape[0]>0):
            x = []
            xeeg_deconstruct_total = []
            xeog_deconstruct_total = []
            xeeg_init_total = []
            xeog_init_total = []
            counter = [0, 0, 0]
            for i in range(len(skip_modality)):
                if skip_modality[i] == 0:
                    x.append(x_common[counter[int(skip_modality[i].item())]:counter[int(skip_modality[i].item())] + 1])
                    if return_reconstruction:

                        xeeg_deconstruct_total.append(xeeg_deconstruct_comb[counter[int(skip_modality[i].item())]:counter[int(skip_modality[i].item())] + 1])
                        xeeg_init_total.append(xeeg_common_init[counter[int(skip_modality[i].item())]:counter[int(skip_modality[i].item())] + 1])
                        xeog_deconstruct_total.append(xeog_deconstruct_comb[counter[int(skip_modality[i].item())]:counter[int(skip_modality[i].item())] + 1])
                        xeog_init_total.append(xeog_common_init[counter[int(skip_modality[i].item())]:counter[int(skip_modality[i].item())] + 1])

                elif skip_modality[i] == 1:
                    x.append(xeog_ca_sole[counter[int(skip_modality[i].item())]:counter[int(skip_modality[i].item())] + 1])
                    if return_reconstruction:
                        xeog_deconstruct_total.append(xeog_deconstruct[counter[int(skip_modality[i].item())]:counter[int(skip_modality[i].item())] + 1])
                        xeog_init_total.append(xeog_dir_sole[counter[int(skip_modality[i].item())]:counter[int(skip_modality[i].item())] + 1])
                elif skip_modality[i] == 2:
                    x.append(xeeg_ca_sole[counter[int(skip_modality[i].item())]:counter[int(skip_modality[i].item())] + 1])
                    if return_reconstruction:
                        xeeg_deconstruct_total.append(xeeg_deconstruct[counter[int(skip_modality[i].item())]:counter[int(skip_modality[i].item())] + 1])
                        xeeg_init_total.append(xeeg_dir_sole[counter[int(skip_modality[i].item())]:counter[int(skip_modality[i].item())] + 1])

                counter[int(skip_modality[i].item())] += 1
            x = torch.cat(x, dim=0)
            if return_reconstruction:

                xeeg_deconstruct_total = torch.cat(xeeg_deconstruct_total, dim=0) if len(xeeg_deconstruct_total)!=0 else torch.empty([])
                xeog_deconstruct_total = torch.cat(xeog_deconstruct_total, dim=0) if len(xeog_deconstruct_total)!=0 else torch.empty([])
                xeeg_init_total = torch.cat(xeeg_init_total, dim=0) if len(xeeg_init_total)!=0 else torch.empty([])
                xeog_init_total = torch.cat(xeog_init_total, dim=0) if len(xeog_init_total)!=0 else torch.empty([])
        else:
            x = x_common
            if return_reconstruction:
                xeeg_deconstruct_total = xeeg_deconstruct_comb
                xeog_deconstruct_total = xeog_deconstruct_comb
                xeeg_init_total = xeeg_common_init
                xeog_init_total = xeog_common_init

        output={"output_features": x}

        if return_matches:
            if ("xeeg_sa_o" in locals() and xeeg_sa_o.shape[0]>0 and len(xeeg_sa_o.shape)==3) and ("xeog_sa_o" in locals() and xeog_sa_o.shape[0]>0 and len(xeog_sa_o.shape)==3):
                x_match = torch.einsum('b o f , b m f -> b o m', xeeg_sa_o, xeog_sa_o)
                output["matches"] = x_match
            else:
                output["matches"] = None
        if return_inter_reps:
            output["intermediate_reps"] = [xeeg_sa_o, xeeg_sa_o]
        if return_order:
            x_sa_o = torch.cat([xeeg_sa_o, xeog_sa_o], dim=3)
            output["order"] =x_sa_o
        if return_reconstruction:
            output["reconstruction"] = {"eeg":xeeg_deconstruct_total, "eog":xeog_deconstruct_total}
            output["input"] = {"eeg": xeeg_init_total, "eog": xeog_init_total}

        return output
class SleepEnc_BLIP_EEG_EOG_twomode_caouter_shared_fullca(nn.Module):
    def __init__(self, args, encs=[None]):
        """
        :param encoder_filters_small, encoder_filters_big: filters of CNN output, quantized by 20
        :param encs_small, encs_big:
        """
        super().__init__()
        self.args = args

        self.enc_0 = encs[0]
        self.enc_1 = encs[0]

        d_model = args.dmodel  # 64*8

        inner_biased = args.inner_biased if "inner_biased" in args else False
        outer_biased = args.outer_biased if "outer_biased" in args else False
        rpos = args.rpos if "rpos" in args else False
        self.pos = args.pos if "pos" in args else True
        dim_proj = args.dim_proj if "dim_proj" in args else 128
        self.outer_rep = args.outer_rep if "outer_rep" in args else False
        self.skip_random_mod = args.skip_random_mod if "skip_random_mod" in args else False

        if inner_biased == "gaussian_learned":
            inner_biased = Gaussian_Learned_Attention_Bias(d_model, heads=8, type="mul")

        # if outer_biased == "gaussian_learned":
        #     outer_biased = Gaussian_Learned_Attention_Bias(d_model, heads=8, type="mul")

        self.inner_tf_eeg = inner_ch_SA_CA_shared_v2_fullca(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, dim_proj=dim_proj, gbiased=inner_biased, num_layers=4)
        self.inner_tf_eog = inner_ch_SA_CA_shared_v2_fullca(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, dim_proj=dim_proj, gbiased=inner_biased, num_layers=4)

        self.outer_tf_eeg_out = outer_mod_ch_SA_CA_shared_v2_fullca(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, dim_proj=dim_proj, gbiased=outer_biased, num_layers=4)
        self.outer_tf_eog_out = outer_mod_ch_SA_CA_shared_v2_fullca(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, dim_proj=dim_proj, gbiased=outer_biased, num_layers=4)

        self.cls_token_eeg = nn.Parameter(torch.randn(1, 1, 1, 1, 1, d_model))
        self.cls_token_eog = nn.Parameter(torch.randn(1, 1, 1, 1, 1, d_model))
        self.cls_token_outer_eeg = nn.Parameter(torch.randn(1, 1, 1, 1, 1, d_model))
        self.cls_token_outer_eog = nn.Parameter(torch.randn(1, 1, 1, 1, 1, d_model))

        # if self.skip_random_mod:
        #     self.skip_dropout = nn.Dropout()
        if self.pos:
            self.inner_positional_embedding = huy_pos_inner(d_model, pos=False, inner=29, outer=21, modalities=1,
                                                              channels=1)
            self.outer_positional_embedding = huy_pos_outer(d_model, pos=False, inner=29, outer=21, modalities=1,
                                                            channels=1)

    def forward(self, x, return_matches=False, extract_norm=False, return_inter_reps=False, return_order=False, skip_modality="random"):

        xeeg = x["stft_eeg"][:, :, :, :, 1:, :]  # mat
        xeog = x["stft_eog"][:, :, :, :, 1:, :]  # mat

        xeeg = einops.rearrange(xeeg, "b outer mod ch f inner -> b outer inner mod ch f")
        xeog = einops.rearrange(xeog, "b outer mod ch f inner -> b outer inner mod ch f")

        if self.pos:
            xeeg = self.inner_positional_embedding(xeeg)
            xeog = self.inner_positional_embedding(xeog)

        cls_token_eeg = self.cls_token_eeg.repeat(xeeg.shape[0], xeeg.shape[1], 1, 1, xeeg.shape[3], 1)
        xeeg = torch.cat([cls_token_eeg, xeeg], dim=2)

        cls_token_eog = self.cls_token_eog.repeat(xeog.shape[0], xeog.shape[1], 1, 1, xeog.shape[3], 1)
        xeog = torch.cat([cls_token_eog, xeog], dim=2)

        xeeg_common_init = xeeg
        xeog_common_init = xeog

        xeeg_common = xeeg
        xeog_common = xeog

        if self.skip_random_mod:
            if  self.training:
                # skip_modality_eeg = torch.bernoulli(torch.ones(xeeg.shape[0])*0.5)
                # skip_modality_eog = torch.bernoulli(torch.ones(xeog.shape[0])*0.5)
                # skip_modality = skip_modality_eeg*2 + skip_modality_eog
                # skip_modality[skip_modality==3] = 0

                skip_modality = torch.rand(xeeg.shape[0])
                # skip_modality[skip_modality>0.5] = 2
                # skip_modality[skip_modality<=0.5] = 1
                skip_modality[skip_modality > 0.66] = 2
                skip_modality[skip_modality < 0.33] = 1
                skip_modality = skip_modality.int()
                # skip_modality[skip_modality < 1.5 ]=0
                # skip_modality = None

            else:
                if skip_modality=="full":
                    skip_modality = None
                elif skip_modality=="random":
                    skip_modality = torch.rand(xeeg.shape[0])
                    # skip_modality[skip_modality > 0.5] = 2
                    # skip_modality[skip_modality <= 0.5] = 1
                    skip_modality[skip_modality > 0.66] = 2
                    skip_modality[skip_modality < 0.33] = 1
                    skip_modality = skip_modality.int()
                    # skip_modality[skip_modality == 0] = 1

                elif skip_modality=="eeg":
                    skip_modality = torch.ones(xeeg.shape[0])*1
                elif skip_modality=="eog":
                    skip_modality = torch.ones(xeeg.shape[0])*2
                elif skip_modality is "vae":
                    with torch.no_grad():
                        vae_eeg_output = self.enc_0([x[0]])
                        output_losses = self.enc_0.module.loss_function(vae_eeg_output[0], vae_eeg_output[1],
                                                                        vae_eeg_output[2], vae_eeg_output[3],
                                                                        reduction="none")
                        eeg_routing = output_losses["total"].flatten(start_dim=1).mean(dim=1)
                        # print(eeg_routing)
                        eeg_routing[eeg_routing < 3] = 0
                        eeg_routing[eeg_routing > 3] = 1

                        vae_eog_output = self.enc_1([x[1]])
                        output_losses = self.enc_1.module.loss_function(vae_eog_output[0], vae_eog_output[1],
                                                                        vae_eog_output[2], vae_eog_output[3],
                                                                        reduction="none")
                        eog_routing = output_losses["total"].flatten(start_dim=1).mean(dim=1)
                        # print(eog_routing)
                        eog_routing[eog_routing < 3] = 0
                        eog_routing[eog_routing > 3] = 2

                        skip_modality = eeg_routing + eog_routing
                        skip_modality[skip_modality == 3] = 0

            if skip_modality is not None:
                # xeeg_common = xeeg_sa[np.logical_or(skip_modality==0, skip_modality==1)] #Process EEG with EOG if you are not skipping EOG
                xeeg_common = xeeg[skip_modality==0] #Process EEG with EOG if you are not skipping EOG
                xeeg_common_init = xeeg[skip_modality==0] #Process EEG with EOG if you are not skipping EOG
                # xeeg_undir_sole = xeeg_sa[skip_modality==1] #If you skip EOG process EEG on its own
                xeeg_dir_sole = xeeg[skip_modality==2] #If you skip EOG process EEG on its own
                xeog_common = xeog[skip_modality==0] #Process EOG with EEG if you are not skipping EEG
                xeog_common_init = xeog[skip_modality==0] #Process EOG with EEG if you are not skipping EEG
                # xeog_undir_sole = xeog_sa[skip_modality==2] #If you skip EEG process EOG on its own
                xeog_dir_sole = xeog[skip_modality==1] #If you skip EEG process EOG on its own
            # print(skip_modality)

        if xeeg_common.shape[0]>0 and xeog_common.shape[0]>0:

            xeeg_common = self.inner_tf_eeg.forward_sa(xeeg_common, return_layer = "all")
            xeog_common = self.inner_tf_eog.forward_sa(xeog_common, return_layer = "all")

            xeeg_common_last_layer = xeeg_common[-1, :, :, :1]#last layer cls!
            xeog_common_last_layer = xeog_common[-1, :, :, :1]

            xeeg_common_outer = self.outer_tf_eeg_out.forward_sa(xeeg_common_last_layer, extract_norm=extract_norm, return_layer = "all")
            xeog_common_outer = self.outer_tf_eog_out.forward_sa(xeog_common_last_layer, extract_norm=extract_norm, return_layer = "all")

            xeeg_sa_o, xeog_sa_o = xeeg_common_outer[-1].squeeze(), xeog_common_outer[-1].squeeze()

            xeeg_ca_common = self.inner_tf_eeg.forward_ca(xeeg_common_init, xeog_common)[:, :, :1]
            xeog_ca_common = self.inner_tf_eog.forward_ca(xeog_common_init, xeeg_common)[:, :, :1]

            if self.pos:
                xeeg_ca_common = self.outer_positional_embedding(xeeg_ca_common)
                xeog_ca_common = self.outer_positional_embedding(xeog_ca_common)

            # cls_token_outer_eeg = self.cls_token_outer_eeg.repeat(xeeg_ca_common.shape[0], xeeg_ca_common.shape[1], 1, 1, 1, 1)
            # cls_token_outer_eog = self.cls_token_outer_eog.repeat(xeog_ca_common.shape[0], xeog_ca_common.shape[1], 1, 1, 1, 1)
            #
            # xeeg_ca_common = torch.cat([cls_token_outer_eeg, xeog_ca_common], dim=3)
            # xeog_ca_common = torch.cat([cls_token_outer_eog, xeog_ca_common], dim=3)

            xeeg_ca_common_outer = self.outer_tf_eeg_out.forward_ca(xeeg_ca_common, xeog_common_outer, extract_norm=extract_norm)
            xeog_ca_common_outer = self.outer_tf_eog_out.forward_ca(xeog_ca_common, xeeg_common_outer, extract_norm=extract_norm)

            x_common = xeeg_ca_common_outer + xeog_ca_common_outer

        if "xeeg_dir_sole" in locals() and xeeg_dir_sole.shape[0]>0:
            xeeg_ca_sole = self.inner_tf_eeg.forward_ca(xeeg_dir_sole)[:, :, :1]
            if self.pos:
                xeeg_ca_sole = self.outer_positional_embedding(xeeg_ca_sole)
            # cls_token_outer_eeg = self.cls_token_outer.repeat(xeeg_ca_sole.shape[0], xeeg_ca_sole.shape[1], 1, 1, 1, 1)
            # xeeg_ca_sole = torch.cat([cls_token_outer_eeg, xeeg_ca_sole], dim=3)
            xeeg_ca_sole = self.outer_tf_eeg_out.forward_ca(xeeg_ca_sole, extract_norm=extract_norm)[:, :, :, :1]

        if "xeog_dir_sole" in locals() and xeog_dir_sole.shape[0]>0:
            xeog_ca_sole = self.inner_tf_eog.forward_ca(xeog_dir_sole)[:, :, :1]
            if self.pos:
                xeog_ca_sole = self.outer_positional_embedding(xeog_ca_sole)
            # cls_token_outer_eog = self.cls_token_outer.repeat(xeog_ca_sole.shape[0], xeog_ca_sole.shape[1], 1, 1, 1, 1)
            # xeog_ca_sole = torch.cat([cls_token_outer_eog, xeog_ca_sole], dim=3)
            xeog_ca_sole = self.outer_tf_eog_out.forward_ca(xeog_ca_sole, extract_norm=extract_norm)[:, :, :, :1]

        if "xeog_dir_sole" in locals() and "xeog_dir_sole" in locals() and (xeog_dir_sole.shape[0]>0 or xeeg_dir_sole.shape[0]>0):
            x = []
            counter = [0, 0, 0]
            for i in range(len(skip_modality)):
                if skip_modality[i] == 0:
                    x.append(x_common[counter[int(skip_modality[i].item())]:counter[int(skip_modality[i].item())] + 1])
                elif skip_modality[i] == 1:
                    x.append(xeog_ca_sole[
                             counter[int(skip_modality[i].item())]:counter[int(skip_modality[i].item())] + 1])
                elif skip_modality[i] == 2:
                    x.append(xeeg_ca_sole[
                             counter[int(skip_modality[i].item())]:counter[int(skip_modality[i].item())] + 1])
                counter[int(skip_modality[i].item())] += 1
            x = torch.cat(x, dim=0)
        else:
            x = x_common

        output={"output_features": x}

        if return_matches:
            if ("xeeg_sa_o" in locals() and xeeg_sa_o.shape[0]>0 and len(xeeg_sa_o.shape)==3) and ("xeog_sa_o" in locals() and xeog_sa_o.shape[0]>0 and len(xeog_sa_o.shape)==3):
                x_match = torch.einsum('b o f , b m f -> b o m', xeeg_sa_o, xeog_sa_o)
                output["matches"] = x_match
            else:
                output["matches"] = None
        if return_inter_reps:
            output["intermediate_reps"] = [xeeg_sa_o, xeeg_sa_o]
        if return_order:
            x_sa_o = torch.cat([xeeg_sa_o, xeog_sa_o], dim=3)
            output["order"] =x_sa_o

        return output
class SleepEnc_BLIP_EEG_EOG_twomode_caouter_shared_limited(nn.Module):
    def __init__(self, args, encs=[None]):
        """
        :param encoder_filters_small, encoder_filters_big: filters of CNN output, quantized by 20
        :param encs_small, encs_big:
        """
        super().__init__()
        self.args = args

        self.enc_0 = encs[0]
        self.enc_1 = encs[0]

        d_model = args.dmodel  # 64*8

        inner_biased = args.inner_biased if "inner_biased" in args else False
        outer_biased = args.outer_biased if "outer_biased" in args else False
        rpos = args.rpos if "rpos" in args else False
        self.pos = args.pos if "pos" in args else True
        dim_proj = args.dim_proj if "dim_proj" in args else 128
        self.outer_rep = args.outer_rep if "outer_rep" in args else False
        self.skip_random_mod = args.skip_random_mod if "skip_random_mod" in args else False

        if inner_biased == "gaussian_learned":
            inner_biased = Gaussian_Learned_Attention_Bias(d_model, heads=8, type="mul")

        # if outer_biased == "gaussian_learned":
        #     outer_biased = Gaussian_Learned_Attention_Bias(d_model, heads=8, type="mul")

        self.inner_tf_eeg_01 = inner_ch_SA_CA_shared_v2(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, dim_proj=dim_proj, gbiased=inner_biased, num_layers=2)
        self.inner_tf_eeg_23 = inner_ch_SA_CA_shared_v2(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, dim_proj=dim_proj, gbiased=inner_biased, num_layers=2)
        self.inner_tf_eog_01 = inner_ch_SA_CA_shared_v2(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, dim_proj=dim_proj, gbiased=inner_biased, num_layers=2)
        self.inner_tf_eog_23 = inner_ch_SA_CA_shared_v2(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, dim_proj=dim_proj, gbiased=inner_biased, num_layers=2)

        self.outer_tf_eeg_out_01 = outer_mod_ch_SA_CA_shared_v2(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, dim_proj=dim_proj, gbiased=outer_biased, num_layers=2)
        self.outer_tf_eeg_out_23 = outer_mod_ch_SA_CA_shared_v2(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, dim_proj=dim_proj, gbiased=outer_biased, num_layers=2)
        self.outer_tf_eog_out_01 = outer_mod_ch_SA_CA_shared_v2(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, dim_proj=dim_proj, gbiased=outer_biased, num_layers=2)
        self.outer_tf_eog_out_23 = outer_mod_ch_SA_CA_shared_v2(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, dim_proj=dim_proj, gbiased=outer_biased, num_layers=2)

        self.cls_token_eeg = nn.Parameter(torch.randn(1, 1, 1, 1, 1, d_model))
        self.cls_token_eog = nn.Parameter(torch.randn(1, 1, 1, 1, 1, d_model))
        self.cls_token_outer_eeg = nn.Parameter(torch.randn(1, 1, 1, 1, 1, d_model))
        self.cls_token_outer_eog = nn.Parameter(torch.randn(1, 1, 1, 1, 1, d_model))

        # if self.skip_random_mod:
        #     self.skip_dropout = nn.Dropout()
        if self.pos:
            self.inner_positional_embedding = huy_pos_inner(d_model, pos=False, inner=29, outer=21, modalities=1,
                                                              channels=1)
            self.outer_positional_embedding = huy_pos_outer(d_model, pos=False, inner=29, outer=21, modalities=1,
                                                            channels=1)

    def forward(self, x, return_matches=False, extract_norm=False, return_inter_reps=False, return_order=False, skip_modality="random", **kwargs):

        xeeg = x["stft_eeg"][:, :, :, :, 1:, :]  # mat
        xeog = x["stft_eog"][:, :, :, :, 1:, :]  # mat

        xeeg = einops.rearrange(xeeg, "b outer mod ch f inner -> b outer inner mod ch f")
        xeog = einops.rearrange(xeog, "b outer mod ch f inner -> b outer inner mod ch f")

        if self.pos:
            xeeg = self.inner_positional_embedding(xeeg)
            xeog = self.inner_positional_embedding(xeog)

        cls_token_eeg = self.cls_token_eeg.repeat(xeeg.shape[0], xeeg.shape[1], 1, 1, xeeg.shape[3], 1)
        xeeg = torch.cat([cls_token_eeg, xeeg], dim=2)

        cls_token_eog = self.cls_token_eog.repeat(xeog.shape[0], xeog.shape[1], 1, 1, xeog.shape[3], 1)
        xeog = torch.cat([cls_token_eog, xeog], dim=2)

        xeeg_common_init = xeeg
        xeog_common_init = xeog

        xeeg_common = xeeg
        xeog_common = xeog

        # if self.skip_random_mod:
        if  self.training:
            skip_modality = torch.rand(xeeg.shape[0])
            skip_modality[skip_modality > 0.66] = 2
            skip_modality[skip_modality < 0.33] = 1
            skip_modality = skip_modality.int().unsqueeze(dim=1).repeat(1, xeeg.shape[1])
        else:
            if skip_modality=="full":
                skip_modality = torch.ones(xeeg.shape[0], xeeg.shape[1])*0
            elif skip_modality=="random":
                skip_modality = torch.rand(xeeg.shape[0])
                skip_modality[skip_modality > 0.66] = 2
                skip_modality[skip_modality < 0.33] = 1
                skip_modality = skip_modality.int().unsqueeze(dim=1).repeat(1, xeeg.shape[1])
            elif skip_modality=="eeg":
                skip_modality = torch.ones(xeeg.shape[0], xeeg.shape[1])*1
            elif skip_modality=="eog":
                skip_modality = torch.ones(xeeg.shape[0], xeeg.shape[1])*2
            elif skip_modality is "vae":
                with torch.no_grad():
                    vae_eeg_output = self.enc_0(x)
                    output_losses = self.enc_0.module.loss_function(vae_eeg_output[0], vae_eeg_output[1],
                                                                    vae_eeg_output[2], vae_eeg_output[3],
                                                                    reduction="none")
                    eeg_routing = output_losses["total"].flatten(start_dim=1).mean(dim=1)
                    # print(eeg_routing)
                    eeg_routing[eeg_routing < 3] = 0
                    eeg_routing[eeg_routing > 3] = 1

                    vae_eog_output = self.enc_1(x)
                    output_losses = self.enc_1.module.loss_function(vae_eog_output[0], vae_eog_output[1],
                                                                    vae_eog_output[2], vae_eog_output[3],
                                                                    reduction="none")
                    eog_routing = output_losses["total"].flatten(start_dim=1).mean(dim=1)
                    # print(eog_routing)
                    eog_routing[eog_routing < 3] = 0
                    eog_routing[eog_routing > 3] = 2

                    skip_modality = eeg_routing + eog_routing
                    skip_modality[skip_modality == 3] = 0
                    skip_modality = skip_modality.unsqueeze(dim=1).repeat(1,21)
            else:
                skip_modality_temp = torch.zeros(xeeg.shape[0],xeeg.shape[1])
                skip_modality_temp[skip_modality["stft_eeg"]==1]=1
                skip_modality_temp[skip_modality["stft_eog"]==1]=2
                skip_modality = skip_modality_temp

        if skip_modality is not None:
            # xeeg_common = xeeg_sa[np.logical_or(skip_modality==0, skip_modality==1)] #Process EEG with EOG if you are not skipping EOG
            if (skip_modality == 0).sum()>0:

                xeeg_common = xeeg[skip_modality == 0]
                xeog_common = xeog[skip_modality == 0]
                if (skip_modality == 0).shape[0]!=1:
                    xeeg_common = einops.rearrange(xeeg_common.unfold(0, xeeg.shape[1], xeeg.shape[1]),"b t m c f i -> b i t m c f")
                    xeog_common = einops.rearrange(xeog_common.unfold(0, xeog.shape[1], xeog.shape[1]),"b t m c f i -> b i t m c f")
                else:
                    xeeg_common = xeeg_common.unsqueeze(dim=0)
                    xeog_common = xeog_common.unsqueeze(dim=0)
            else:
                xeeg_common = None
                xeog_common = None

            # xeeg_common_init = copy.deepcopy(xeeg_common) #Process EEG with EOG if you are not skipping EOG
            # xeog_common_init = copy.deepcopy(xeog_common) #Process EOG with EEG if you are not skipping EEG

            xeeg_common_init = xeeg_common #Process EEG with EOG if you are not skipping EOG
            xeog_common_init = xeog_common #Process EOG with EEG if you are not skipping EEG

            skip_eog_len = (skip_modality == 2).sum()
            skip_eeg_len = (skip_modality == 1).sum()
            if skip_eog_len>0:
                xeeg_dir_sole = xeeg[skip_modality == 2]
                if xeeg.shape[0]!=1:
                    # print(xeeg.shape)
                    # print(xeeg_dir_sole.shape)
                    xeeg_dir_sole = einops.rearrange(xeeg_dir_sole.unfold(0, xeeg.shape[1], xeeg.shape[1]),
                                                   "b t m c f i -> b i t m c f")
                else:
                    xeeg_dir_sole = xeeg_dir_sole.unsqueeze(dim=0)
            if skip_eeg_len>0:
                xeog_dir_sole = xeog[skip_modality == 1]
                if xeog.shape[0]!=1:
                    xeog_dir_sole = einops.rearrange(xeog_dir_sole.unfold(0, xeog.shape[1], xeog.shape[1]),
                                                   "b t m c f i -> b i t m c f")
                else:
                    xeog_dir_sole = xeog_dir_sole.unsqueeze(dim=0)
        a, b = np.unique(skip_modality.cpu().numpy(), return_counts=True)
        x_output = torch.ones(xeeg.shape)[:,:,:1].to(xeeg.device)
        if xeeg_common is not None and xeog_common is not None and xeeg_common.shape[0]>0 and xeog_common.shape[0]>0:
            xeeg_common = self.inner_tf_eeg_01.forward_sa(xeeg_common)
            xeeg_common = self.inner_tf_eeg_23.forward_sa(xeeg_common)
            xeog_common = self.inner_tf_eog_01.forward_sa(xeog_common)
            xeog_common = self.inner_tf_eog_23.forward_sa(xeog_common)

            # xeeg_sa_o = self.outer_positional_embedding(xeeg_sa_o)
            # xeog_sa_o = self.outer_positional_embedding(xeog_sa_o)
            xeeg_common_outer = self.outer_tf_eeg_out_01.forward_sa(xeeg_common, extract_norm=extract_norm)
            xeeg_common_outer = self.outer_tf_eeg_out_23.forward_sa(xeeg_common_outer, extract_norm=extract_norm)[:, :, :1]
            xeog_common_outer = self.outer_tf_eog_out_01.forward_sa(xeog_common, extract_norm=extract_norm)
            xeog_common_outer = self.outer_tf_eog_out_23.forward_sa(xeog_common_outer, extract_norm=extract_norm)[:, :, :1]

            xeeg_sa_o, xeog_sa_o = xeeg_common_outer.squeeze(), xeog_common_outer.squeeze()

            xeeg_ca_common = self.inner_tf_eeg_01.forward_sa(xeeg_common_init)
            xeeg_ca_common = self.inner_tf_eeg_23.forward_ca(xeeg_ca_common, xeog_common)[:, :, :1]
            xeog_ca_common = self.inner_tf_eog_01.forward_sa(xeog_common_init)
            xeog_ca_common = self.inner_tf_eog_23.forward_ca(xeog_ca_common, xeeg_common)[:, :, :1]

            if self.pos:
                xeeg_ca_common = self.outer_positional_embedding(xeeg_ca_common)
                xeog_ca_common = self.outer_positional_embedding(xeog_ca_common)

            # cls_token_outer_eeg = self.cls_token_outer_eeg.repeat(xeeg_ca_common.shape[0], xeeg_ca_common.shape[1], 1, 1, 1, 1)
            # cls_token_outer_eog = self.cls_token_outer_eog.repeat(xeog_ca_common.shape[0], xeog_ca_common.shape[1], 1, 1, 1, 1)
            #
            # xeeg_ca_common = torch.cat([cls_token_outer_eeg, xeog_ca_common], dim=3)
            # xeog_ca_common = torch.cat([cls_token_outer_eog, xeog_ca_common], dim=3)

            xeeg_ca_common_outer = self.outer_tf_eeg_out_01.forward_sa(xeeg_ca_common, extract_norm=extract_norm)
            xeeg_ca_common_outer = self.outer_tf_eeg_out_23.forward_ca(xeeg_ca_common_outer, xeog_common_outer, extract_norm=extract_norm)
            xeog_ca_common_outer = self.outer_tf_eog_out_01.forward_sa(xeog_ca_common, extract_norm=extract_norm)
            xeog_ca_common_outer = self.outer_tf_eog_out_23.forward_ca(xeog_ca_common_outer, xeeg_common_outer, extract_norm=extract_norm)

            x_common = xeeg_ca_common_outer + xeog_ca_common_outer
            x_output[skip_modality == 0] = x_common.flatten(start_dim=0, end_dim=1)
        if "xeeg_dir_sole" in locals() and xeeg_dir_sole.shape[0]>0:
            xeeg_ca_sole = self.inner_tf_eeg_01.forward_ca(xeeg_dir_sole)
            xeeg_ca_sole = self.inner_tf_eeg_23.forward_ca(xeeg_ca_sole)[:, :, :1]
            if self.pos:
                xeeg_ca_sole = self.outer_positional_embedding(xeeg_ca_sole)
            # cls_token_outer_eeg = self.cls_token_outer.repeat(xeeg_ca_sole.shape[0], xeeg_ca_sole.shape[1], 1, 1, 1, 1)
            # xeeg_ca_sole = torch.cat([cls_token_outer_eeg, xeeg_ca_sole], dim=3)
            xeeg_ca_sole = self.outer_tf_eeg_out_01.forward_ca(xeeg_ca_sole, extract_norm=extract_norm)
            xeeg_ca_sole = self.outer_tf_eeg_out_23.forward_ca(xeeg_ca_sole, extract_norm=extract_norm)[:, :, :, :1]
            x_output[skip_modality == 2] = xeeg_ca_sole.flatten(start_dim=0, end_dim=1)
        if "xeog_dir_sole" in locals() and xeog_dir_sole.shape[0]>0:
            xeog_ca_sole = self.inner_tf_eog_01.forward_ca(xeog_dir_sole)
            xeog_ca_sole = self.inner_tf_eog_23.forward_ca(xeog_ca_sole)[:, :, :1]
            if self.pos:
                xeog_ca_sole = self.outer_positional_embedding(xeog_ca_sole)
            # cls_token_outer_eog = self.cls_token_outer.repeat(xeog_ca_sole.shape[0], xeog_ca_sole.shape[1], 1, 1, 1, 1)
            # xeog_ca_sole = torch.cat([cls_token_outer_eog, xeog_ca_sole], dim=3)
            xeog_ca_sole = self.outer_tf_eog_out_01.forward_ca(xeog_ca_sole, extract_norm=extract_norm)
            xeog_ca_sole = self.outer_tf_eog_out_23.forward_ca(xeog_ca_sole, extract_norm=extract_norm)[:, :, :, :1]
            x_output[skip_modality == 1] = xeog_ca_sole.flatten(start_dim=0, end_dim=1)

        # x_output = einops.rearrange(x_output.unfold(0, xeeg.shape[1], xeeg.shape[1]),
        #                  "b t m c f o -> b o t m c f")
        output={"output_features": x_output}

        if return_matches:
            if ("xeeg_sa_o" in locals() and xeeg_sa_o.shape[0]>0 and len(xeeg_sa_o.shape)==3) and ("xeog_sa_o" in locals() and xeog_sa_o.shape[0]>0 and len(xeog_sa_o.shape)==3):
                x_match = torch.einsum('b o f , b m f -> b o m', xeeg_sa_o, xeog_sa_o)
                output["matches"] = x_match
            else:
                output["matches"] = None
        if return_inter_reps:
            output["intermediate_reps"] = [xeeg_sa_o, xeeg_sa_o]
        if return_order:
            x_sa_o = torch.cat([xeeg_sa_o, xeog_sa_o], dim=3)
            output["order"] =x_sa_o

        return output
class SleepEnc_BLIP_EEG_EOG_twomode_caouter_shared_asbefore(nn.Module):
    def __init__(self, args, encs=[None]):
        """
        :param encoder_filters_small, encoder_filters_big: filters of CNN output, quantized by 20
        :param encs_small, encs_big:
        """
        super().__init__()
        self.args = args

        self.enc_0 = encs[0]
        self.enc_1 = encs[0]

        d_model = args.dmodel  # 64*8

        inner_biased = args.inner_biased if "inner_biased" in args else False
        outer_biased = args.outer_biased if "outer_biased" in args else False
        rpos = args.rpos if "rpos" in args else False
        self.pos = args.pos if "pos" in args else True
        dim_proj = args.dim_proj if "dim_proj" in args else 128
        self.outer_rep = args.outer_rep if "outer_rep" in args else False
        self.skip_random_mod = args.skip_random_mod if "skip_random_mod" in args else False

        if inner_biased == "gaussian_learned":
            inner_biased = Gaussian_Learned_Attention_Bias(d_model, heads=8, type="mul")

        # if outer_biased == "gaussian_learned":
        #     outer_biased = Gaussian_Learned_Attention_Bias(d_model, heads=8, type="mul")

        self.inner_tf_eeg = inner_ch_SA_CA_shared_v2(d_model, pos=False, rpos=rpos, inner=29, outer=21,
                                                     modalities=1, dim_proj=dim_proj, gbiased=inner_biased,
                                                     num_layers=4)
        self.inner_tf_eog = inner_ch_SA_CA_shared_v2(d_model, pos=False, rpos=rpos, inner=29, outer=21,
                                                     modalities=1, dim_proj=dim_proj, gbiased=inner_biased,
                                                     num_layers=4)

        self.outer_tf_eeg_out = outer_mod_ch_SA_CA_shared_v2(d_model, pos=False, rpos=rpos, inner=29, outer=21,
                                                             modalities=1, dim_proj=dim_proj, gbiased=outer_biased,
                                                             num_layers=4)
        self.outer_tf_eog_out = outer_mod_ch_SA_CA_shared_v2(d_model, pos=False, rpos=rpos, inner=29, outer=21,
                                                             modalities=1, dim_proj=dim_proj, gbiased=outer_biased,
                                                             num_layers=4)

        self.cls_token_eeg = nn.Parameter(torch.randn(1, 1, 1, 1, 1, d_model))
        self.cls_token_eog = nn.Parameter(torch.randn(1, 1, 1, 1, 1, d_model))
        self.cls_token_outer_eeg = nn.Parameter(torch.randn(1, 1, 1, 1, 1, d_model))
        self.cls_token_outer_eog = nn.Parameter(torch.randn(1, 1, 1, 1, 1, d_model))

        # if self.skip_random_mod:
        #     self.skip_dropout = nn.Dropout()
        if self.pos:
            self.inner_positional_embedding = huy_pos_inner(d_model, pos=False, inner=29, outer=21, modalities=1,
                                                            channels=1)
            self.outer_positional_embedding = huy_pos_outer(d_model, pos=False, inner=29, outer=21, modalities=1,
                                                            channels=1)

    def forward(self, x, return_matches=False, extract_norm=False, return_inter_reps=False, return_order=False,
                return_reconstruction=False, skip_modality="random", **kwargs):

        xeeg = x["stft_eeg"][:, :, :, :, 1:, :]  # mat
        xeog = x["stft_eog"][:, :, :, :, 1:, :]  # mat

        xeeg = einops.rearrange(xeeg, "b outer mod ch f inner -> b outer inner mod ch f")
        xeog = einops.rearrange(xeog, "b outer mod ch f inner -> b outer inner mod ch f")

        if self.pos:
            xeeg = self.inner_positional_embedding(xeeg)
            xeog = self.inner_positional_embedding(xeog)

        cls_token_eeg = self.cls_token_eeg.repeat(xeeg.shape[0], xeeg.shape[1], 1, 1, xeeg.shape[3], 1)
        xeeg = torch.cat([cls_token_eeg, xeeg], dim=2)

        cls_token_eog = self.cls_token_eog.repeat(xeog.shape[0], xeog.shape[1], 1, 1, xeog.shape[3], 1)
        xeog = torch.cat([cls_token_eog, xeog], dim=2)

        xeeg_common_init = xeeg
        xeog_common_init = xeog

        xeeg_common = xeeg
        xeog_common = xeog

        if self.skip_random_mod:
            if self.training:
                # skip_modality_eeg = torch.bernoulli(torch.ones(xeeg.shape[0])*0.5)
                # skip_modality_eog = torch.bernoulli(torch.ones(xeog.shape[0])*0.5)
                # skip_modality = skip_modality_eeg*2 + skip_modality_eog
                # skip_modality[skip_modality==3] = 0

                skip_modality = torch.rand(xeeg.shape[0])
                # skip_modality[skip_modality>0.5] = 2
                # skip_modality[skip_modality<=0.5] = 1
                skip_modality[skip_modality > 0.66] = 2
                skip_modality[skip_modality < 0.33] = 1
                skip_modality = skip_modality.int()
                # skip_modality[skip_modality < 1.5 ]=0
                skip_modality = None

            else:
                if skip_modality == "full":
                    skip_modality = None
                elif skip_modality == "random":
                    skip_modality = torch.rand(xeeg.shape[0])
                    # skip_modality[skip_modality > 0.5] = 2
                    # skip_modality[skip_modality <= 0.5] = 1
                    skip_modality[skip_modality > 0.66] = 2
                    skip_modality[skip_modality < 0.33] = 1
                    skip_modality = skip_modality.int()
                    # skip_modality[skip_modality == 0] = 1

                elif skip_modality == "eeg":
                    skip_modality = torch.ones(xeeg.shape[0]) * 1
                elif skip_modality == "eog":
                    skip_modality = torch.ones(xeeg.shape[0]) * 2
                elif skip_modality is "vae":
                    with torch.no_grad():
                        vae_eeg_output = self.enc_0([x[0]])
                        output_losses = self.enc_0.module.loss_function(vae_eeg_output[0], vae_eeg_output[1],
                                                                        vae_eeg_output[2], vae_eeg_output[3],
                                                                        reduction="none")
                        eeg_routing = output_losses["total"].flatten(start_dim=1).mean(dim=1)
                        # print(eeg_routing)
                        eeg_routing[eeg_routing < 3] = 0
                        eeg_routing[eeg_routing > 3] = 1

                        vae_eog_output = self.enc_1([x[1]])
                        output_losses = self.enc_1.module.loss_function(vae_eog_output[0], vae_eog_output[1],
                                                                        vae_eog_output[2], vae_eog_output[3],
                                                                        reduction="none")
                        eog_routing = output_losses["total"].flatten(start_dim=1).mean(dim=1)
                        # print(eog_routing)
                        eog_routing[eog_routing < 3] = 0
                        eog_routing[eog_routing > 3] = 2

                        skip_modality = eeg_routing + eog_routing
                        skip_modality[skip_modality == 3] = 0

            if skip_modality is not None:
                # xeeg_common = xeeg_sa[np.logical_or(skip_modality==0, skip_modality==1)] #Process EEG with EOG if you are not skipping EOG
                xeeg_common = xeeg[skip_modality == 0]  # Process EEG with EOG if you are not skipping EOG
                xeeg_common_init = xeeg[skip_modality == 0]  # Process EEG with EOG if you are not skipping EOG
                # xeeg_undir_sole = xeeg_sa[skip_modality==1] #If you skip EOG process EEG on its own
                xeeg_dir_sole = xeeg[skip_modality == 2]  # If you skip EOG process EEG on its own
                xeog_common = xeog[skip_modality == 0]  # Process EOG with EEG if you are not skipping EEG
                xeog_common_init = xeog[skip_modality == 0]  # Process EOG with EEG if you are not skipping EEG
                # xeog_undir_sole = xeog_sa[skip_modality==2] #If you skip EEG process EOG on its own
                xeog_dir_sole = xeog[skip_modality == 1]  # If you skip EEG process EOG on its own
            # print(skip_modality)

        if xeeg_common.shape[0] > 0 and xeog_common.shape[0] > 0:
            xeeg_common = self.inner_tf_eeg.forward_sa(xeeg_common)
            xeog_common = self.inner_tf_eog.forward_sa(xeog_common)

            # xeeg_sa_o = self.outer_positional_embedding(xeeg_sa_o)
            # xeog_sa_o = self.outer_positional_embedding(xeog_sa_o)
            xeeg_common_outer = self.outer_tf_eeg_out.forward_sa(xeeg_common, extract_norm=extract_norm)[:, :, :1]
            xeog_common_outer = self.outer_tf_eog_out.forward_sa(xeog_common, extract_norm=extract_norm)[:, :, :1]

            xeeg_sa_o, xeog_sa_o = xeeg_common_outer.squeeze(), xeog_common_outer.squeeze()

            xeeg_ca_common = self.inner_tf_eeg.forward_ca(xeeg_common_init, xeog_common)[:, :, :1]
            xeog_ca_common = self.inner_tf_eog.forward_ca(xeog_common_init, xeeg_common)[:, :, :1]

            if self.pos:
                xeeg_ca_common = self.outer_positional_embedding(xeeg_ca_common)
                xeog_ca_common = self.outer_positional_embedding(xeog_ca_common)

            # cls_token_outer_eeg = self.cls_token_outer_eeg.repeat(xeeg_ca_common.shape[0], xeeg_ca_common.shape[1], 1, 1, 1, 1)
            # cls_token_outer_eog = self.cls_token_outer_eog.repeat(xeog_ca_common.shape[0], xeog_ca_common.shape[1], 1, 1, 1, 1)
            #
            # xeeg_ca_common = torch.cat([cls_token_outer_eeg, xeog_ca_common], dim=3)
            # xeog_ca_common = torch.cat([cls_token_outer_eog, xeog_ca_common], dim=3)

            xeeg_ca_common_outer = self.outer_tf_eeg_out.forward_ca(xeeg_ca_common, xeog_common_outer,
                                                                    extract_norm=extract_norm)
            xeog_ca_common_outer = self.outer_tf_eog_out.forward_ca(xeog_ca_common, xeeg_common_outer,
                                                                    extract_norm=extract_norm)

            x_common = xeeg_ca_common_outer + xeog_ca_common_outer

        if "xeeg_dir_sole" in locals() and xeeg_dir_sole.shape[0] > 0:
            xeeg_ca_sole = self.inner_tf_eeg.forward_ca(xeeg_dir_sole)[:, :, :1]
            if self.pos:
                xeeg_ca_sole = self.outer_positional_embedding(xeeg_ca_sole)
            # cls_token_outer_eeg = self.cls_token_outer.repeat(xeeg_ca_sole.shape[0], xeeg_ca_sole.shape[1], 1, 1, 1, 1)
            # xeeg_ca_sole = torch.cat([cls_token_outer_eeg, xeeg_ca_sole], dim=3)
            xeeg_ca_sole = self.outer_tf_eeg_out.forward_ca(xeeg_ca_sole, extract_norm=extract_norm)[:, :, :, :1]

        if "xeog_dir_sole" in locals() and xeog_dir_sole.shape[0] > 0:
            xeog_ca_sole = self.inner_tf_eog.forward_ca(xeog_dir_sole)[:, :, :1]
            if self.pos:
                xeog_ca_sole = self.outer_positional_embedding(xeog_ca_sole)
            # cls_token_outer_eog = self.cls_token_outer.repeat(xeog_ca_sole.shape[0], xeog_ca_sole.shape[1], 1, 1, 1, 1)
            # xeog_ca_sole = torch.cat([cls_token_outer_eog, xeog_ca_sole], dim=3)
            xeog_ca_sole = self.outer_tf_eog_out.forward_ca(xeog_ca_sole, extract_norm=extract_norm)[:, :, :, :1]

        if "xeog_dir_sole" in locals() and "xeog_dir_sole" in locals() and (
                xeog_dir_sole.shape[0] > 0 or xeeg_dir_sole.shape[0] > 0):
            x = []
            counter = [0, 0, 0]
            for i in range(len(skip_modality)):
                if skip_modality[i] == 0:
                    x.append(
                        x_common[counter[int(skip_modality[i].item())]:counter[int(skip_modality[i].item())] + 1])
                elif skip_modality[i] == 1:
                    x.append(xeog_ca_sole[
                             counter[int(skip_modality[i].item())]:counter[int(skip_modality[i].item())] + 1])
                elif skip_modality[i] == 2:
                    x.append(xeeg_ca_sole[
                             counter[int(skip_modality[i].item())]:counter[int(skip_modality[i].item())] + 1])
                counter[int(skip_modality[i].item())] += 1
            x = torch.cat(x, dim=0)
        else:
            x = x_common

        output = {"output_features": x}

        if return_matches:
            if ("xeeg_sa_o" in locals() and xeeg_sa_o.shape[0] > 0 and len(xeeg_sa_o.shape) == 3) and (
                    "xeog_sa_o" in locals() and xeog_sa_o.shape[0] > 0 and len(xeog_sa_o.shape) == 3):
                x_match = torch.einsum('b o f , b m f -> b o m', xeeg_sa_o, xeog_sa_o)
                output["matches"] = x_match
            else:
                output["matches"] = None
        if return_inter_reps:
            output["intermediate_reps"] = [xeeg_sa_o, xeeg_sa_o]
        if return_order:
            x_sa_o = torch.cat([xeeg_sa_o, xeog_sa_o], dim=3)
            output["order"] = x_sa_o

        return output
class SleepEnc_COCA_EEG_EOG(nn.Module):
    def __init__(self, args):
        """
        :param encoder_filters_small, encoder_filters_big: filters of CNN output, quantized by 20
        :param encs_small, encs_big:
        """
        super().__init__()
        self.args = args

        d_model = args.dmodel  # 64*8

        inner_biased = args.inner_biased if "inner_biased" in args else False
        outer_biased = args.outer_biased if "outer_biased" in args else False
        rpos = args.rpos if "rpos" in args else False
        self.pos = args.pos if "pos" in args else True
        self.outer_rep = args.outer_rep if "outer_rep" in args else False

        if inner_biased == "gaussian_learned":
            inner_biased = Gaussian_Learned_Attention_Bias(d_model, heads=8, type="mul")

        # if outer_biased == "gaussian_learned":
        #     outer_biased = Gaussian_Learned_Attention_Bias(d_model, heads=8, type="mul")

        self.inner_tf_eeg = inner_ch_SA_CA(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, gbiased=inner_biased, num_layers=4)
        self.inner_tf_eog = inner_ch_SA_CA(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, gbiased=inner_biased, num_layers=4)

        if self.outer_rep:
            self.outer_tf_eeg = outer_mod_att_RA(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, gbiased=outer_biased, num_layers=4)
            self.outer_tf_eog = outer_mod_att_RA(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, gbiased=outer_biased, num_layers=4)

        self.outer_tf = outer_mod_att_RA(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, gbiased=outer_biased, num_layers=4)

        self.cls_token_eeg = nn.Parameter(torch.randn(1, 1, 1, 1, 1, d_model))
        self.cls_token_eog = nn.Parameter(torch.randn(1, 1, 1, 1, 1, d_model))

        self.cls_token_eeg_giveog = nn.Parameter(torch.randn(1, 1, 1, 1, 1, d_model))
        self.cls_token_eog_giveeg = nn.Parameter(torch.randn(1, 1, 1, 1, 1, d_model))

        if self.pos:
            self.inner_positional_embedding = huy_pos_inner(d_model, pos=False, inner=29, outer=21, modalities=1,
                                                              channels=1)
            self.outer_positional_embedding = huy_pos_outer(d_model, pos=False, inner=29, outer=21, modalities=1,
                                                            channels=1)

    def forward(self, x, return_matches=False, extract_norm=False, return_inter_reps=False, return_order=False):
        xeeg = x[0][:, :, :, :, 1:, :]  # mat
        xeog = x[1][:, :, :, :, 1:, :]  # mat

        xeeg = einops.rearrange(xeeg, "b outer mod ch f inner -> b outer inner mod ch f")
        xeog = einops.rearrange(xeog, "b outer mod ch f inner -> b outer inner mod ch f")

        if self.pos:
            xeeg = self.inner_positional_embedding(xeeg)
            xeog = self.inner_positional_embedding(xeog)

        cls_token_eeg = self.cls_token_eeg.repeat(xeeg.shape[0], xeeg.shape[1], 1, 1, xeeg.shape[3], 1)
        xeeg_u = torch.cat([cls_token_eeg, xeeg], dim=2)

        cls_token_eog = self.cls_token_eog.repeat(xeog.shape[0], xeog.shape[1], 1, 1, xeog.shape[3], 1)
        xeog_u = torch.cat([cls_token_eog, xeog], dim=2)

        xeeg_u = self.inner_tf_eeg.forward_sa(xeeg_u)
        xeog_u = self.inner_tf_eog.forward_sa(xeog_u)

        xeeg_u_o, xeog_u_o = xeeg_u[:, :, :], xeog_u[:, :, :]
        if self.outer_rep:
            # xeeg_sa_o = self.outer_positional_embedding(xeeg_sa_o)
            # xeog_sa_o = self.outer_positional_embedding(xeog_sa_o)
            xeeg_u_o = self.outer_tf_eeg(xeeg_u_o, extract_norm=extract_norm)
            xeog_u_o = self.outer_tf_eog(xeog_u_o, extract_norm=extract_norm)

        # xeeg = self.inner_positional_embedding(xeeg)
        # xeog = self.inner_positional_embedding(xeog)

        cls_token_eeg_giveog = self.cls_token_eeg_giveog.repeat(xeeg_u_o.shape[0], xeeg_u_o.shape[1], 1, 1, xeeg_u_o.shape[3], 1)
        xeeg_ca = torch.cat([cls_token_eeg_giveog, xeeg_u_o], dim=2)

        cls_token_eog_giveeg = self.cls_token_eog_giveeg.repeat(xeog_u_o.shape[0], xeog_u_o.shape[1], 1, 1, xeog_u_o.shape[3], 1)
        xeog_ca = torch.cat([cls_token_eog_giveeg, xeog_u_o], dim=2)

        xeeg_ca = self.inner_tf_eeg.forward_ca(xeeg_ca, xeog_ca)
        xeog_ca = self.inner_tf_eog.forward_ca(xeog_ca, xeeg_ca)
        
        # xeeg_ca_o, xeog_ca_o = xeeg_ca[:, :, :1].squeeze(dim=4).squeeze(dim=3).squeeze(dim=2), xeog_ca[:, :, :1].squeeze(dim=4).squeeze(dim=3).squeeze(dim=2)

        x = torch.cat([xeeg_ca[:, :, :1], xeog_ca[:, :, :1]], dim=3)
        x = self.outer_positional_embedding(x)
        x = self.outer_tf(x, extract_norm=extract_norm)

        output = {
            "outuput_features":{
                "eeg":xeeg_u_o[:, :, :1],
                "eog":xeog_u_o[:, :, :1],
                "combined":x,
            }
        }
        if return_matches:
            x_match = torch.einsum('b o f , b m f -> b o m', xeeg_u_o[:, :, :1], xeog_u_o[:, :, :1])
            output.update({"matches":x_match})
        if return_inter_reps:
            output.update({"inter_views":[xeeg_u_o, xeog_u_o]})
        if return_order:
            x_sa_o = torch.cat([xeeg_u_o[:, :, :1], xeog_u_o[:, :, :1]], dim=3)
            output.update({"order":x_sa_o})
        return output

class SleepEnc_COCA_true_EEG_EOG(nn.Module):
    def __init__(self, args):
        """
        :param encoder_filters_small, encoder_filters_big: filters of CNN output, quantized by 20
        :param encs_small, encs_big:
        """
        super().__init__()
        self.args = args

        d_model = args.dmodel  # 64*8

        inner_biased = args.inner_biased if "inner_biased" in args else False
        outer_biased = args.outer_biased if "outer_biased" in args else False
        rpos = args.rpos if "rpos" in args else False
        self.outer_rep = args.outer_rep if "outer_rep" in args else False

        if inner_biased == "gaussian_learned":
            inner_biased = Gaussian_Learned_Attention_Bias(d_model, heads=8, type="mul")

        # if outer_biased == "gaussian_learned":
        #     outer_biased = Gaussian_Learned_Attention_Bias(d_model, heads=8, type="mul")

        self.inner_tf_eeg = inner_ch_SA_CA(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, gbiased=inner_biased, num_layers=4)
        self.inner_tf_eog = inner_ch_SA_CA(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, gbiased=inner_biased, num_layers=4)

        if self.outer_rep:
            self.outer_tf_eeg = outer_mod_att_RA(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, gbiased=outer_biased, num_layers=4)
            self.outer_tf_eog = outer_mod_att_RA(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, gbiased=outer_biased, num_layers=4)

        self.outer_tf = outer_mod_att_RA(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, gbiased=outer_biased, num_layers=4)

        self.cls_token_eeg = nn.Parameter(torch.randn(1, 1, 1, 1, 1, d_model))
        self.cls_token_eog = nn.Parameter(torch.randn(1, 1, 1, 1, 1, d_model))

        self.inner_positional_embedding = huy_pos_inner(d_model, pos=False, inner=29, outer=21, modalities=1,
                                                          channels=1)
        self.outer_positional_embedding = huy_pos_outer(d_model, pos=False, inner=29, outer=21, modalities=1,
                                                        channels=1)

    def forward(self, x, return_matches=False, extract_norm=False, return_inter_reps=False, return_order=False):
        xeeg = x[0][:, :, :, :, 1:, :]  # mat
        xeog = x[1][:, :, :, :, 1:, :]  # mat

        xeeg = einops.rearrange(xeeg, "b outer mod ch f inner -> b outer inner mod ch f")
        xeog = einops.rearrange(xeog, "b outer mod ch f inner -> b outer inner mod ch f")

        cls_token_eeg = self.cls_token_eeg.repeat(xeeg.shape[0], xeeg.shape[1], 1, 1, xeeg.shape[3], 1)
        xeeg = torch.cat([cls_token_eeg, xeeg], dim=2)

        cls_token_eog = self.cls_token_eog.repeat(xeog.shape[0], xeog.shape[1], 1, 1, xeog.shape[3], 1)
        xeog = torch.cat([cls_token_eog, xeog], dim=2)

        xeeg_sa = self.inner_tf_eeg.forward_sa(xeeg)
        xeog_sa = self.inner_tf_eog.forward_sa(xeog)

        if self.outer_rep:
            # xeeg_sa_o = self.outer_positional_embedding(xeeg_sa)
            # xeog_sa_o = self.outer_positional_embedding(xeog_sa)
            xeeg_sa_o = self.outer_tf_eeg(xeeg_sa, extract_norm=extract_norm)
            xeog_sa_o = self.outer_tf_eog(xeog_sa, extract_norm=extract_norm)

        # xeeg = self.inner_positional_embedding(xeeg)
        # xeog = self.inner_positional_embedding(xeog)

        xeeg_ca = self.inner_tf_eeg.forward_ca(xeeg_sa, xeog_sa_o)
        xeog_ca = self.inner_tf_eog.forward_ca(xeog_sa, xeeg_sa_o)

        xeeg_sa_o_p, xeog_sa_o_p = xeeg_sa_o[:, :, :1].squeeze(dim=4).squeeze(dim=3).squeeze(dim=2), xeog_sa_o[:, :, :1].squeeze(dim=4).squeeze(dim=3).squeeze(dim=2)

        x = torch.cat([xeeg_ca[:, :, :1], xeog_ca[:, :, :1]], dim=3)
        x = self.outer_positional_embedding(x)
        x = self.outer_tf(x, extract_norm=extract_norm)

        output = [x]
        if return_matches:
            x_match = torch.einsum('b o f , b m f -> b o m', xeeg_sa_o_p, xeog_sa_o_p)
            output.append(x_match)
        if return_inter_reps:
            output.append([xeeg_sa, xeog_sa])
        if return_order:
            x_sa_o = torch.cat([xeeg_sa_o[:, :, :1], xeog_sa_o[:, :, :1]], dim=3)
            output.append(x_sa_o)

        return output
class SleepEnc_COCA_seperable_EEG_EOG(nn.Module):
    def __init__(self, args):
        """
        :param encoder_filters_small, encoder_filters_big: filters of CNN output, quantized by 20
        :param encs_small, encs_big:
        """
        super().__init__()
        self.args = args

        d_model = args.dmodel  # 64*8

        inner_biased = args.inner_biased if "inner_biased" in args else False
        outer_biased = args.outer_biased if "outer_biased" in args else False
        rpos = args.rpos if "rpos" in args else False
        self.outer_rep = args.outer_rep if "outer_rep" in args else False

        if inner_biased == "gaussian_learned":
            inner_biased = Gaussian_Learned_Attention_Bias(d_model, heads=8, type="mul")

        # if outer_biased == "gaussian_learned":
        #     outer_biased = Gaussian_Learned_Attention_Bias(d_model, heads=8, type="mul")

        self.inner_tf_eeg = inner_ch_SA_CA(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, gbiased=inner_biased, num_layers=4)
        self.inner_tf_eog = inner_ch_SA_CA(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, gbiased=inner_biased, num_layers=4)

        if self.outer_rep:
            self.outer_tf_eeg = outer_mod_att_RA(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, gbiased=outer_biased, num_layers=4)
            self.outer_tf_eog = outer_mod_att_RA(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, gbiased=outer_biased, num_layers=4)

        self.outer_tf_eeg_plus = outer_mod_att_RA(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, gbiased=outer_biased, num_layers=4)
        self.outer_tf_eog_plus = outer_mod_att_RA(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, gbiased=outer_biased, num_layers=4)

        self.cls_token_eeg = nn.Parameter(torch.randn(1, 1, 1, 1, 1, d_model))
        self.cls_token_eog = nn.Parameter(torch.randn(1, 1, 1, 1, 1, d_model))

        self.inner_positional_embedding = huy_pos_inner(d_model, pos=False, inner=29, outer=21, modalities=1,
                                                          channels=1)
        self.outer_positional_embedding = huy_pos_outer(d_model, pos=False, inner=29, outer=21, modalities=1,
                                                        channels=1)

    def forward(self, x, return_matches=False, extract_norm=False, return_inter_reps=False, return_order=False):
        xeeg = x[0][:, :, :, :, 1:, :]  # mat
        xeog = x[1][:, :, :, :, 1:, :]  # mat

        xeeg = einops.rearrange(xeeg, "b outer mod ch f inner -> b outer inner mod ch f")
        xeog = einops.rearrange(xeog, "b outer mod ch f inner -> b outer inner mod ch f")

        cls_token_eeg = self.cls_token_eeg.repeat(xeeg.shape[0], xeeg.shape[1], 1, 1, xeeg.shape[3], 1)
        xeeg = torch.cat([cls_token_eeg, xeeg], dim=2)

        cls_token_eog = self.cls_token_eog.repeat(xeog.shape[0], xeog.shape[1], 1, 1, xeog.shape[3], 1)
        xeog = torch.cat([cls_token_eog, xeog], dim=2)

        xeeg_sa = self.inner_tf_eeg.forward_sa(xeeg)
        xeog_sa = self.inner_tf_eog.forward_sa(xeog)

        if self.outer_rep:
            # xeeg_sa_o = self.outer_positional_embedding(xeeg_sa)
            # xeog_sa_o = self.outer_positional_embedding(xeog_sa)
            xeeg_sa_o = self.outer_tf_eeg(xeeg_sa, extract_norm=extract_norm)
            xeog_sa_o = self.outer_tf_eog(xeog_sa, extract_norm=extract_norm)

        # xeeg = self.inner_positional_embedding(xeeg)
        # xeog = self.inner_positional_embedding(xeog)

        xeeg_ca = self.inner_tf_eeg.forward_ca(xeeg_sa, xeog_sa_o)
        xeog_ca = self.inner_tf_eog.forward_ca(xeog_sa, xeeg_sa_o)

        xeeg_sa_o_p, xeog_sa_o_p = xeeg_sa_o[:, :, :1].squeeze(dim=4).squeeze(dim=3).squeeze(dim=2), xeog_sa_o[:, :, :1].squeeze(dim=4).squeeze(dim=3).squeeze(dim=2)

        # x = torch.cat([xeeg_ca[:, :, :1], xeog_ca[:, :, :1]], dim=3)
        xeeg_ca = self.outer_positional_embedding(xeeg_ca[:, :, :1])
        xeog_ca = self.outer_positional_embedding(xeog_ca[:, :, :1])
        xeeg_ca = self.outer_tf_eeg_plus(xeeg_ca, extract_norm=extract_norm)
        xeog_ca = self.outer_tf_eog_plus(xeog_ca, extract_norm=extract_norm)
        # x = torch.cat([xeeg_ca[:, :, :1], xeog_ca[:, :, :1]], dim=3)

        output = [[xeeg_ca, xeog_ca]]
        if return_matches:
            x_match = torch.einsum('b o f , b m f -> b o m', xeeg_sa_o_p, xeog_sa_o_p)
            output.append(x_match)
        if return_inter_reps:
            output.append([xeeg_sa, xeog_sa])
        if return_order:
            x_sa_o = torch.cat([xeeg_sa_o[:, :, :1], xeog_sa_o[:, :, :1]], dim=3)
            output.append(x_sa_o)

        return output

class SleepEnc_seperable_EEG_EOG(nn.Module):
    def __init__(self, args):
        """
        :param encoder_filters_small, encoder_filters_big: filters of CNN output, quantized by 20
        :param encs_small, encs_big:
        """
        super().__init__()
        self.args = args

        d_model = args.dmodel  # 64*8

        inner_biased = args.inner_biased if "inner_biased" in args else False
        outer_biased = args.outer_biased if "outer_biased" in args else False
        rpos = args.rpos if "rpos" in args else False
        self.outer_rep = args.outer_rep if "outer_rep" in args else False

        if inner_biased == "gaussian_learned":
            inner_biased = Gaussian_Learned_Attention_Bias(d_model, heads=8, type="mul")

        # if outer_biased == "gaussian_learned":
        #     outer_biased = Gaussian_Learned_Attention_Bias(d_model, heads=8, type="mul")

        self.inner_tf_eeg = inner_ch_att_RA(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, gbiased=inner_biased, num_layers=8)
        self.inner_tf_eog = inner_ch_att_RA(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, gbiased=inner_biased, num_layers=8)

        self.outer_tf_eeg_plus = outer_mod_att_RA(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, gbiased=outer_biased, num_layers=4)
        self.outer_tf_eog_plus = outer_mod_att_RA(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, gbiased=outer_biased, num_layers=4)

        self.cls_token_eeg = nn.Parameter(torch.randn(1, 1, 1, 1, 1, d_model))
        self.cls_token_eog = nn.Parameter(torch.randn(1, 1, 1, 1, 1, d_model))

        self.inner_positional_embedding = huy_pos_inner(d_model, pos=False, inner=29, outer=21, modalities=1,
                                                          channels=1)
        self.outer_positional_embedding = huy_pos_outer(d_model, pos=False, inner=29, outer=21, modalities=1,
                                                        channels=1)

    def forward(self, x, return_matches=False, extract_norm=False, return_inter_reps=False, return_order=False):
        xeeg = x[0][:, :, :, :, 1:, :]  # mat
        xeog = x[1][:, :, :, :, 1:, :]  # mat

        xeeg = einops.rearrange(xeeg, "b outer mod ch f inner -> b outer inner mod ch f")
        xeog = einops.rearrange(xeog, "b outer mod ch f inner -> b outer inner mod ch f")

        cls_token_eeg = self.cls_token_eeg.repeat(xeeg.shape[0], xeeg.shape[1], 1, 1, xeeg.shape[3], 1)
        xeeg = torch.cat([cls_token_eeg, xeeg], dim=2)

        xeeg_sa = self.inner_tf_eeg(xeeg)
        xeeg_ca = self.outer_positional_embedding(xeeg_sa[:, :, :1])
        xeeg_ca = self.outer_tf_eeg_plus(xeeg_ca, extract_norm=extract_norm)
        xeeg_sa_p = xeeg_sa[:, :, :1].squeeze(dim=4).squeeze(dim=3).squeeze(dim=2)

        cls_token_eog = self.cls_token_eog.repeat(xeog.shape[0], xeog.shape[1], 1, 1, xeog.shape[3], 1)
        xeog = torch.cat([cls_token_eog, xeog], dim=2)

        xeog_sa = self.inner_tf_eog(xeog)
        xeog_ca = self.outer_positional_embedding(xeog_sa[:, :, :1])
        xeog_ca = self.outer_tf_eog_plus(xeog_ca, extract_norm=extract_norm)
        xeog_sa_p = xeog_sa[:, :, :1].squeeze(dim=4).squeeze(dim=3).squeeze(dim=2)

        output = [[xeeg_ca, xeog_ca]]

        if return_matches:
            x_match = torch.einsum('b o f , b m f -> b o m', xeeg_sa_p, xeog_sa_p)
            output.append(x_match)
        if return_inter_reps:
            output.append([xeeg_sa, xeog_sa])
        if return_order:
            x_sa_o = torch.cat([xeeg_sa_p, xeeg_sa_p], dim=3)
            output.append(x_sa_o)

        return output
class SleepEnc_seperable_n_merged_EEG_EOG(nn.Module):
    def __init__(self, args):
        """
        :param encoder_filters_small, encoder_filters_big: filters of CNN output, quantized by 20
        :param encs_small, encs_big:
        """
        super().__init__()
        self.args = args

        d_model = args.dmodel  # 64*8

        inner_biased = args.inner_biased if "inner_biased" in args else False
        outer_biased = args.outer_biased if "outer_biased" in args else False
        rpos = args.rpos if "rpos" in args else False
        self.outer_rep = args.outer_rep if "outer_rep" in args else False

        if inner_biased == "gaussian_learned":
            inner_biased = Gaussian_Learned_Attention_Bias(d_model, heads=8, type="mul")

        # if outer_biased == "gaussian_learned":
        #     outer_biased = Gaussian_Learned_Attention_Bias(d_model, heads=8, type="mul")

        self.inner_tf_eeg = inner_ch_att_RA(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, gbiased=inner_biased, num_layers=4)
        self.inner_tf_eeg_plus = inner_ch_att_RA(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, gbiased=inner_biased, num_layers=4)

        self.inner_tf_eog = inner_ch_att_RA(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, gbiased=inner_biased, num_layers=4)
        self.inner_tf_eog_plus = inner_ch_att_RA(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, gbiased=inner_biased, num_layers=4)

        self.inner_tf_merged = inner_ch_att_RA(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, gbiased=inner_biased, num_layers=4)
        self.outer_tf_merged = outer_mod_att_RA(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, gbiased=outer_biased, num_layers=4)

        self.outer_tf_eeg_plus = outer_mod_att_RA(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, gbiased=outer_biased, num_layers=4)
        self.outer_tf_eog_plus = outer_mod_att_RA(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, gbiased=outer_biased, num_layers=4)

        self.cls_token_eeg = nn.Parameter(torch.randn(1, 1, 1, 1, 1, d_model))
        self.cls_token_eog = nn.Parameter(torch.randn(1, 1, 1, 1, 1, d_model))
        self.cls_token_merged = nn.Parameter(torch.randn(1, 1, 1, 1, 1, d_model))

        self.inner_positional_embedding = huy_pos_inner(d_model, pos=False, inner=29, outer=21, modalities=1,
                                                          channels=1)
        self.outer_positional_embedding = huy_pos_outer(d_model, pos=False, inner=29, outer=21, modalities=1,
                                                        channels=1)

    def forward(self, x, return_matches=False, extract_norm=False, return_inter_reps=False, return_order=False):
        xeeg = x[0][:, :, :, :, 1:, :]  # mat
        xeog = x[1][:, :, :, :, 1:, :]  # mat

        xeeg = einops.rearrange(xeeg, "b outer mod ch f inner -> b outer inner mod ch f")
        xeog = einops.rearrange(xeog, "b outer mod ch f inner -> b outer inner mod ch f")

        cls_token_eeg = self.cls_token_eeg.repeat(xeeg.shape[0], xeeg.shape[1], 1, 1, xeeg.shape[3], 1)
        xeeg = torch.cat([cls_token_eeg, xeeg], dim=2)

        xeeg_sa = self.inner_tf_eeg(xeeg)
        xeeg_sa_plus = self.inner_tf_eeg_plus(xeeg_sa)
        xeeg_ca = self.outer_positional_embedding(xeeg_sa_plus[:, :, :1])
        xeeg_ca = self.outer_tf_eeg_plus(xeeg_ca, extract_norm=extract_norm)
        xeeg_sa_p = xeeg_sa[:, :, :1].squeeze(dim=4).squeeze(dim=3).squeeze(dim=2)

        cls_token_eog = self.cls_token_eog.repeat(xeog.shape[0], xeog.shape[1], 1, 1, xeog.shape[3], 1)
        xeog = torch.cat([cls_token_eog, xeog], dim=2)

        xeog_sa = self.inner_tf_eog(xeog)
        xeog_sa_plus = self.inner_tf_eog_plus(xeog_sa)
        xeog_ca = self.outer_positional_embedding(xeog_sa_plus[:, :, :1])
        xeog_ca = self.outer_tf_eog_plus(xeog_ca, extract_norm=extract_norm)
        xeog_sa_p = xeog_sa[:, :, :1].squeeze(dim=4).squeeze(dim=3).squeeze(dim=2)

        cls_token_merged = self.cls_token_merged.repeat(xeeg.shape[0], xeeg.shape[1], 1, 1, xeeg.shape[3], 1)
        x_merged = torch.cat([cls_token_merged, xeeg_sa, xeog_sa], dim=2)
        x_merged_sa = self.inner_tf_eog(x_merged)
        x_merged_ca = self.outer_positional_embedding(x_merged_sa[:, :, :1])
        x_merged_ca = self.outer_tf_eog_plus(x_merged_ca, extract_norm=extract_norm)

        output = [[xeeg_ca, xeog_ca, x_merged_ca]]

        if return_matches:
            x_match = torch.einsum('b o f , b m f -> b o m', xeeg_sa_p, xeog_sa_p)
            output.append(x_match)
        if return_inter_reps:
            output.append([xeeg_sa, xeog_sa])
        if return_order:
            x_sa_o = torch.cat([xeeg_sa_p, xeeg_sa_p], dim=3)
            output.append(x_sa_o)

        return output

class SleepEnc_BLIP_Double_EEG_EOG(nn.Module):
    def __init__(self, args):
        """
        :param encoder_filters_small, encoder_filters_big: filters of CNN output, quantized by 20
        :param encs_small, encs_big:
        """
        super().__init__()
        self.args = args

        d_model = args.dmodel  # 64*8

        inner_biased, outer_biased, rpos = False, False, False
        if "inner_biased" in args:
            inner_biased = args.inner_biased
        if "outer_biased" in args:
            outer_biased = args.outer_biased
        if "rpos" in args:
            rpos = args.rpos

        if inner_biased == "gaussian_learned":
            inner_biased = Gaussian_Learned_Attention_Bias(d_model, heads=8, type="mul")

        # if outer_biased == "gaussian_learned":
        #     outer_biased = Gaussian_Learned_Attention_Bias(d_model, heads=8, type="mul")

        self.inner_tf_eeg_l0 = inner_ch_SA_CA_CA(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, gbiased=inner_biased, num_layers=4)
        self.inner_tf_eeg_l1 = inner_ch_SA_CA_CA(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, gbiased=inner_biased, num_layers=4)
        self.inner_tf_eeg_l2 = inner_ch_SA_CA_CA(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, gbiased=inner_biased, num_layers=4)
        self.inner_tf_eeg_l3 = inner_ch_SA_CA_CA(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, gbiased=inner_biased, num_layers=4)

        self.inner_tf_eog_l0 = inner_ch_SA_CA_CA(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, gbiased=inner_biased, num_layers=4)
        self.inner_tf_eog_l1 = inner_ch_SA_CA_CA(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, gbiased=inner_biased, num_layers=4)
        self.inner_tf_eog_l2 = inner_ch_SA_CA_CA(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, gbiased=inner_biased, num_layers=4)
        self.inner_tf_eog_l3 = inner_ch_SA_CA_CA(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, gbiased=inner_biased, num_layers=4)

        # self.outer_tf_eeg = outer_mod_att_RA(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, gbiased=outer_biased, num_layers=4)
        # self.outer_tf_eog = outer_mod_att_RA(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, gbiased=outer_biased, num_layers=4)

        self.outer_tf = outer_mod_att_RA(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, gbiased=outer_biased, num_layers=4)

        self.cls_token = nn.Parameter(torch.randn(1, 1, 1, 1, 1, d_model))

        self.inner_positional_embedding = huy_pos_inner(d_model, pos=False, inner=29, outer=21, modalities=1,
                                                          channels=1)
        self.outer_positional_embedding = huy_pos_outer(d_model, pos=False, inner=29, outer=21, modalities=1,
                                                        channels=1)

    def forward(self, x, inits=None, extract_norm=False):
        xeeg = x[0][:, :, :, :, 1:, :]  # mat
        xeog = x[1][:, :, :, :, 1:, :]  # mat

        xeeg = einops.rearrange(xeeg, "b outer mod ch f inner -> b outer inner mod ch f")
        xeog = einops.rearrange(xeog, "b outer mod ch f inner -> b outer inner mod ch f")

        xeeg = self.inner_positional_embedding(xeeg)
        xeog = self.inner_positional_embedding(xeog)

        cls_token_eeg = self.cls_token.repeat(xeeg.shape[0], xeeg.shape[1], 1, 1, xeeg.shape[3], 1)
        xeeg = torch.cat([cls_token_eeg, xeeg], dim=2)

        cls_token_eog = self.cls_token.repeat(xeog.shape[0], xeog.shape[1], 1, 1, xeog.shape[3], 1)
        xeog = torch.cat([cls_token_eog, xeog], dim=2)

        xeeg_sa = self.inner_tf_eeg_l0.forward_sa(xeeg)
        xeeg_sa = self.inner_tf_eeg_l1.forward_sa(xeeg_sa)
        xeeg_sa = self.inner_tf_eeg_l2.forward_sa(xeeg_sa)
        xeeg_sa = self.inner_tf_eeg_l3.forward_sa(xeeg_sa)

        xeog_sa = self.inner_tf_eog_l0.forward_sa(xeog)
        xeog_sa = self.inner_tf_eog_l1.forward_sa(xeog_sa)
        xeog_sa = self.inner_tf_eog_l2.forward_sa(xeog_sa)
        xeog_sa = self.inner_tf_eog_l3.forward_sa(xeog_sa)

        xeeg_ca = self.inner_tf_eeg_l0.forward_ca(xeeg, xeeg_sa, xeog_sa)
        xeeg_ca = self.inner_tf_eeg_l1.forward_ca(xeeg_ca, xeeg_sa, xeog_sa)
        xeeg_ca = self.inner_tf_eeg_l2.forward_ca(xeeg_ca, xeeg_sa, xeog_sa)
        xeeg_ca = self.inner_tf_eeg_l3.forward_ca(xeeg_ca, xeeg_sa, xeog_sa)

        xeog_ca = self.inner_tf_eog_l0.forward_ca(xeog, xeog_sa, xeeg_sa)
        xeog_ca = self.inner_tf_eog_l1.forward_ca(xeog_ca, xeog_sa, xeeg_sa)
        xeog_ca = self.inner_tf_eog_l2.forward_ca(xeog_ca, xeog_sa, xeeg_sa)
        xeog_ca = self.inner_tf_eog_l3.forward_ca(xeog_ca, xeog_sa, xeeg_sa)

        xeeg_sa, xeog_sa = xeeg_sa[:, :, :1].squeeze(dim=4).squeeze(dim=3).squeeze(dim=2), xeog_sa[:, :, :1].squeeze(dim=4).squeeze(dim=3).squeeze(dim=2)

        x_match = torch.einsum('b o f , b m f -> b o m', xeeg_sa, xeog_sa)

        x = torch.cat([xeeg_ca[:, :, :1], xeog_ca[:, :, :1]], dim=3)
        x = self.outer_positional_embedding(x)
        x = self.outer_tf(x, extract_norm=extract_norm)

        return x, x_match
class SleepEnc_BLIP_shared_EEG_EOG(nn.Module):
    def __init__(self, args):
        """
        :param encoder_filters_small, encoder_filters_big: filters of CNN output, quantized by 20
        :param encs_small, encs_big:
        """
        super().__init__()
        self.args = args

        d_model = args.dmodel  # 64*8

        inner_biased, outer_biased, rpos = False, False, False
        if "inner_biased" in args:
            inner_biased = args.inner_biased
        if "outer_biased" in args:
            outer_biased = args.outer_biased
        if "rpos" in args:
            rpos = args.rpos

        if inner_biased == "gaussian_learned":
            inner_biased = Gaussian_Learned_Attention_Bias(d_model, heads=8, type="mul")

        # if outer_biased == "gaussian_learned":
        #     outer_biased = Gaussian_Learned_Attention_Bias(d_model, heads=8, type="mul")

        self.inner_tf_eeg_l0 = inner_ch_SA_CA_shared(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, gbiased=inner_biased, num_layers=4)
        self.inner_tf_eeg_l1 = inner_ch_SA_CA_shared(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, gbiased=inner_biased, num_layers=1)
        self.inner_tf_eeg_l2 = inner_ch_SA_CA_shared(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, gbiased=inner_biased, num_layers=1)
        self.inner_tf_eeg_l3 = inner_ch_SA_CA_shared(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, gbiased=inner_biased, num_layers=1)

        self.inner_tf_eog_l0 = inner_ch_SA_CA_shared(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, gbiased=inner_biased, num_layers=4)
        self.inner_tf_eog_l1 = inner_ch_SA_CA_shared(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, gbiased=inner_biased, num_layers=1)
        self.inner_tf_eog_l2 = inner_ch_SA_CA_shared(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, gbiased=inner_biased, num_layers=1)
        self.inner_tf_eog_l3 = inner_ch_SA_CA_shared(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, gbiased=inner_biased, num_layers=1)

        # self.outer_tf_eeg = outer_mod_att_RA(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, gbiased=outer_biased, num_layers=4)
        # self.outer_tf_eog = outer_mod_att_RA(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, gbiased=outer_biased, num_layers=4)

        self.outer_tf = outer_mod_att_RA(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, gbiased=outer_biased, num_layers=4)

        self.cls_token = nn.Parameter(torch.randn(1, 1, 1, 1, 1, d_model))

        self.inner_positional_embedding = huy_pos_inner(d_model, pos=False, inner=29, outer=21, modalities=1,
                                                          channels=1)
        self.outer_positional_embedding = huy_pos_outer(d_model, pos=False, inner=29, outer=21, modalities=1,
                                                        channels=1)

    def forward(self, x, inits=None, extract_norm=False):
        xeeg = x[0][:, :, :, :, 1:, :]  # mat
        xeog = x[1][:, :, :, :, 1:, :]  # mat

        xeeg = einops.rearrange(xeeg, "b outer mod ch f inner -> b outer inner mod ch f")
        xeog = einops.rearrange(xeog, "b outer mod ch f inner -> b outer inner mod ch f")

        xeeg = self.inner_positional_embedding(xeeg)
        xeog = self.inner_positional_embedding(xeog)

        cls_token_eeg = self.cls_token.repeat(xeeg.shape[0], xeeg.shape[1], 1, 1, xeeg.shape[3], 1)
        xeeg = torch.cat([cls_token_eeg, xeeg], dim=2)

        cls_token_eog = self.cls_token.repeat(xeog.shape[0], xeog.shape[1], 1, 1, xeog.shape[3], 1)
        xeog = torch.cat([cls_token_eog, xeog], dim=2)

        # xeeg_sa = self.inner_tf_eeg.forward_sa(xeeg)
        xeeg_sa = self.inner_tf_eeg_l0.forward_sa(xeeg)
        xeeg_sa = self.inner_tf_eeg_l1.forward_sa(xeeg_sa)
        xeeg_sa = self.inner_tf_eeg_l2.forward_sa(xeeg_sa)
        xeeg_sa = self.inner_tf_eeg_l3.forward_sa(xeeg_sa)
        #
        # xeog_sa = self.inner_tf_eog.forward_sa(xeog)
        xeog_sa = self.inner_tf_eog_l0.forward_sa(xeog)
        xeog_sa = self.inner_tf_eog_l1.forward_sa(xeog_sa)
        xeog_sa = self.inner_tf_eog_l2.forward_sa(xeog_sa)
        xeog_sa = self.inner_tf_eog_l3.forward_sa(xeog_sa)

        # xeeg_ca = self.inner_tf_eeg.forward_ca(xeeg, xeog_sa)
        xeeg_ca = self.inner_tf_eeg_l0.forward_ca(xeeg, xeog_sa)
        xeeg_ca = self.inner_tf_eeg_l1.forward_ca(xeeg_ca, xeog_sa)
        xeeg_ca = self.inner_tf_eeg_l2.forward_ca(xeeg_ca, xeog_sa)
        xeeg_ca = self.inner_tf_eeg_l3.forward_ca(xeeg_ca, xeog_sa)

        # xeog_ca = self.inner_tf_eog.forward_ca(xeog, xeeg_sa)
        xeog_ca = self.inner_tf_eog_l0.forward_ca(xeog, xeeg_sa)
        xeog_ca = self.inner_tf_eog_l1.forward_ca(xeog_ca, xeeg_sa)
        xeog_ca = self.inner_tf_eog_l2.forward_ca(xeog_ca, xeeg_sa)
        xeog_ca = self.inner_tf_eog_l3.forward_ca(xeog_ca, xeeg_sa)

        xeeg_sa, xeog_sa = xeeg_sa[:, :, :1].squeeze(dim=4).squeeze(dim=3).squeeze(dim=2), xeog_sa[:, :, :1].squeeze(dim=4).squeeze(dim=3).squeeze(dim=2)

        x_match = torch.einsum('b o f , b m f -> b o m', xeeg_sa, xeog_sa)

        x = torch.cat([xeeg_ca[:, :, :1], xeog_ca[:, :, :1]], dim=3)
        x = self.outer_positional_embedding(x)
        x = self.outer_tf(x, extract_norm=extract_norm)

        return x, x_match
class SleepEnc_BLIP_EEG_EOG_Simple(nn.Module):
    def __init__(self, args):
        """
        :param encoder_filters_small, encoder_filters_big: filters of CNN output, quantized by 20
        :param encs_small, encs_big:
        """
        super().__init__()
        self.args = args

        d_model = args.dmodel  # 64*8

        inner_biased, outer_biased, rpos = False, False, False
        if "inner_biased" in args:
            inner_biased = args.inner_biased
        if "outer_biased" in args:
            outer_biased = args.outer_biased
        if "rpos" in args:
            rpos = args.rpos

        if inner_biased == "gaussian_learned":
            inner_biased = Gaussian_Learned_Attention_Bias(d_model, heads=8, type="mul")

        # if outer_biased == "gaussian_learned":
        #     outer_biased = Gaussian_Learned_Attention_Bias(d_model, heads=8, type="mul")

        self.inner_tf_eeg = inner_ch_att_RA(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, gbiased=inner_biased, num_layers=4)
        self.inner_tf_eog = inner_ch_att_RA(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, gbiased=inner_biased, num_layers=4)

        self.outer_tf = outer_mod_att_RA(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, gbiased=outer_biased, num_layers=4)

        self.cls_token = nn.Parameter(torch.randn(1, 1, 1, 1, 1, d_model))

        self.inner_positional_embedding = huy_pos_inner(d_model, pos=False, inner=29, outer=21, modalities=1,
                                                          channels=1)
        self.outer_positional_embedding = huy_pos_outer(d_model, pos=False, inner=29, outer=21, modalities=1,
                                                        channels=1)

    def forward(self, x, inits=None, extract_norm=False):
        xeeg = x[0][:, :, :, :, 1:, :]  # mat
        xeog = x[1][:, :, :, :, 1:, :]  # mat

        xeeg = einops.rearrange(xeeg, "b outer mod ch f inner -> b outer inner mod ch f")
        xeog = einops.rearrange(xeog, "b outer mod ch f inner -> b outer inner mod ch f")

        xeeg = self.inner_positional_embedding(xeeg)
        xeog = self.inner_positional_embedding(xeog)

        cls_token_eeg = self.cls_token.repeat(xeeg.shape[0], xeeg.shape[1], 1, 1, xeeg.shape[3], 1)
        xeeg = torch.cat([cls_token_eeg, xeeg], dim=2)

        cls_token_eog = self.cls_token.repeat(xeog.shape[0], xeog.shape[1], 1, 1, xeog.shape[3], 1)
        xeog = torch.cat([cls_token_eog, xeog], dim=2)

        xeeg_sa = self.inner_tf_eeg(xeeg)
        xeog_sa = self.inner_tf_eog(xeog)

        xeeg_sa_p, xeog_sa_p = xeeg_sa[:, :, :1].squeeze(dim=4).squeeze(dim=3).squeeze(dim=2), xeog_sa[:, :, :1].squeeze(dim=4).squeeze(dim=3).squeeze(dim=2)

        x_match = torch.einsum('b o f , b m f -> b o m', xeeg_sa_p, xeog_sa_p)

        x = torch.cat([xeeg_sa[:, :, :1], xeeg_sa[:, :, :1]], dim=3)
        x = self.outer_positional_embedding(x)
        x = self.outer_tf(x, extract_norm=extract_norm)

        return x, x_match
class SleepEnc_BLIP_OUTER_EEG_EOG_Simple(nn.Module):
    def __init__(self, args):
        """
        :param encoder_filters_small, encoder_filters_big: filters of CNN output, quantized by 20
        :param encs_small, encs_big:
        """
        super().__init__()
        self.args = args

        d_model = args.dmodel  # 64*8

        inner_biased, outer_biased, rpos = False, False, False
        if "inner_biased" in args:
            inner_biased = args.inner_biased
        if "outer_biased" in args:
            outer_biased = args.outer_biased
        if "rpos" in args:
            rpos = args.rpos

        if inner_biased == "gaussian_learned":
            inner_biased = Gaussian_Learned_Attention_Bias(d_model, heads=8, type="mul")

        # if outer_biased == "gaussian_learned":
        #     outer_biased = Gaussian_Learned_Attention_Bias(d_model, heads=8, type="mul")

        self.inner_tf_eeg = inner_ch_att_RA(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, gbiased=inner_biased, num_layers=4)
        self.inner_tf_eog = inner_ch_att_RA(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, gbiased=inner_biased, num_layers=4)

        self.outer_tf_inner_eeg = outer_mod_att_RA(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, gbiased=outer_biased, num_layers=4)
        self.outer_tf_inner_eog = outer_mod_att_RA(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, gbiased=outer_biased, num_layers=4)

        self.cls_token = nn.Parameter(torch.randn(1, 1, 1, 1, 1, d_model))

        self.inner_positional_embedding = huy_pos_inner(d_model, pos=False, inner=29, outer=21, modalities=1,
                                                          channels=1)
        self.outer_positional_embedding = huy_pos_outer(d_model, pos=False, inner=29, outer=21, modalities=1,
                                                        channels=1)

    def forward(self, x, inits=None, extract_norm=False):
        xeeg = x[0][:, :, :, :, 1:, :]  # mat
        xeog = x[1][:, :, :, :, 1:, :]  # mat

        xeeg = einops.rearrange(xeeg, "b outer mod ch f inner -> b outer inner mod ch f")
        xeog = einops.rearrange(xeog, "b outer mod ch f inner -> b outer inner mod ch f")

        xeeg = self.inner_positional_embedding(xeeg)
        xeog = self.inner_positional_embedding(xeog)

        cls_token_eeg = self.cls_token.repeat(xeeg.shape[0], xeeg.shape[1], 1, 1, xeeg.shape[3], 1)
        xeeg = torch.cat([cls_token_eeg, xeeg], dim=2)

        cls_token_eog = self.cls_token.repeat(xeog.shape[0], xeog.shape[1], 1, 1, xeog.shape[3], 1)
        xeog = torch.cat([cls_token_eog, xeog], dim=2)

        xeeg_sa = self.inner_tf_eeg(xeeg)
        xeog_sa = self.inner_tf_eog(xeog)

        xeeg_sa, xeog_sa = xeeg_sa[:, :, :1], xeog_sa[:, :, :1]

        xeeg_sa = self.outer_positional_embedding(xeeg_sa)
        xeog_sa = self.outer_positional_embedding(xeog_sa)

        xeeg_sa = self.outer_tf_inner_eeg(xeeg_sa)
        xeog_sa = self.outer_tf_inner_eog(xeog_sa)

        # xeeg_sa_p, xeog_sa_p = xeeg_sa[:, :, :1].squeeze(dim=4).squeeze(dim=3).squeeze(dim=2), xeog_sa[:, :, :1].squeeze(dim=4).squeeze(dim=3).squeeze(dim=2)

        x_match = torch.einsum('b o f , b m f -> b o m', xeeg_sa.squeeze(), xeog_sa.squeeze())

        x = torch.cat([xeeg_sa, xeeg_sa], dim=3)

        return x, x_match
class SleepEnc_BLIP_EEG_EOG_SingleMulti(nn.Module):
    def __init__(self, args):
        """
        :param encoder_filters_small, encoder_filters_big: filters of CNN output, quantized by 20
        :param encs_small, encs_big:
        """
        super().__init__()
        self.args = args

        d_model = args.dmodel  # 64*8

        inner_biased, outer_biased, rpos = False, False, False
        if "inner_biased" in args:
            inner_biased = args.inner_biased
        if "outer_biased" in args:
            outer_biased = args.outer_biased
        if "rpos" in args:
            rpos = args.rpos

        if inner_biased == "gaussian_learned":
            inner_biased = Gaussian_Learned_Attention_Bias(d_model, heads=8, type="mul")

        # if outer_biased == "gaussian_learned":
        #     outer_biased = Gaussian_Learned_Attention_Bias(d_model, heads=8, type="mul")

        self.inner_tf_eeg = inner_ch_SA_CA(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, gbiased=inner_biased, num_layers=4)
        self.inner_tf_eog = inner_ch_SA_CA(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, gbiased=inner_biased, num_layers=4)

        self.outer_tf = outer_mod_att_RA(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, gbiased=outer_biased, num_layers=4)

        self.cls_token = nn.Parameter(torch.randn(1, 1, 1, 1, 1, d_model))

        self.inner_positional_embedding = huy_pos_inner(d_model, pos=False, inner=29, outer=21, modalities=1,
                                                          channels=1)
        self.outer_positional_embedding = huy_pos_outer(d_model, pos=False, inner=29, outer=21, modalities=1,
                                                        channels=1)

    def forward(self, x, inits=None, extract_norm=False):
        xeeg = x[0][:, :, :, :, 1:, :]  # mat
        xeog = x[1][:, :, :, :, 1:, :]  # mat

        xeeg = einops.rearrange(xeeg, "b outer mod ch f inner -> b outer inner mod ch f")
        xeog = einops.rearrange(xeog, "b outer mod ch f inner -> b outer inner mod ch f")

        xeeg = self.inner_positional_embedding(xeeg)
        xeog = self.inner_positional_embedding(xeog)

        cls_token_eeg = self.cls_token.repeat(xeeg.shape[0], xeeg.shape[1], 1, 1, xeeg.shape[3], 1)
        xeeg = torch.cat([cls_token_eeg, xeeg], dim=2)

        cls_token_eog = self.cls_token.repeat(xeog.shape[0], xeog.shape[1], 1, 1, xeog.shape[3], 1)
        xeog = torch.cat([cls_token_eog, xeog], dim=2)

        xeeg_sa = self.inner_tf_eeg.forward_sa(xeeg)
        xeog_sa = self.inner_tf_eog.forward_sa(xeog)

        xeeg_ca = self.inner_tf_eeg.forward_ca(xeeg, xeog_sa)
        xeog_ca = self.inner_tf_eog.forward_ca(xeog, xeeg_sa)

        xeeg_sa_p, xeog_sa_p = xeeg_sa[:, :, :1].squeeze(dim=4).squeeze(dim=3).squeeze(dim=2), xeog_sa[:, :, :1].squeeze(dim=4).squeeze(dim=3).squeeze(dim=2)

        x_match = torch.einsum('b o f , b m f -> b o m', xeeg_sa_p, xeog_sa_p)

        x = torch.cat([xeeg_sa[:, :, :1] ,xeeg_ca[:, :, :1], xeog_sa[:, :, :1], xeog_ca[:, :, :1]], dim=3)
        x = self.outer_positional_embedding(x)
        x = self.outer_tf(x, extract_norm=extract_norm)

        return x, x_match

class SleepEnc_BLIP_outer_EEG_EOG(nn.Module):
    def __init__(self, args):
        """
        :param encoder_filters_small, encoder_filters_big: filters of CNN output, quantized by 20
        :param encs_small, encs_big:
        """
        super().__init__()
        self.args = args

        d_model = args.dmodel  # 64*8

        inner_biased, outer_biased, rpos = False, False, False
        if "inner_biased" in args:
            inner_biased = args.inner_biased
        if "outer_biased" in args:
            outer_biased = args.outer_biased
        if "rpos" in args:
            rpos = args.rpos

        if inner_biased == "gaussian_learned":
            inner_biased = Gaussian_Learned_Attention_Bias(d_model, heads=8, type="mul")

        # if outer_biased == "gaussian_learned":
        #     outer_biased = Gaussian_Learned_Attention_Bias(d_model, heads=8, type="mul")

        self.inner_tf_eeg = inner_ch_SA_CA(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, gbiased=inner_biased, num_layers=4)
        self.inner_tf_eog = inner_ch_SA_CA(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, gbiased=inner_biased, num_layers=4)

        self.outer_tf_eeg = outer_mod_att_RA(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, gbiased=outer_biased, num_layers=4)
        self.outer_tf_eog = outer_mod_att_RA(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, gbiased=outer_biased, num_layers=4)

        self.outer_tf = outer_mod_att_RA(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, gbiased=outer_biased, num_layers=4)

        self.cls_token = nn.Parameter(torch.randn(1, 1, 1, 1, 1, d_model))

        self.inner_positional_embedding = huy_pos_inner(d_model, pos=False, inner=29, outer=21, modalities=1,
                                                          channels=1)
        self.outer_positional_embedding = huy_pos_outer(d_model, pos=False, inner=29, outer=21, modalities=1,
                                                        channels=1)

    def forward(self, x, inits=None, extract_norm=False, return_reps=False):
        xeeg = x[0][:, :, :, :, 1:, :]  # mat
        xeog = x[1][:, :, :, :, 1:, :]  # mat

        xeeg = einops.rearrange(xeeg, "b outer mod ch f inner -> b outer inner mod ch f")
        xeog = einops.rearrange(xeog, "b outer mod ch f inner -> b outer inner mod ch f")

        cls_token_eeg = self.cls_token.repeat(xeeg.shape[0], xeeg.shape[1], 1, 1, xeeg.shape[3], 1)
        xeeg = torch.cat([cls_token_eeg, xeeg], dim=2)

        cls_token_eog = self.cls_token.repeat(xeog.shape[0], xeog.shape[1], 1, 1, xeog.shape[3], 1)
        xeog = torch.cat([cls_token_eog, xeog], dim=2)

        xeeg_sa = self.inner_tf_eeg.forward_sa(xeeg)
        xeog_sa = self.inner_tf_eog.forward_sa(xeog)

        xeeg_sa = self.outer_positional_embedding(xeeg_sa)
        xeog_sa = self.outer_positional_embedding(xeog_sa)
        xeeg_sa = self.outer_tf_eeg(xeeg_sa, extract_norm=extract_norm)
        xeog_sa = self.outer_tf_eog(xeog_sa, extract_norm=extract_norm)

        xeeg = self.inner_positional_embedding(xeeg)
        xeog = self.inner_positional_embedding(xeog)

        xeeg_ca = self.inner_tf_eeg.forward_ca(xeeg, xeog_sa)
        xeog_ca = self.inner_tf_eog.forward_ca(xeog, xeeg_sa)

        xeeg_sa, xeog_sa = xeeg_sa[:, :, :1].squeeze(dim=4).squeeze(dim=3).squeeze(dim=2), xeog_sa[:, :, :1].squeeze(dim=4).squeeze(dim=3).squeeze(dim=2)

        x_match = torch.einsum('b o f , b m f -> b o m', xeeg_sa, xeog_sa)

        x = torch.cat([xeeg_ca[:, :, :1], xeog_ca[:, :, :1]], dim=3)
        x = self.outer_positional_embedding(x)
        x = self.outer_tf(x, extract_norm=extract_norm)
        if return_reps:
            return x, x_match, [xeeg_sa, xeog_sa]
        return x, x_match

class SleepEnc_Router_COCA_EEG_EOG(nn.Module):
    def __init__(self, args):
        """
        :param encoder_filters_small, encoder_filters_big: filters of CNN output, quantized by 20
        :param encs_small, encs_big:
        """
        super().__init__()
        self.args = args

        d_model = args.dmodel  # 64*8

        inner_biased, outer_biased, rpos = False, False, False
        if "inner_biased" in args:
            inner_biased = args.inner_biased
        if "outer_biased" in args:
            outer_biased = args.outer_biased
        if "rpos" in args:
            rpos = args.rpos

        if inner_biased == "gaussian_learned":
            inner_biased = Gaussian_Learned_Attention_Bias(d_model, heads=8, type="mul")

        # if outer_biased == "gaussian_learned":
        #     outer_biased = Gaussian_Learned_Attention_Bias(d_model, heads=8, type="mul")

        self.inner_tf_eeg = inner_ch_SA_CA(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, gbiased=inner_biased, num_layers=2)
        self.inner_tf_eog = inner_ch_SA_CA(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, gbiased=inner_biased, num_layers=2)

        self.outer_tf = outer_mod_att_RA(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, gbiased=outer_biased, num_layers=2)

        self.cls_token = nn.Parameter(torch.randn(1, 1, 1, 1, 1, d_model))
        self.cls_token_outer = nn.Parameter(torch.randn(1, 1, 1, 1, 1, d_model))

        self.inner_positional_embedding = huy_pos_inner(d_model, pos=False, inner=29, outer=21, modalities=1,
                                                          channels=1)
        self.outer_positional_embedding = huy_pos_outer(d_model, pos=False, inner=29, outer=21, modalities=1,
                                                        channels=1)

    def forward(self, x, inits=None, extract_norm=False, return_matches=False, return_inter_reps=False, return_order=False):
        xeeg = x[0][:, :, :, :, 1:, :]  # mat
        xeog = x[1][:, :, :, :, 1:, :]  # mat

        xeeg = einops.rearrange(xeeg, "b outer mod ch f inner -> b outer inner mod ch f")
        xeog = einops.rearrange(xeog, "b outer mod ch f inner -> b outer inner mod ch f")

        cls_token_eeg = self.cls_token.repeat(xeeg.shape[0], xeeg.shape[1], 1, 1, xeeg.shape[3], 1)
        xeeg = torch.cat([cls_token_eeg, xeeg], dim=2)

        cls_token_eog = self.cls_token.repeat(xeog.shape[0], xeog.shape[1], 1, 1, xeog.shape[3], 1)
        xeog = torch.cat([cls_token_eog, xeog], dim=2)

        xeeg_sa = self.inner_tf_eeg.forward_sa(xeeg)
        xeog_sa = self.inner_tf_eog.forward_sa(xeog)

        xeeg_ca = self.inner_tf_eeg.forward_ca(xeeg_sa, xeog_sa)
        xeog_ca = self.inner_tf_eog.forward_ca(xeog_sa, xeeg_sa)

        x = torch.cat([xeeg_ca[:, :, :1], xeog_ca[:, :, :1]], dim=1)
        cls_token_outer = self.cls_token_outer.repeat(x.shape[0], x.shape[1], 1, 1, 1, 1)
        x = torch.cat([cls_token_outer, x], dim=1)
        x = self.outer_positional_embedding(x)
        x = self.outer_tf(x)

        return {"preds":{"combined":x[:,:1]}}

class VanillaVAE(nn.Module):

    def __init__(self, encs=None, args={}) -> None:
        super(VanillaVAE, self).__init__()

        in_channels = args.in_channels if "latent_dim" in args else 1
        latent_dim = args.latent_dim if "latent_dim" in args else 128
        hidden_dims = args.hidden_dims if "hidden_dims" in args else None

        self.latent_dim = latent_dim

        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 128, 32]

        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size= 3, stride= 2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU()))

            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1]*20, latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1]*20, latent_dim)

        # Build Decoder
        modules = []

        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * 20)

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            output_pad = (1,1) if i==0 else (0,1)
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride = 2,
                                       padding=1,
                                       output_padding=output_pad),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU()
                )
            )


        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
                            nn.ConvTranspose2d(hidden_dims[-1],
                                               1,
                                               kernel_size=3, padding=1, output_padding=(0,1),
                                               stride=2),
                            # nn.BatchNorm2d(hidden_dims[-1]),
                            # nn.LeakyReLU(),
                            # nn.Conv2d(hidden_dims[-1], out_channels= 1,
                            #           kernel_size= 3, padding=(0,1)),
                            nn.Tanh())

    def encode(self, input: Tensor):
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        result = self.decoder_input(z)

        result = result.view(-1, 32, 10, 2)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: Tensor, **kwargs):
        input = input["stft_eeg"][:, :, :, :, 1:, :]  # mat
        [b, outer, mod, ch, f, inner] = input.shape
        input_m = einops.rearrange(input, "b outer mod ch f inner -> b mod (outer inner) (ch f)")
        mu, log_var = self.encode(input_m)
        z = self.reparameterize(mu, log_var)
        output = self.decode(z)
        output = einops.rearrange(output, "b mod (outer inner) (ch f) -> b outer mod ch f inner", outer=outer, inner=inner, ch=ch, f=f)
        return  [output, input, mu, log_var]

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]

        reduction = kwargs["reduction"] if "reduction" in kwargs else "mean"
        kld_weight = kwargs["kld_weight"] if "kld_weight" in kwargs else 0.0005

        recons_loss = F.mse_loss(recons, input, reduction=reduction)

        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        loss = recons_loss + kld_weight * kld_loss

        output_loss = {'total': loss, 'reconstruction_Loss': recons_loss.detach().cpu().numpy()}
        if kld_weight!=0: output_loss.update({"kld_loss": kld_loss.detach().cpu().numpy()})

        return output_loss

    def sample(self,
               num_samples:int,
               current_device: int, **kwargs) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]
class VanillaVAE_eog(nn.Module):

    def __init__(self, encs=None, args={}) -> None:
        super(VanillaVAE_eog, self).__init__()

        in_channels = args.in_channels if "latent_dim" in args else 1
        latent_dim = args.latent_dim if "latent_dim" in args else 128
        hidden_dims = args.hidden_dims if "hidden_dims" in args else None

        self.latent_dim = latent_dim

        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 128, 32]

        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size= 3, stride= 2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU()))

            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1]*20, latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1]*20, latent_dim)

        # Build Decoder
        modules = []

        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * 20)

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            output_pad = (1,1) if i==0 else (0,1)
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride = 2,
                                       padding=1,
                                       output_padding=output_pad),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU()
                )
            )


        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
                            nn.ConvTranspose2d(hidden_dims[-1],
                                               1,
                                               kernel_size=3, padding=1, output_padding=(0,1),
                                               stride=2),
                            # nn.BatchNorm2d(hidden_dims[-1]),
                            # nn.LeakyReLU(),
                            # nn.Conv2d(hidden_dims[-1], out_channels= 1,
                            #           kernel_size= 3, padding=(0,1)),
                            nn.Tanh())

    def encode(self, input: Tensor):
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        result = self.decoder_input(z)

        result = result.view(-1, 32, 10, 2)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: Tensor, **kwargs):
        input = input["stft_eog"][:, :, :, :, 1:, :]  # mat
        [b, outer, mod, ch, f, inner] = input.shape
        input_m = einops.rearrange(input, "b outer mod ch f inner -> b mod (outer inner) (ch f)")
        mu, log_var = self.encode(input_m)
        z = self.reparameterize(mu, log_var)
        output = self.decode(z)
        output = einops.rearrange(output, "b mod (outer inner) (ch f) -> b outer mod ch f inner", outer=outer, inner=inner, ch=ch, f=f)
        return  [output, input, mu, log_var]

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]

        reduction = kwargs["reduction"] if "reduction" in kwargs else "mean"
        kld_weight = kwargs["kld_weight"] if "kld_weight" in kwargs else 0.0005

        recons_loss = F.mse_loss(recons, input, reduction=reduction)

        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        loss = recons_loss + kld_weight * kld_loss

        output_loss = {'total': loss, 'reconstruction_Loss': recons_loss.detach().cpu().numpy()}
        if kld_weight!=0: output_loss.update({"kld_loss": kld_loss.detach().cpu().numpy()})

        return output_loss

    def sample(self,
               num_samples:int,
               current_device: int, **kwargs) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]

class My_TF_RA_ADJ(nn.Module):
    def __init__(self, d_model, nhead, gbiased=False, predefined_SA=False, predefined_CA=False, predefined_CA_CA=False, predefined_FC=False, extra_attention = False, rpos=False, modalities=1, dim_feedforward=1024, dim_proj= 128, dropout=0.1, activation="relu"):
        # super(nn.TransformerEncoderLayer, self).__init__()
        super().__init__()

        self.extra_attention = extra_attention
        if self.extra_attention:
            self.extra_self_attn = My_MultiHeadAttention(d_model, nhead, dim_proj=128, activation=None, gbiased=self.extra_attention)
            self.extra_norm = nn.LayerNorm(d_model)
            self.extra_dropout = nn.Dropout(dropout)

        if predefined_SA:
            self.self_attn = predefined_SA
        else:
            self.self_attn = My_MultiHeadAttention(d_model,  nhead, dim_proj=dim_proj, rpos=rpos, activation=None, gbiased = gbiased)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        self.cross_att_flag_0 = False
        if predefined_CA:
            self.cross_att = predefined_CA
            self.norm2 = nn.LayerNorm(d_model)
            self.dropout2 = nn.Dropout(dropout)
            self.cross_att_flag_0 = True
            # self.fc_CA = Positionwise_FC(d_model=d_model)

        self.cross_att_flag_1 = False
        if predefined_CA_CA:
            self.cross_att_1 = predefined_CA_CA
            self.norm2_1 = nn.LayerNorm(d_model)
            self.dropout2_1 = nn.Dropout(dropout)
            self.cross_att_flag_1 = True
            # self.fc_CA_CA = Positionwise_FC(d_model=d_model)


        if predefined_FC:
            self.fc_SA = predefined_FC
        else:
            self.fc_SA = Positionwise_FC(d_model=d_model)

        self.activation = nn.ReLU()
        self.norm_calc = BertNormOutput(num_attention_heads=nhead, hidden_size=d_model)
        self.nhead = nhead


    def forward(self, src: Tensor, crossatt_src: Optional[Tensor] = None, crossatt_src_1: Optional[Tensor] = None, src_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None, extract_norm = False) -> Tensor:
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
            src_extr_att, att_0, value_0, linear_o_0 = self.extra_self_attn(src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)
            src_extr_att = src + self.extra_dropout(src_extr_att)
            src_extr_att = self.extra_norm(src_extr_att)
        else:
            src_extr_att = src

        src_att, att, value, linear_o = self.self_attn(src_extr_att, src_extr_att, src_extr_att, attn_mask=src_mask,  key_padding_mask=src_key_padding_mask)
        src_att = self.norm1(src_extr_att + self.dropout1(src_att))

        if self.cross_att_flag_0 and crossatt_src is not None:
            src_ca, att, value, linear_o = self.cross_att(src_att, src_att, crossatt_src, attn_mask=src_mask,  key_padding_mask=src_key_padding_mask)
            src_att = self.norm2(src_att + self.dropout2(src_ca))
            # src_att = self.fc_CA(src_att)

        if self.cross_att_flag_1 and crossatt_src_1 is not None:
            src_ca, att, value, linear_o = self.cross_att_1(src_att, src_att, crossatt_src_1, attn_mask=src_mask,  key_padding_mask=src_key_padding_mask)
            src_att = self.norm2_1(src_att + self.dropout2_1(src_ca))
            # src_att = self.fc_CA_CA(src_att)

        src_att = self.fc_SA(src_att)

        # src_fc = self.mod_0_linear2(self.mod_0_dropout(self.activation(self.mod_0_linear1(src_att))))
        # src_att = self.mod_0_norm2(src_att + self.mod_0_dropout2(src_fc))

        return src_att
class My_TF_RA_ADJ_SA_CA(nn.Module):
    def __init__(self, d_model, nhead=8, gbiased=False,   SA={"shared":False, "use":True,"dim_proj":128,"dropout":0.1,"rpos":False, "gbiased":False},
                                                        CA={"shared":False, "use":False,"dim_proj":128,"dropout":0.1,"rpos":False, "gbiased":False},
                                                        CA_CA={"shared":False, "use":False,"dim_proj":128, "dropout":0.1,"rpos":False, "gbiased":False},
                                                        FC={"shared":False, "use":False,"dim_feedforward":1024, "dropout":0.1,"activation":"relu"}):
        # super(nn.TransformerEncoderLayer, self).__init__()
        super().__init__()

        # self.extra_attention = extra_attention
        # if self.extra_attention:
        #     self.extra_self_attn = My_MultiHeadAttention(d_model, nhead, dim_proj=128, activation=None, gbiased=self.extra_attention)
        #     self.extra_norm = nn.LayerNorm(d_model)
        #     self.extra_dropout = nn.Dropout(dropout)


        self.sa_use = SA["use"]
        self.sa_shared = SA["shared"]

        if SA["use"] and SA["shared"]:
            self.sa_att = My_MultiHeadAttention(d_model, head_num=8, dim_proj= SA["dim_proj"], rpos=SA["rpos"], activation=None, gbiased=SA["gbiased"])
            self.sa_norm = nn.LayerNorm(d_model)
            self.sa_drop = nn.Dropout(SA["dropout"])
        
        if not self.sa_shared:
            self.sa_att_s2 = My_MultiHeadAttention(d_model, head_num=8, dim_proj=SA["dim_proj"], rpos=SA["rpos"],
                                            activation=None, gbiased=SA["gbiased"])
            self.sa_norm_s2 = nn.LayerNorm(d_model)
            self.sa_drop_s2 = nn.Dropout(SA["dropout"])


        self.ca_use = CA["use"]
        self.ca_shared = CA["shared"]

        if CA["use"] and CA["shared"]:
            self.ca_att = My_MultiHeadAttention(d_model, head_num=8, dim_proj=CA["dim_proj"], rpos=CA["rpos"], activation=None,
                                                gbiased=CA["gbiased"])
            self.ca_norm = nn.LayerNorm(d_model)
            self.ca_drop = nn.Dropout(CA["dropout"])

        if not self.ca_shared:
            self.ca_att_s2= My_MultiHeadAttention(d_model, head_num=8, dim_proj=CA["dim_proj"], rpos=CA["rpos"], activation=None,
                                                gbiased=CA["gbiased"])
            self.ca_norm_s2 = nn.LayerNorm(d_model)
            self.ca_drop_s2 = nn.Dropout(CA["dropout"])


        self.ca_ca_use = CA_CA["use"]
        self.ca_ca_shared = CA_CA["shared"]

        if CA_CA["use"] and CA_CA["shared"]:
            self.ca_ca_att = My_MultiHeadAttention(d_model, head_num=8, dim_proj=CA_CA["dim_proj"], rpos=CA_CA["rpos"], activation=None,
                                                gbiased=CA_CA["gbiased"])
            self.ca_ca_norm = nn.LayerNorm(d_model)
            self.ca_ca_drop = nn.Dropout(CA_CA["dropout"])

        if not self.ca_ca_shared:
            self.ca_ca_att_s2 = My_MultiHeadAttention(d_model, head_num=8, dim_proj=CA_CA["dim_proj"], rpos=CA_CA["rpos"],
                                                   activation=None,
                                                   gbiased=CA_CA["gbiased"])
            self.ca_ca_norm_s2 = nn.LayerNorm(d_model)
            self.ca_ca_drop_s2 = nn.Dropout(CA_CA["dropout"])


        self.fc_use = FC["use"]
        self.fc_shared = FC["shared"]

        if FC["use"] and FC["shared"]:
            self.fc_linear1 = nn.Linear(d_model, FC["dim_feedforward"])
            self.fc_dropout1 = nn.Dropout(FC["dropout"])
            self.fc_linear2 = nn.Linear(FC["dim_feedforward"], d_model)
            self.fc_norm = nn.LayerNorm(d_model)
            self.fc_dropout2 = nn.Dropout(FC["dropout"])

        if not self.fc_shared:

            self.fc_linear1_s2 = nn.Linear(d_model, FC["dim_feedforward"])
            self.fc_dropout1_s2 = nn.Dropout(FC["dropout"])
            self.fc_linear2_s2 = nn.Linear(FC["dim_feedforward"], d_model)
            self.fc_norm_s2 = nn.LayerNorm(d_model)
            self.fc_dropout2_s2 = nn.Dropout(FC["dropout"])
        self.activation = nn.ReLU()


    def forward(self, src_att: Tensor, crossatt_src: Optional[Tensor] = None, crossatt_src_1: Optional[Tensor] = None, src_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None, extract_norm = False) -> Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        x_shape = src_att.shape

        if self.sa_use and src_att is not None:

            if self.sa_shared:
                src_sa, att, value, linear_o = self.sa_att(src_att, src_att, src_att, attn_mask=src_mask,  key_padding_mask=src_key_padding_mask)
                src_att = self.sa_norm(src_att + self.sa_drop(src_sa))
            else:
                src_sa, att, value, linear_o = self.sa_att_s2(src_att, src_att, src_att, attn_mask=src_mask,  key_padding_mask=src_key_padding_mask)
                src_att = self.ca_norm_s2(src_att + self.sa_drop_s2(src_sa))

        if self.ca_use and crossatt_src is not None:
            if self.ca_shared:
                src_ca, att, value, linear_o = self.ca_att(src_att, src_att, crossatt_src, attn_mask=src_mask,  key_padding_mask=src_key_padding_mask)
                src_att = self.ca_norm(src_att + self.ca_drop(src_ca))
            else:
                src_ca, att, value, linear_o = self.ca_att_s2(src_att, src_att, crossatt_src, attn_mask=src_mask,  key_padding_mask=src_key_padding_mask)
                src_att = self.ca_norm_s2(src_att + self.ca_drop_s2(src_ca))

        if self.ca_ca_use and crossatt_src_1 is not None:
            if self.ca_ca_shared:
                src_ca, att, value, linear_o = self.ca_ca_att(src_att, src_att, crossatt_src, attn_mask=src_mask,  key_padding_mask=src_key_padding_mask)
                src_att = self.ca_ca_norm(src_att + self.ca_ca_drop(src_ca))
            else:
                src_ca, att, value, linear_o = self.ca_ca_att_s2(src_att, src_att, crossatt_src, attn_mask=src_mask,  key_padding_mask=src_key_padding_mask)
                src_att = self.ca_ca_norm_s2(src_att + self.ca_ca_drop_s2(src_ca))

        if self.fc_use:
            if self.fc_shared:
                src_fc = self.fc_linear2(self.fc_dropout1(self.activation(self.fc_linear1(src_att))))
                src_att = self.fc_norm(src_att + self.fc_dropout2(src_fc))
            else:
                src_fc = self.fc_linear2_s2(self.fc_dropout1_s2(self.activation(self.fc_linear1_s2(src_att))))
                src_att = self.fc_norm_s2(src_att + self.fc_dropout2_s2(src_fc))

        # src_fc = self.mod_0_linear2(self.mod_0_dropout(self.activation(self.mod_0_linear1(src_att))))
        # src_att = self.mod_0_norm2(src_att + self.mod_0_dropout2(src_fc))

        return src_att
class Positionwise_FC(nn.Module):
    def __init__(self, d_model, dim_feedforward=1024, dropout=0.1, activation="relu"):
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
class inner_ch_SA_CA(nn.Module):
    def __init__(self, d_model, pos, inner, outer, modalities, gbiased=False, extra_attention=False, rpos=False, num_layers=1, dim_proj=128, heads=8, dim_feedforward=1024):
        super().__init__()
        self.pos = pos
        if pos:
            self.pos_inner = PositionalEncoder(d_model=d_model)


        predefined_SA = My_MultiHeadAttention(d_model,  head_num=8, dim_proj=dim_proj, rpos=rpos, activation=None, gbiased = gbiased)
        predefined_CA = My_MultiHeadAttention(d_model,  head_num=8, dim_proj=dim_proj, rpos=rpos, activation=None, gbiased = gbiased)
        predefined_FC = Positionwise_FC(d_model=d_model)

        # self.inner_tf_sa = My_TF_RA_ADJ(d_model, extra_attention=extra_attention, predefined_SA=predefined_SA, predefined_CA=False, predefined_FC=predefined_FC, nhead=heads, rpos=rpos, dim_proj=dim_proj, dim_feedforward=dim_feedforward, gbiased=gbiased)
        # self.inner_tf_ca = My_TF_RA_ADJ(d_model, extra_attention=extra_attention, predefined_SA=predefined_SA, predefined_CA=predefined_CA, predefined_FC=predefined_FC, nhead=heads, rpos=rpos, dim_proj=dim_proj, dim_feedforward=dim_feedforward, gbiased=gbiased)

        enc_sa = My_TF_RA_ADJ(d_model, extra_attention=extra_attention, predefined_SA=predefined_SA, predefined_CA=False, predefined_FC=predefined_FC, nhead=heads, rpos=rpos, dim_proj=dim_proj, dim_feedforward=dim_feedforward, gbiased=gbiased)
        enc_ca = My_TF_RA_ADJ(d_model, extra_attention=extra_attention, predefined_SA=predefined_SA, predefined_CA=predefined_CA, predefined_FC=predefined_FC, nhead=heads, rpos=rpos, dim_proj=dim_proj, dim_feedforward=dim_feedforward, gbiased=gbiased)

        self.inner_tf_sa = My_TransformerEncoder(enc_sa, num_layers)
        self.inner_tf_ca = My_TransformerEncoder_CA(enc_ca, num_layers)

    def forward(self, x, extract_norm=False):
        return self.forward_sa(x)

    def forward_sa(self, x, extract_norm=False):
        x_shape = x.shape
        self.batch, self.outer, self.inner, self.mod, self.ch, self.features = x_shape[0], x_shape[1], x_shape[2], x_shape[3], x_shape[4], x_shape[5]

        x = einops.rearrange(x, "b outer inner mod ch k -> (inner mod ch) (b outer) k")
        x = self.inner_tf_sa(x, extract_norm=extract_norm)
        x = einops.rearrange(x, " (inner mod ch) (b outer) k -> b outer inner mod ch k", outer=self.outer, mod=self.mod, ch=self.ch,  b=self.batch)
        return x

    def forward_ca(self, x, x_ca = None, extract_norm=False):
        x_shape = x.shape
        self.batch, self.outer, self.inner, self.mod, self.ch, self.features = x_shape[0], x_shape[1], x_shape[2], x_shape[3], x_shape[4], x_shape[5]

        x = einops.rearrange(x, "b outer inner mod ch k -> (inner mod ch) (b outer) k")
        if x_ca is not None:
            x_ca = einops.rearrange(x_ca, "b outer inner mod ch k -> (inner mod ch) (b outer) k")
        x = self.inner_tf_ca(x, src_ca = x_ca, extract_norm=extract_norm)
        x = einops.rearrange(x, " (inner mod ch) (b outer) k -> b outer inner mod ch k", outer=self.outer, mod=self.mod, ch=self.ch,  b=self.batch)
        return x
class inner_ch_SA_CA_shared_v2(nn.Module):
    def __init__(self, d_model, pos, inner, outer, modalities, gbiased=False, extra_attention=False, rpos=False, num_layers=1, dim_proj=128, heads=8, dim_feedforward=1024):
        super().__init__()
        self.pos = pos
        if pos:
            self.pos_inner = PositionalEncoder(d_model=d_model)

        predefined_SA = My_MultiHeadAttention(d_model,  head_num=8, dim_proj=dim_proj, rpos=rpos, activation=None, gbiased = gbiased)
        predefined_CA = My_MultiHeadAttention(d_model,  head_num=8, dim_proj=dim_proj, rpos=rpos, activation=None, gbiased = gbiased)
        predefined_FC = Positionwise_FC(d_model=d_model)

        enc = My_TF_RA_ADJ(d_model, extra_attention=extra_attention, predefined_SA=predefined_SA, predefined_CA=predefined_CA, predefined_FC=predefined_FC, nhead=heads, rpos=rpos, dim_proj=dim_proj, dim_feedforward=dim_feedforward, gbiased=gbiased)
        self.inner_tf = My_TransformerEncoder(enc, num_layers=num_layers)

    def forward(self, x, extract_norm=False):
        return self.forward_sa(x)

    def forward_sa(self, x, extract_norm=False):
        x_shape = x.shape
        self.batch, self.outer, self.inner, self.mod, self.ch, self.features = x_shape[0], x_shape[1], x_shape[2], x_shape[3], x_shape[4], x_shape[5]

        x = einops.rearrange(x, "b outer inner mod ch k -> (inner mod ch) (b outer) k")
        x = self.inner_tf(x, extract_norm=extract_norm)
        x = einops.rearrange(x, " (inner mod ch) (b outer) k -> b outer inner mod ch k", outer=self.outer, mod=self.mod, ch=self.ch,  b=self.batch)
        return x

    def forward_ca(self, x, x_ca = None, extract_norm=False):
        x_shape = x.shape
        self.batch, self.outer, self.inner, self.mod, self.ch, self.features = x_shape[0], x_shape[1], x_shape[2], x_shape[3], x_shape[4], x_shape[5]

        x = einops.rearrange(x, "b outer inner mod ch k -> (inner mod ch) (b outer) k")
        if x_ca is not None:
            x_ca = einops.rearrange(x_ca, "b outer inner mod ch k -> (inner mod ch) (b outer) k")
        x = self.inner_tf(x, src_ca = x_ca, extract_norm=extract_norm)
        x = einops.rearrange(x, " (inner mod ch) (b outer) k -> b outer inner mod ch k", outer=self.outer, mod=self.mod, ch=self.ch,  b=self.batch)
        return x
class inner_ch_SA_CA_CA_shared_v2(nn.Module):
    def __init__(self, d_model, pos, inner, outer, modalities, gbiased=False, extra_attention=False, rpos=False, num_layers=1, dim_proj=128, heads=8, dim_feedforward=1024):
        super().__init__()
        self.pos = pos
        if pos:
            self.pos_inner = PositionalEncoder(d_model=d_model)

        predefined_SA = My_MultiHeadAttention(d_model,  head_num=8, dim_proj=dim_proj, rpos=rpos, activation=None, gbiased = gbiased)
        predefined_CA = My_MultiHeadAttention(d_model,  head_num=8, dim_proj=dim_proj, rpos=rpos, activation=None, gbiased = gbiased)
        predefined_CA_CA = My_MultiHeadAttention(d_model,  head_num=8, dim_proj=dim_proj, rpos=rpos, activation=None, gbiased = gbiased)
        predefined_FC = Positionwise_FC(d_model=d_model)

        enc = My_TF_RA_ADJ(d_model, extra_attention=extra_attention, predefined_SA=predefined_SA, predefined_CA=predefined_CA, predefined_CA_CA=predefined_CA_CA, predefined_FC=predefined_FC, nhead=heads, rpos=rpos, dim_proj=dim_proj, dim_feedforward=dim_feedforward, gbiased=gbiased)
        self.inner_tf = My_TransformerEncoder(enc, num_layers=num_layers)

    def forward(self, x, extract_norm=False):
        return self.forward_sa(x)

    def forward_sa(self, x, extract_norm=False):
        x_shape = x.shape
        self.batch, self.outer, self.inner, self.mod, self.ch, self.features = x_shape[0], x_shape[1], x_shape[2], x_shape[3], x_shape[4], x_shape[5]

        x = einops.rearrange(x, "b outer inner mod ch k -> (inner mod ch) (b outer) k")
        x = self.inner_tf(x, extract_norm=extract_norm)
        x = einops.rearrange(x, " (inner mod ch) (b outer) k -> b outer inner mod ch k", outer=self.outer, mod=self.mod, ch=self.ch,  b=self.batch)
        return x

    def forward_ca(self, x, x_ca = None,  x_ca_ca = None , extract_norm=False):
        x_shape = x.shape
        self.batch, self.outer, self.inner, self.mod, self.ch, self.features = x_shape[0], x_shape[1], x_shape[2], x_shape[3], x_shape[4], x_shape[5]

        x = einops.rearrange(x, "b outer inner mod ch k -> (inner mod ch) (b outer) k")
        if x_ca is not None:
            x_ca = einops.rearrange(x_ca, "b outer inner mod ch k -> (inner mod ch) (b outer) k")
        if x_ca_ca is not None:
            x_ca_ca = einops.rearrange(x_ca_ca, "b outer inner mod ch k -> (inner mod ch) (b outer) k")
        x = self.inner_tf(x, src_ca = x_ca, src_ca_ca=x_ca_ca, extract_norm=extract_norm)
        x = einops.rearrange(x, " (inner mod ch) (b outer) k -> b outer inner mod ch k", outer=self.outer, mod=self.mod, ch=self.ch,  b=self.batch)
        return x
class inner_ch_SA_CA_shared_v2_fullca(nn.Module):
    def __init__(self, d_model, pos, inner, outer, modalities, gbiased=False, extra_attention=False, rpos=False, num_layers=1, dim_proj=128, heads=8, dim_feedforward=1024):
        super().__init__()
        self.pos = pos
        if pos:
            self.pos_inner = PositionalEncoder(d_model=d_model)

        predefined_SA = My_MultiHeadAttention(d_model,  head_num=8, dim_proj=dim_proj, rpos=rpos, activation=None, gbiased = gbiased)
        predefined_CA = My_MultiHeadAttention(d_model,  head_num=8, dim_proj=dim_proj, rpos=rpos, activation=None, gbiased = gbiased)
        predefined_FC = Positionwise_FC(d_model=d_model)

        enc = My_TF_RA_ADJ(d_model, extra_attention=extra_attention, predefined_SA=predefined_SA, predefined_CA=predefined_CA, predefined_FC=predefined_FC, nhead=heads, rpos=rpos, dim_proj=dim_proj, dim_feedforward=dim_feedforward, gbiased=gbiased)
        self.inner_tf = My_TransformerEncoder(enc, num_layers=num_layers)

    def forward(self, x, extract_norm=False):
        return self.forward_sa(x)

    def forward_sa(self, x, extract_norm=False, return_layer="last"):
        x_shape = x.shape
        self.batch, self.outer, self.inner, self.mod, self.ch, self.features = x_shape[0], x_shape[1], x_shape[2], x_shape[3], x_shape[4], x_shape[5]

        x = einops.rearrange(x, "b outer inner mod ch k -> (inner mod ch) (b outer) k")
        x = self.inner_tf(x, extract_norm=extract_norm, return_layer=return_layer)
        x = einops.rearrange(x, "layer (inner mod ch) (b outer) k -> layer b outer inner mod ch k", outer=self.outer, mod=self.mod, ch=self.ch,  b=self.batch)
        return x

    def forward_ca(self, x, x_ca = None, extract_norm=False):
        x_shape = x.shape
        self.batch, self.outer, self.inner, self.mod, self.ch, self.features = x_shape[0], x_shape[1], x_shape[2], x_shape[3], x_shape[4], x_shape[5]

        x = einops.rearrange(x, "b outer inner mod ch k -> (inner mod ch) (b outer) k")
        if x_ca is not None:
            x_ca = einops.rearrange(x_ca, "layer b outer inner mod ch k ->layer (inner mod ch) (b outer) k")
        x = self.inner_tf(x, src_ca = x_ca, extract_norm=extract_norm, ca_type="full")
        x = einops.rearrange(x, " (inner mod ch) (b outer) k -> b outer inner mod ch k", outer=self.outer, mod=self.mod, ch=self.ch,  b=self.batch)
        return x

class outer_mod_ch_SA_CA(nn.Module):
    def __init__(self, d_model, pos, inner, outer, modalities, gbiased=False, extra_attention=False, rpos=False, num_layers=1, dim_proj=128, heads=8, dim_feedforward=1024):
        super().__init__()
        self.pos = pos
        if pos:
            self.pos_inner = PositionalEncoder(d_model=d_model)


        predefined_SA = My_MultiHeadAttention(d_model,  head_num=8, dim_proj=dim_proj, rpos=rpos, activation=None, gbiased = gbiased)
        predefined_CA = My_MultiHeadAttention(d_model,  head_num=8, dim_proj=dim_proj, rpos=rpos, activation=None, gbiased = gbiased)
        predefined_FC = Positionwise_FC(d_model=d_model)

        # self.inner_tf_sa = My_TF_RA_ADJ(d_model, extra_attention=extra_attention, predefined_SA=predefined_SA, predefined_CA=False, predefined_FC=predefined_FC, nhead=heads, rpos=rpos, dim_proj=dim_proj, dim_feedforward=dim_feedforward, gbiased=gbiased)
        # self.inner_tf_ca = My_TF_RA_ADJ(d_model, extra_attention=extra_attention, predefined_SA=predefined_SA, predefined_CA=predefined_CA, predefined_FC=predefined_FC, nhead=heads, rpos=rpos, dim_proj=dim_proj, dim_feedforward=dim_feedforward, gbiased=gbiased)

        enc_sa = My_TF_RA_ADJ(d_model, extra_attention=extra_attention, predefined_SA=predefined_SA, predefined_CA=False, predefined_FC=predefined_FC, nhead=heads, rpos=rpos, dim_proj=dim_proj, dim_feedforward=dim_feedforward, gbiased=gbiased)
        enc_ca = My_TF_RA_ADJ(d_model, extra_attention=extra_attention, predefined_SA=predefined_SA, predefined_CA=predefined_CA, predefined_FC=predefined_FC, nhead=heads, rpos=rpos, dim_proj=dim_proj, dim_feedforward=dim_feedforward, gbiased=gbiased)

        self.inner_tf_sa = My_TransformerEncoder(enc_sa, num_layers)
        self.inner_tf_ca = My_TransformerEncoder_CA(enc_ca, num_layers)

    def forward(self, x, extract_norm=False):
        return self.forward_sa(x)

    def forward_sa(self, x, extract_norm=False):
        x_shape = x.shape
        self.batch, self.outer, self.inner, self.mod, self.ch, self.features = x_shape[0], x_shape[1], x_shape[2], x_shape[3], x_shape[4], x_shape[5]

        x = einops.rearrange(x, "b outer inner mod ch k -> (outer mod) (b inner ch) k")
        x = self.inner_tf_sa(x, extract_norm=extract_norm)
        x = einops.rearrange(x, " (outer mod) (b inner ch) k -> b outer inner mod ch k", outer=self.outer, mod=self.mod, ch=self.ch,  b=self.batch)
        return x

    def forward_ca(self, x, x_ca = None, extract_norm=False):
        x_shape = x.shape
        self.batch, self.outer, self.inner, self.mod, self.ch, self.features = x_shape[0], x_shape[1], x_shape[2], x_shape[3], x_shape[4], x_shape[5]

        x = einops.rearrange(x, "b outer inner mod ch k -> (outer mod) (b inner ch) k")
        if x_ca is not None:
            x_ca = einops.rearrange(x_ca, "b outer inner mod ch k -> (outer mod) (b inner ch) k")
        x = self.inner_tf_ca(x, src_ca = x_ca, extract_norm=extract_norm)
        x = einops.rearrange(x, " (outer mod) (b inner ch) k -> b outer inner mod ch k", outer=self.outer, mod=self.mod, ch=self.ch,  b=self.batch)
        return x
class outer_mod_ch_SA_CA_shared_v2(nn.Module):
    def __init__(self, d_model, pos, inner, outer, modalities, gbiased=False, extra_attention=False, rpos=False, num_layers=1, dim_proj=128, heads=8, dim_feedforward=1024):
        super().__init__()
        self.pos = pos
        if pos:
            self.pos_outer = PositionalEncoder(d_model=d_model)

        predefined_SA = My_MultiHeadAttention(d_model,  head_num=8, dim_proj=dim_proj, rpos=rpos, activation=None, gbiased = gbiased)
        predefined_CA = My_MultiHeadAttention(d_model,  head_num=8, dim_proj=dim_proj, rpos=rpos, activation=None, gbiased = gbiased)
        predefined_FC = Positionwise_FC(d_model=d_model)

        enc = My_TF_RA_ADJ(d_model, extra_attention=extra_attention, predefined_SA=predefined_SA, predefined_CA=predefined_CA, predefined_FC=predefined_FC, nhead=heads, rpos=rpos, dim_proj=dim_proj, dim_feedforward=dim_feedforward, gbiased=gbiased)
        self.outer_tf = My_TransformerEncoder(enc, num_layers=num_layers)

    def forward(self, x, extract_norm=False):
        return self.forward_sa(x)

    def forward_sa(self, x, extract_norm=False):
        x_shape = x.shape
        self.batch, self.outer, self.inner, self.mod, self.ch, self.features = x_shape[0], x_shape[1], x_shape[2], x_shape[3], x_shape[4], x_shape[5]

        x = einops.rearrange(x, "b outer inner mod ch k -> (outer mod) (b inner ch) k")
        x = self.outer_tf(x, extract_norm=extract_norm)
        x = einops.rearrange(x, " (outer mod) (b inner ch) k -> b outer inner mod ch k", outer=self.outer, mod=self.mod, ch=self.ch,  b=self.batch)
        return x

    def forward_ca(self, x, x_ca = None, extract_norm=False):
        x_shape = x.shape
        self.batch, self.outer, self.inner, self.mod, self.ch, self.features = x_shape[0], x_shape[1], x_shape[2], x_shape[3], x_shape[4], x_shape[5]

        x = einops.rearrange(x, "b outer inner mod ch k -> (outer mod) (b inner ch) k")
        if x_ca is not None:
            x_ca = einops.rearrange(x_ca, "b outer inner mod ch k -> (outer mod) (b inner ch) k")
        x = self.outer_tf(x, src_ca = x_ca, extract_norm=extract_norm)
        x = einops.rearrange(x, " (outer mod) (b inner ch) k -> b outer inner mod ch k", outer=self.outer, mod=self.mod, ch=self.ch,  b=self.batch)
        return x
class outer_mod_ch_SA_CA_CA_shared_v2(nn.Module):
    def __init__(self, d_model, pos, inner, outer, modalities, gbiased=False, extra_attention=False, rpos=False, num_layers=1, dim_proj=128, heads=8, dim_feedforward=1024):
        super().__init__()
        self.pos = pos
        if pos:
            self.pos_outer = PositionalEncoder(d_model=d_model)

        predefined_SA = My_MultiHeadAttention(d_model,  head_num=8, dim_proj=dim_proj, rpos=rpos, activation=None, gbiased = gbiased)
        predefined_CA = My_MultiHeadAttention(d_model,  head_num=8, dim_proj=dim_proj, rpos=rpos, activation=None, gbiased = gbiased)
        predefined_FC = Positionwise_FC(d_model=d_model)

        enc = My_TF_RA_ADJ(d_model, extra_attention=extra_attention, predefined_SA=predefined_SA, predefined_CA=predefined_CA, predefined_FC=predefined_FC, nhead=heads, rpos=rpos, dim_proj=dim_proj, dim_feedforward=dim_feedforward, gbiased=gbiased)
        self.outer_tf = My_TransformerEncoder(enc, num_layers=num_layers)

    def forward(self, x, extract_norm=False):
        return self.forward_sa(x)

    def forward_sa(self, x, extract_norm=False):
        x_shape = x.shape
        self.batch, self.outer, self.inner, self.mod, self.ch, self.features = x_shape[0], x_shape[1], x_shape[2], x_shape[3], x_shape[4], x_shape[5]

        x = einops.rearrange(x, "b outer inner mod ch k -> (outer mod) (b inner ch) k")
        x = self.outer_tf(x, extract_norm=extract_norm)
        x = einops.rearrange(x, " (outer mod) (b inner ch) k -> b outer inner mod ch k", outer=self.outer, mod=self.mod, ch=self.ch,  b=self.batch)
        return x

    def forward_ca(self, x, x_ca = None, x_ca_ca = None, extract_norm=False):
        x_shape = x.shape
        self.batch, self.outer, self.inner, self.mod, self.ch, self.features = x_shape[0], x_shape[1], x_shape[2], x_shape[3], x_shape[4], x_shape[5]

        x = einops.rearrange(x, "b outer inner mod ch k -> (outer mod) (b inner ch) k")
        if x_ca is not None:
            x_ca = einops.rearrange(x_ca, "b outer inner mod ch k -> (outer mod) (b inner ch) k")
        if x_ca_ca is not None:
            x_ca_ca = einops.rearrange(x_ca_ca, "b outer inner mod ch k -> (outer mod) (b inner ch) k")
        x = self.outer_tf(x, src_ca = x_ca, src_ca_ca = x_ca_ca, extract_norm=extract_norm)
        x = einops.rearrange(x, " (outer mod) (b inner ch) k -> b outer inner mod ch k", outer=self.outer, mod=self.mod, ch=self.ch,  b=self.batch)
        return x
class outer_mod_ch_SA_CA_shared_v2_fullca(nn.Module):
    def __init__(self, d_model, pos, inner, outer, modalities, gbiased=False, extra_attention=False, rpos=False, num_layers=1, dim_proj=128, heads=8, dim_feedforward=1024):
        super().__init__()
        self.pos = pos
        if pos:
            self.pos_outer = PositionalEncoder(d_model=d_model)

        predefined_SA = My_MultiHeadAttention(d_model,  head_num=8, dim_proj=dim_proj, rpos=rpos, activation=None, gbiased = gbiased)
        predefined_CA = My_MultiHeadAttention(d_model,  head_num=8, dim_proj=dim_proj, rpos=rpos, activation=None, gbiased = gbiased)
        predefined_FC = Positionwise_FC(d_model=d_model)

        enc = My_TF_RA_ADJ(d_model, extra_attention=extra_attention, predefined_SA=predefined_SA, predefined_CA=predefined_CA, predefined_FC=predefined_FC, nhead=heads, rpos=rpos, dim_proj=dim_proj, dim_feedforward=dim_feedforward, gbiased=gbiased)
        self.outer_tf = My_TransformerEncoder(enc, num_layers=num_layers)

    def forward(self, x, extract_norm=False):
        return self.forward_sa(x)

    def forward_sa(self, x, extract_norm=False, return_layer = "last"):
        x_shape = x.shape
        self.batch, self.outer, self.inner, self.mod, self.ch, self.features = x_shape[0], x_shape[1], x_shape[2], x_shape[3], x_shape[4], x_shape[5]

        x = einops.rearrange(x, "b outer inner mod ch k -> (outer mod) (b inner ch) k")
        x = self.outer_tf(x, extract_norm=extract_norm,  return_layer = return_layer)
        x = einops.rearrange(x, "layer (outer mod) (b inner ch) k ->layer b outer inner mod ch k", outer=self.outer, mod=self.mod, ch=self.ch,  b=self.batch)
        return x

    def forward_ca(self, x, x_ca = None, extract_norm=False):
        x_shape = x.shape
        self.batch, self.outer, self.inner, self.mod, self.ch, self.features = x_shape[0], x_shape[1], x_shape[2], x_shape[3], x_shape[4], x_shape[5]

        x = einops.rearrange(x, "b outer inner mod ch k -> (outer mod) (b inner ch) k")
        if x_ca is not None:
            x_ca = einops.rearrange(x_ca, "layer b outer inner mod ch k -> layer (outer mod) (b inner ch) k")
        x = self.outer_tf(x, src_ca = x_ca, extract_norm=extract_norm, ca_type="full")
        x = einops.rearrange(x, " (outer mod) (b inner ch) k -> b outer inner mod ch k", outer=self.outer, mod=self.mod, ch=self.ch,  b=self.batch)
        return x
class inner_ch_SA_CA_shared(nn.Module):
    def __init__(self, d_model, pos, inner, outer, modalities, gbiased=False, extra_attention=False, rpos=False, num_layers=1, dim_proj=128, heads=8, dim_feedforward=1024):
        super().__init__()
        self.pos = pos
        if pos:
            self.pos_inner = PositionalEncoder(d_model=d_model)

        self.inner_tf = My_TF_RA_ADJ_SA_CA(d_model, gbiased=gbiased, SA={"shared":True, "use":True,"dim_proj":128,"dropout":0.1,"rpos":rpos, "gbiased":False},
                                                        CA={"shared":True, "use":True,"dim_proj":128,"dropout":0.1,"rpos":rpos, "gbiased":False},
                                                        CA_CA={"shared":False, "use":False,"dim_proj":128, "dropout":0.1,"rpos":rpos, "gbiased":False},
                                                        FC={"shared":True, "use":True,"dim_feedforward":1024, "dropout":0.1,"activation":"relu"})
        # enc_sa = My_TF_RA_ADJ(d_model, extra_attention=extra_attention, predefined_SA=predefined_SA, predefined_CA=False, predefined_FC=predefined_FC, nhead=heads, rpos=rpos, dim_proj=dim_proj, dim_feedforward=dim_feedforward, gbiased=gbiased)
        # enc_ca = My_TF_RA_ADJ(d_model, extra_attention=extra_attention, predefined_SA=predefined_SA, predefined_CA=predefined_CA, predefined_FC=predefined_FC, nhead=heads, rpos=rpos, dim_proj=dim_proj, dim_feedforward=dim_feedforward, gbiased=gbiased)
        #
        # self.inner_tf_sa = My_TransformerEncoder(enc_sa, num_layers)
        # self.inner_tf_ca = My_TransformerEncoder_CA(enc_ca, num_layers)

    def forward(self, x, extract_norm=False):
        return self.forward_sa(x)

    def forward_sa(self, x, extract_norm=False):
        x_shape = x.shape
        self.batch, self.outer, self.inner, self.mod, self.ch, self.features = x_shape[0], x_shape[1], x_shape[2], x_shape[3], x_shape[4], x_shape[5]

        x = einops.rearrange(x, "b outer inner mod ch k -> (inner mod ch) (b outer) k")
        x = self.inner_tf(x, extract_norm=extract_norm)
        x = einops.rearrange(x, " (inner mod ch) (b outer) k -> b outer inner mod ch k", outer=self.outer, mod=self.mod, ch=self.ch,  b=self.batch)
        return x

    def forward_ca(self, x, x_ca, extract_norm=False):
        x_shape = x.shape
        self.batch, self.outer, self.inner, self.mod, self.ch, self.features = x_shape[0], x_shape[1], x_shape[2], x_shape[3], x_shape[4], x_shape[5]

        x = einops.rearrange(x, "b outer inner mod ch k -> (inner mod ch) (b outer) k")
        x_ca_u = einops.rearrange(x_ca, "b outer inner mod ch k -> (inner mod ch) (b outer) k")
        x = self.inner_tf(x, x_ca_u, extract_norm=extract_norm)
        x = einops.rearrange(x, " (inner mod ch) (b outer) k -> b outer inner mod ch k", outer=self.outer, mod=self.mod, ch=self.ch,  b=self.batch)
        return x
class inner_ch_SA_CA_CA(nn.Module):
    def __init__(self, d_model, pos, inner, outer, modalities, gbiased=False, extra_attention=False, rpos=False, num_layers=1, dim_proj=128, heads=8, dim_feedforward=1024):
        super().__init__()
        self.pos = pos
        if pos:
            self.pos_inner = PositionalEncoder(d_model=d_model)


        predefined_SA = My_MultiHeadAttention(d_model,  head_num=8, dim_proj=dim_proj, rpos=rpos, activation=None, gbiased = gbiased)
        predefined_CA_0 = My_MultiHeadAttention(d_model,  head_num=8, dim_proj=dim_proj, rpos=rpos, activation=None, gbiased = gbiased)
        predefined_CA_1 = My_MultiHeadAttention(d_model,  head_num=8, dim_proj=dim_proj, rpos=rpos, activation=None, gbiased = gbiased)
        predefined_FC = Positionwise_FC(d_model=d_model)

        self.inner_tf_sa = My_TF_RA_ADJ(d_model, extra_attention=extra_attention, predefined_SA=predefined_SA, predefined_CA=False, predefined_FC=predefined_FC, nhead=heads, rpos=rpos, dim_proj=dim_proj, dim_feedforward=dim_feedforward, gbiased=gbiased)
        self.inner_tf_ca = My_TF_RA_ADJ(d_model, extra_attention=extra_attention, predefined_SA=predefined_SA, predefined_CA=predefined_CA_0, predefined_CA_CA=predefined_CA_1, predefined_FC=predefined_FC, nhead=heads, rpos=rpos, dim_proj=dim_proj, dim_feedforward=dim_feedforward, gbiased=gbiased)

        # self.inner_tf_sa = My_TransformerEncoder(enc_sa, num_layers)
        # self.inner_tf_ca = My_TransformerEncoder_CA_CA(enc_ca, num_layers)

    def forward(self, x, extract_norm=False):
        return self.forward_sa(x)

    def forward_sa(self, x, extract_norm=False):
        x_shape = x.shape
        self.batch, self.outer, self.inner, self.mod, self.ch, self.features = x_shape[0], x_shape[1], x_shape[2], x_shape[3], x_shape[4], x_shape[5]

        x = einops.rearrange(x, "b outer inner mod ch k -> (inner mod ch) (b outer) k")
        x = self.inner_tf_sa(x, extract_norm=extract_norm)
        x = einops.rearrange(x, " (inner mod ch) (b outer) k -> b outer inner mod ch k", outer=self.outer, mod=self.mod, ch=self.ch,  b=self.batch)
        return x

    def forward_ca(self, x, x_ca, x_ca_1, extract_norm=False):
        x_shape = x.shape
        self.batch, self.outer, self.inner, self.mod, self.ch, self.features = x_shape[0], x_shape[1], x_shape[2], x_shape[3], x_shape[4], x_shape[5]

        x = einops.rearrange(x, "b outer inner mod ch k -> (inner mod ch) (b outer) k")
        x_ca_u = einops.rearrange(x_ca, "b outer inner mod ch k -> (inner mod ch) (b outer) k")
        x_ca_u_1 = einops.rearrange(x_ca_1, "b outer inner mod ch k -> (inner mod ch) (b outer) k")
        x = self.inner_tf_ca(x, x_ca_u, x_ca_u_1, extract_norm=extract_norm)
        x = einops.rearrange(x, " (inner mod ch) (b outer) k -> b outer inner mod ch k", outer=self.outer, mod=self.mod, ch=self.ch,  b=self.batch)
        return x

class modtype_embedding(nn.Module):
    def __init__(self, num_modalities, dim):
        super().__init__()
        self.mod_tokens = nn.Embedding(num_modalities, dim)

    def forward(self, data, mod_num):
        return data + self.mod_tokens(torch.IntTensor([mod_num]).to(data.device)).squeeze()

