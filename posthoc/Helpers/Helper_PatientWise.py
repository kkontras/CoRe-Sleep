import torch
import einops
import copy
import numpy as np
from sklearn.metrics import f1_score, cohen_kappa_score, roc_auc_score, confusion_matrix
from scipy.stats import entropy
from scipy.special import softmax
import matplotlib.pyplot as plt
import pickle
from collections import defaultdict
import plotly.graph_objects as go


class PatientWise_Analyser():
    def __init__(self, data_loader, device):
        self.data_loader = data_loader
        self.device = device

    def validate_specific_patient(self, set, data_loader, patient_num, models, only_align_model, device,
                                  plot_hypnograms=True, return_matches=True, std_router=False, zc_router=False, plot_matches=False, router_models=False,
                                  plot_entropy=True, once_whole_set=False):
        for m in models: models[m].eval()

        this_data_loader = copy.deepcopy(data_loader)

        if not once_whole_set:
            this_data_loader.dataset.choose_specific_patient([patient_num])

        if len(this_data_loader)==0: return {}
        # this_data_loader.dataset.config.statistics["print"] = True
        # this_data_loader.dataset.print_statistics_per_patient()
        print(patient_num)
        with torch.no_grad():
            try:
                tts, ids,  preds, matches, matches_onlyalign, inits, inter_eeg, inter_eog = [], [], [], [], [], [], [], []
                keep_views, keep_ids = defaultdict(lambda: []), defaultdict(lambda: [])
                preds = {i:[] for i in list(models.keys())}
                if router_models:
                    # preds.update({i: [] for i in list(router_models.keys())})
                    vae_router = {i: [] for i in list(router_models.keys())}

                for batch_idx, served_dict in enumerate(this_data_loader):
                    # print("{}/{}".format(batch_idx, len(this_data_loader)))

                    served_dict["data"] = {view: served_dict["data"][view].float().to(self.device) for view in served_dict["data"]}
                    label = served_dict["label"][list(served_dict["label"].keys())[0]].flatten(start_dim=0, end_dim=1).to(self.device)

                    # if return_matches:
                    #     output = model(views, return_matches=return_matches, return_inter_reps=True)
                    #     matches.append(output["matches"])

                    for i, m in enumerate(models):
                        if m=="eog":
                            output = models[m](served_dict["data"])
                        elif m=="blip_tm":
                            output = models[m](served_dict["data"], skip_modality="vae")
                        elif m=="blip_tm_i":
                            output = models[m](served_dict["data"], skip_modality="full", return_matches=True)
                            # output_1 = only_align_model(served_dict["data"], skip_modality="full", return_matches=True)

                            if len(output["matches"]["stft_eeg"].shape) == 2:
                                alignment_target = torch.arange(output["matches"]["stft_eeg"].shape[0]).to(
                                    output["matches"]["stft_eeg"].device)
                            else:
                                alignment_target = torch.arange(output["matches"]["stft_eeg"].shape[1]).tile(
                                    output["matches"]["stft_eeg"].shape[0]).to(output["matches"]["stft_eeg"].device)
                                output["matches"]["stft_eeg"] = output["matches"]["stft_eeg"].flatten(start_dim=0, end_dim=1)
                                output["matches"]["stft_eog"] = output["matches"]["stft_eog"].flatten(start_dim=0, end_dim=1)


                            import torch.nn as nn
                            alignment_loss = nn.CrossEntropyLoss(reduction="none")(output["matches"]["stft_eeg"], alignment_target)
                            alignment_loss += nn.CrossEntropyLoss(reduction="none")(output["matches"]["stft_eog"], alignment_target)
                            matches.append(alignment_loss.cpu())

                            # alignment_target = torch.arange(output_1["matches"]["stft_eeg"].shape[1]).tile(
                            #     output_1["matches"]["stft_eeg"].shape[0]).to(output_1["matches"]["stft_eeg"].device)
                            # output_1["matches"]["stft_eeg"] = output_1["matches"]["stft_eeg"].flatten(start_dim=0, end_dim=1)
                            # output_1["matches"]["stft_eog"] = output_1["matches"]["stft_eog"].flatten(start_dim=0, end_dim=1)
                            # import torch.nn as nn
                            # alignment_loss = nn.CrossEntropyLoss(reduction="none")(output_1["matches"]["stft_eeg"], alignment_target)
                            # alignment_loss += nn.CrossEntropyLoss(reduction="none")(output_1["matches"]["stft_eog"], alignment_target)
                            # matches_onlyalign.append(alignment_loss.cpu())

                        elif m=="blip_tm_eeg":
                            output = models[m](served_dict["data"], skip_modality="eog")
                            output["preds"]["combined"] = output["preds"]["eeg"]
                        elif m=="blip_tm_eog":
                            output = models[m](served_dict["data"], skip_modality="eeg")
                            output["preds"]["combined"] = output["preds"]["eog"]
                        elif m=="blip_skip":
                            output = self.get_predictions_time_series(models[m], served_dict["data"], served_dict["init"], skip_modality=served_dict["skip_view"])
                        else:
                            output = models[m](served_dict["data"])
                        preds[m].append(output["preds"]["combined"].cpu())
                        # if return_matches:
                        #     raise NotImplementedError()
                    for v in served_dict["data"]:
                        keep_views[v].append(served_dict["data"][v].cpu())
                        keep_ids[v].append(served_dict["ids"][v])
                    tts.append(label.cpu())
                    ids.append(id)

                    if router_models:
                        for router_key in list(router_models.keys()):
                            pred = router_models[router_key](served_dict["data"])
                            output_losses = router_models[router_key].module.loss_function(pred[0], pred[1], pred[2], pred[3], reduction="none")
                            this_routing = output_losses["total"].flatten(start_dim=1).mean(dim=1).cpu().numpy()
                            vae_router[router_key].append(this_routing)
                            # vae_router_indications = copy.deepcopy(this_routing)
                            # vae_router_indications[vae_router_indications < 3] = 0
                            # vae_router_indications[vae_router_indications > 3] = 1
                            # # views[0][vae_router_indications.astype(bool)] = 0
                            # output = models["blip"](views, skip_modality = vae_router_indications)
                            # preds[router_key].append(output["preds"]["combined"])

                    # preds.append(output["preds"]["combined"])
                    # preds.append(pred)
                    # views_eeg_time.append(views[2])
                    # views_eog_time.append(views[3])
                    # if type(output)==dict and "inter_reps" in output:
                    #     inter_eeg.append(output["inter_reps"][0])
                    #     inter_eog.append(output["inter_reps"][1])
                    # inits.append(init.flatten()
                for i in preds:
                    if preds[i] != []: preds[i] = torch.cat(preds[i]).cpu().numpy()
                keep_views, keep_ids = dict(keep_views), dict(keep_ids)
                single_batch_flag = True if keep_views[list(keep_views.keys())[0]][0].shape[0]==1 else False

                for v in keep_views: keep_views[v] = torch.cat(keep_views[v]).cpu().squeeze()
                for v in keep_ids: keep_ids[v] = torch.cat(keep_ids[v]).cpu().squeeze()
                if single_batch_flag:
                    for v in keep_views: keep_views[v] = keep_views[v].unsqueeze(dim=0)
                    for v in keep_ids: keep_ids[v] = keep_ids[v].unsqueeze(dim=0)
                tts = torch.cat(tts).cpu().numpy()

                if router_models:
                    vae_router_indications = {}
                    for router_key in list(router_models.keys()):
                        vae_router[router_key] = np.concatenate(vae_router[router_key])
                        vae_router_indications[router_key] = copy.deepcopy(vae_router[router_key])
                        vae_router_indications[router_key][vae_router_indications[router_key]<=3] = 0
                        if router_key == "vae_eeg":
                            vae_router_indications[router_key][vae_router_indications[router_key] >= 1] = 1
                        if router_key == "vae_eog":
                            vae_router_indications[router_key][vae_router_indications[router_key] > 1] = 2
                    vae_preds = copy.deepcopy(preds["blip_tm_i"])
                    vae_router_bools = vae_router_indications["vae_eog"] + vae_router_indications["vae_eeg"]
                    vae_router_bools[vae_router_bools==3]=0
                    # print(vae_router_bools)

                    vae_router_bools = vae_router_bools.repeat(21)

                    vae_preds[vae_router_bools==1] = preds["blip_tm_eog"][vae_router_bools==1]
                    vae_preds[vae_router_bools==2] = preds["blip_tm_eeg"][vae_router_bools==2]
                    preds.update({"vae":vae_preds})
                if std_router:
                    std_router_indications = {}
                    std_router_bools = {}
                    for router_key in list(["std_eeg", "std_eog"]):
                        data_key = "time_eeg" if router_key=="std_eeg" else "time_eog"
                        std_router_indications[router_key] = keep_views[data_key].std(dim=2).mean(dim=1)
                        std_router_bools[router_key] = copy.deepcopy(std_router_indications[router_key])
                        std_router_bools[router_key][std_router_bools[router_key] <= 35] = 0
                        std_router_bools[router_key][std_router_bools[router_key] > 35] = 1  if router_key == "std_eeg" else 2

                    std_router_bools["total"] = std_router_bools["std_eeg"] + std_router_bools["std_eog"]
                    # std_router_bools["total"][std_router_bools["total"] == 3] = 0


                    # for i in std_router_bools: std_router_bools[i] = std_router_bools[i].unsqueeze(dim=1).repeat(1,21).flatten()
                    # std_preds = copy.deepcopy(preds["blip_tm_i"])
                    # std_preds[std_router_bools["total"] == 1] = preds["blip_tm_eog"][std_router_bools["total"] == 1]
                    # std_preds[std_router_bools["total"] == 2] = preds["blip_tm_eeg"][std_router_bools["total"] == 2]
                    # preds.update({"std": std_preds})
                if zc_router:
                    zc_router_indications = {}
                    zc_router_bools = {}
                    for router_key in list(["zc_eeg", "zc_eog"]):
                        data_key = "time_eeg" if router_key=="zc_eeg" else "time_eog"
                        zc_router_indications[router_key] = keep_views[data_key]
                        zc_router_shape = zc_router_indications[router_key].shape
                        zc_router_indications[router_key] = einops.rearrange(zc_router_indications[router_key], "b s t -> (b s) t")
                        zc_router_indications[router_key] = einops.rearrange(einops.rearrange(zc_router_indications[router_key], "b t -> t b") - zc_router_indications[router_key].mean(dim=1), "t b -> b t")
                        zc_router_indications[router_key] = np.array([len(np.where(np.diff(np.sign(zc_router_indications[router_key][i])))[0]) for i in range(0, len(zc_router_indications[router_key]))])
                        zc_router_indications[router_key] = einops.rearrange(zc_router_indications[router_key], "(b s) -> b s", b = zc_router_shape[0], s = zc_router_shape[1]).mean(axis=1)
                        zc_router_bools[router_key] = copy.deepcopy(zc_router_indications[router_key])
                        zc_router_bools[router_key][zc_router_bools[router_key] <= 650] = 0
                        zc_router_bools[router_key][zc_router_bools[router_key] > 650] = 1  if router_key == "zc_eeg" else 2

                    zc_preds = copy.deepcopy(preds["blip_tm_i"])
                    zc_router_bools["total"] = zc_router_bools["zc_eog"] + zc_router_bools["zc_eeg"]
                    zc_router_bools["total"][zc_router_bools["total"] == 3] = 0

                    # print(zc_router_indications)
                    # print(zc_router_bools["total"])

                    for i in zc_router_bools: zc_router_bools[i] = torch.from_numpy(zc_router_bools[i]).unsqueeze(dim=1).repeat(1,21).flatten().numpy()
                    for i in zc_router_indications: zc_router_indications[i] = torch.from_numpy(zc_router_indications[i]).unsqueeze(dim=1).repeat(1,21).flatten().numpy()

                    zc_preds[zc_router_bools["total"] == 1] = preds["blip_tm_eog"][zc_router_bools["total"] == 1]
                    zc_preds[zc_router_bools["total"] == 2] = preds["blip_tm_eeg"][zc_router_bools["total"] == 2]
                    preds.update({"zc": zc_preds})
                for i in keep_views: keep_views[i] = keep_views[i].flatten(start_dim=0, end_dim=1)
                # for i in keep_ids: keep_ids[i] = keep_ids[i].flatten(start_dim=0, end_dim=1)

                results = {}
                if std_router:

                    std_est = std_router_indications["std_eeg"]-std_router_indications["std_eog"]
                    # vae_est = np.abs(vae_router["vae_eeg"]-vae_router["vae_eog"])
                    std_threshold = 35
                    vae_threshold = 3.5

                    std_est_eeg = copy.deepcopy(std_est)
                    std_est_eeg[std_est_eeg<=std_threshold] = 0
                    std_est_eeg[std_est_eeg>std_threshold] = 1
                    std_est_eog = copy.deepcopy(std_est)
                    std_est_eog[std_est_eog >= -std_threshold] = 0
                    std_est_eog[std_est_eog < -std_threshold] = 1

                    # vae_est_eeg = copy.deepcopy(vae_est)
                    # vae_est_eeg[vae_est_eeg<=vae_threshold] = 0
                    # vae_est_eeg[vae_est_eeg>vae_threshold] = 1
                    # vae_est_eog = copy.deepcopy(std_est)
                    # vae_est_eog[vae_est_eog >= -vae_threshold] = 0
                    # vae_est_eog[vae_est_eog < -vae_threshold] = 1

                    chosen_indices_std_eeg = keep_ids["stft_eeg"][std_router_bools["total"] == 1]
                    temp = copy.deepcopy(chosen_indices_std_eeg)[:, :, 0:1]
                    temp[:,:,0]=1
                    chosen_indices_std_eeg = torch.cat([chosen_indices_std_eeg, temp], dim=2)
                    chosen_indices_std_eog = keep_ids["stft_eog"][std_router_bools["total"] == 2]
                    temp = copy.deepcopy(chosen_indices_std_eog)[:, :, 0:1]
                    temp[:, :, 0] = 2
                    chosen_indices_std_eog = torch.cat([chosen_indices_std_eog, temp], dim=2)
                    chosen_indices_std_both = keep_ids["stft_eeg"][std_router_bools["total"] == 3]
                    temp = copy.deepcopy(chosen_indices_std_both)[:, :, 0:1]
                    temp[:, :, 0] = 3
                    chosen_indices_std_both = torch.cat([chosen_indices_std_both, temp], dim=2)
                    chosen_indices_std = torch.cat([chosen_indices_std_eeg, chosen_indices_std_eog, chosen_indices_std_both], dim=0)


                    # chosen_indices_vae_eeg = keep_ids["stft_eeg"][vae_est_eeg == 1]
                    # temp = copy.deepcopy(chosen_indices_vae_eeg)[:, :, 0:1]
                    # temp[:,:,0]=1
                    # chosen_indices_vae_eeg = torch.cat([chosen_indices_vae_eeg, temp], dim=2)
                    # chosen_indices_vae_eog = keep_ids["stft_eog"][vae_est_eog == 1]
                    # temp = copy.deepcopy(chosen_indices_vae_eog)[:, :, 0:1]
                    # temp[:, :, 0] = 2
                    # chosen_indices_vae_eog = torch.cat([chosen_indices_vae_eog, temp], dim=2)
                    # chosen_indices_vae = torch.cat([chosen_indices_vae_eeg, chosen_indices_vae_eog], dim=0)

                    file = open("/users/sista/kkontras/Documents/Sleep_Project/experiments/baddiff_chosen_shhs_train.pkl",
                                "rb")
                    prev_chosen_indices = pickle.load(file)
                    file.close()
                    prev_chosen_indices["std_diff"] = torch.cat([prev_chosen_indices["std_diff"], chosen_indices_std],
                                                                dim=0)
                    # prev_chosen_indices["vae_diff"] = torch.cat([prev_chosen_indices["vae_diff"], chosen_indices_vae],
                    #                                             dim=0)
                    print(prev_chosen_indices["std_diff"].shape)
                    file = open("/users/sista/kkontras/Documents/Sleep_Project/experiments/baddiff_chosen_shhs_train.pkl",
                                "wb")
                    pickle.dump(prev_chosen_indices, file)
                    file.close()
                    print("Results saved!")
                    results["chosen_idx_std"] = chosen_indices_std

                # print(chosen_indices_std)
                # print(chosen_indices_vae)

                # inter_eeg = torch.cat(inter_eeg).cpu().squeeze().flatten(start_dim=0, end_dim=1)
                # inter_eeg = einops.rearrange(inter_eeg, "a b c -> (a c) b").numpy()
                # inter_eog = torch.cat(inter_eog).cpu().squeeze().flatten(start_dim=0, end_dim=1)
                # inter_eog = einops.rearrange(inter_eog, "a b c -> (a c) b").numpy()
                window_length = 21


                tts_unfolded = torch.from_numpy(tts).unfold(0, window_length, window_length).numpy()
                preds_unfolded, perf_window = {}, {}
                for p in preds:
                    preds_unfolded[p] = torch.from_numpy(preds[p]).argmax(dim=-1).unfold(0, window_length, window_length).numpy()
                for p in preds_unfolded:
                    perf_window[p] = np.array([np.equal(tts_unfolded[i], preds_unfolded[p][i]).sum() / len(tts_unfolded[i]) for i in range(tts_unfolded.shape[0])])
                    perf_window[p][perf_window[p] != perf_window[p]] = 1
                    # for i in perf_window[p]: print("{:.3f}".format(i), end=" ")
                    perf_window[p] = perf_window[p].repeat(window_length)

                # if return_matches:
                matches = torch.cat(matches).unfold(0, window_length, window_length).mean(dim=1).cpu()
                # matches_onlyalign = torch.cat(matches_onlyalign).cpu()
                multiclass = True

                if len(tts.shape) > 2: tts = tts.argmax(axis=-1)
                tts = tts.flatten()

                this_entropy = {}
                for p in preds:
                    this_pred = preds[p].argmax(-1)
                    this_entropy[p] = entropy(softmax(preds[p], axis=1), axis=1)
                    this_entropy_correct = this_entropy[p][this_pred == tts].mean()
                    this_entropy_correct_var = this_entropy[p][this_pred == tts].std()
                    this_entropy_wrong = this_entropy[p][this_pred != tts].mean()
                    this_entropy_wrong_var = this_entropy[p][this_pred != tts].std()
                    test_acc = np.equal(tts, this_pred).sum() / len(tts)
                    test_f1 = f1_score(this_pred, tts) if not multiclass else f1_score(this_pred, tts, average="macro")
                    test_perclass_f1 = f1_score(this_pred, tts) if not multiclass else f1_score(this_pred, tts, average=None)
                    test_k = cohen_kappa_score(tts, this_pred)
                    test_auc = roc_auc_score(tts, this_pred) if not multiclass else 0
                    test_conf = confusion_matrix(tts, this_pred)
                    tp, fp, tn, fn = self._perf_measure(tts, this_pred)
                    test_spec = tn / (tn + fp) if (tn + fp) != 0 else 0
                    test_sens = tp / (tp + fn) if (tp + fn) != 0 else 0
                    # models_key = list(models.keys())[p]
                    results[p] = {"acc": test_acc, "f1": test_f1, "k":test_k, "f1_perclass":test_perclass_f1, "entropy": this_entropy[p].mean(), "entropy_correct":this_entropy_correct, "entropy_wrong":this_entropy_wrong }
                    print("Patient {} Model {} has acc: {}, f1: {}, k:{} and f1_per_class: {}".format(patient_num, p,
                                                                                                    round(test_acc * 100,
                                                                                                          1),
                                                                                                    round(test_f1 * 100, 1),
                                                                                                    round(test_k, 3),
                                                                                                    np.round(
                                                                                                        test_perclass_f1 * 100,
                                                                                                        1)))
                # results["chosen_idx_vae"] = chosen_indices_vae
                    # print("Entropy {:.2f}, Entropy Corr {:.2f}, var {:.2f} and Entropy Wrong {:.2f}, var {:.2f}".format( this_entropy[p].mean(), this_entropy_correct, this_entropy_correct_var, this_entropy_wrong, this_entropy_wrong_var))

                print() #Just to recognize difference between patients in log output!

                if not plot_hypnograms: return results
                # if results["eeg"]["acc"] > 0.75: return results
                # if plot_hypnogram_flag or True:
                    # if test_f1>0.85:

                def renumbering(p, argmaxit = False):
                    if argmaxit: p = p.argmax(-1)
                    p[p == 4] = 5
                    p[p == 3] = 4
                    p[p == 2] = 3
                    p[p == 5] = 2
                    return p
                pred_plus = {p: renumbering(preds[p], argmaxit=True) for p in preds}
                target_plus = renumbering(tts)

                from_hours_to_plot = int(120 * 4)
                hours_to_plot = -1  # int(120*2.5)

                # print("TIME", from_hours_to_plot, hours_to_plot)
                target_plus = target_plus[from_hours_to_plot:hours_to_plot]
                non_matches_idx = {}
                for p in preds:
                    pred_plus[p] = pred_plus[p][from_hours_to_plot:hours_to_plot]
                    non_matches = (pred_plus[p] != target_plus).astype(int)
                    non_matches_idx[p] = non_matches.nonzero()[0]

                for i in keep_views:
                    keep_views[i] = keep_views[i][from_hours_to_plot:hours_to_plot]

                # match_loss = match_loss[:hours_to_plot]

                target_plus = target_plus + 0.02
                hours = len(target_plus)

                    # ce_loss = nn.CrossEntropyLoss(reduction='none')(torch.from_numpy(preds_for_loss),
                    #                                                 torch.from_numpy(tts))

                    # combined_loss = match_loss + ce_loss

                    # match_loss = (match_loss - match_loss.mean())/match_loss.std()
                    # ce_loss = (ce_loss - ce_loss.mean())/ce_loss.std()
                    # combined_loss = (combined_loss - combined_loss.mean())/combined_loss.std()

                    # def hl_envelopes_idx(s, dmin=-1, dmax=1, split=False):
                    #     """
                    #     Input :
                    #     s: 1d-array, data signal from which to extract high and low envelopes
                    #     dmin, dmax: int, optional, size of chunks, use this if the size of the input signal is too big
                    #     split: bool, optional, if True, split the signal in half along its mean, might help to generate the envelope in some cases
                    #     Output :
                    #     lmin,lmax : high/low envelope idx of input signal s
                    #     """
                    #
                    #     # locals min
                    #     lmin = (np.diff(np.sign(np.diff(s))) > 0).nonzero()[0] + 1
                    #     # locals max
                    #     lmax = (np.diff(np.sign(np.diff(s))) < 0).nonzero()[0] + 1
                    #
                    #     if split:
                    #         # s_mid is zero if s centered around x-axis or more generally mean of signal
                    #         s_mid = np.mean(s)
                    #         # pre-sorting of locals min based on relative position with respect to s_mid
                    #         lmin = lmin[s[lmin] < s_mid]
                    #         # pre-sorting of local max based on relative position with respect to s_mid
                    #         lmax = lmax[s[lmax] > s_mid]
                    #
                    #     # global max of dmax-chunks of locals max
                    #     lmin = lmin[[i + np.argmin(s[lmin[i:i + dmin]]) for i in range(0, len(lmin), dmin)]]
                    #     # global min of dmin-chunks of locals min
                    #     lmax = lmax[[i + np.argmax(s[lmax[i:i + dmax]]) for i in range(0, len(lmax), dmax)]]
                    #
                    #     return lmin, lmax
                    #
                    # # high_idx_match, low_idx_match = hl_envelopes_idx(match_loss.numpy())
                    # # high_idx_inter_dist, low_idx_inter_dist = hl_envelopes_idx(inter_distance.numpy())
                    # high_idx_ce, low_idx_ce = hl_envelopes_idx(ce_loss.numpy())
                    # high_idx_combined, low_idx_combined = hl_envelopes_idx(combined_loss.numpy())

                    # x = np.linspace(0, len(match_loss) - 1, len(match_loss))

                    # plt.subplot(212)
                    # # plt.plot(match_loss.numpy(), label="Matches")
                    # # plt.plot(entropy_pred, label="Entropy")
                    # # plt.plot(low_idx, label="Match Loss")
                    # plt.plot(x[low_idx], match_loss[low_idx], 'g', label='high')
                    # plt.scatter(non_matches_idx, entropy_pred[non_matches_idx], marker='*', edgecolors="r", label="Mistakes")
                    # # plt.scatter(non_matches_idx, match_loss[non_matches_idx], marker='*', edgecolors="r", label="Mistakes")
                    # plt.xticks([i * 120 for i in range((hours // 120) + 1)],
                    #            labels=["{}".format(i) for i in range((hours // 120) + 1)])
                    # # plt.title("Patient {}".format(patient_num))
                    # # plt.ylabel("CE loss Envelope")
                    # plt.ylabel("Entropy Envelope")
                    # plt.xlabel("Hours")
                    # plt.show()

                def plot_signals(input_views, current_figure, num_figures, xlabel_hours=False, only=None):
                    views = copy.deepcopy(input_views)
                    for v in views:
                        if len(views[v].shape) == 2:
                            if only == "fourier": continue
                            current_figure = next_subplot(num_figures, current_figure)
                            print("This passed")
                            plot_single(views[v].flatten().numpy())
                            continue
                        if only == "time": continue
                        current_figure = next_subplot(num_figures, current_figure)
                        views[v] = einops.rearrange(views[v], "b f i -> (b i) f").cpu().numpy()
                        t = np.linspace(0, len(views[v]) - 1, len(views[v]))
                        f = np.linspace(0, 129 - 1, 129)
                        plt.pcolormesh(t, f, views[v].transpose(), shading='auto')
                        plt.xticks([])
                        plt.yticks([])
                        if v==0:
                            plt.ylabel("F EEG")
                        elif v==2:
                            plt.ylabel("F EOG")
                    # hours = len(t)//29
                    if xlabel_hours:
                        plt.xticks([i * 120 * 29 for i in range((hours // 120) + 1)],
                                   labels=["{}".format(i) for i in range((hours // 120) + 1)])
                        plt.xlabel("Hours")
                    return current_figure
                def plot_pred(preds, targets, non_matches_idx, description="", xlabel_hours=False):

                    plt.plot(preds, label="Prediction", linewidth=0.6)
                    plt.plot(targets, label="True Label", linewidth=0.6)
                    plt.scatter(non_matches_idx, preds[non_matches_idx],
                                marker='*', edgecolors="r", label="Mistakes", linewidth=0.6)
                    # plt.plot(non_matches_idx,"*")
                    plt.yticks([0, 1, 2, 3, 4], labels=["Wake", "N1", "REM", "N2", "N3"])
                    if xlabel_hours:
                        plt.xticks([i * 120 for i in range((hours // 120) + 1)],
                                   labels=["{}".format(i) for i in range((hours // 120) + 1)])
                        plt.xlabel("Hours")

                    plt.xticks([])
                    plt.xlim(0, hours)
                    plt.yticks(fontsize=8)
                    plt.ylabel("{}".format(description), fontsize=8)
                def plot_single(var, description="",  legend= "", xlabel_hours=False, ylim=False):

                    plt.plot(var, linewidth=0.6, label = legend)
                    plt.xlim(0, len(var)-1)
                    if ylim:
                        plt.ylim(ylim[0], ylim[1])
                    plt.yticks(fontsize=8)
                    plt.xticks([])
                    plt.ylabel("{}".format(description), fontsize=8)
                    if xlabel_hours:
                        plt.xticks([i * 120 for i in range((hours // 120) + 1)],
                                   labels=["{}".format(i) for i in range((hours // 120) + 1)])
                        plt.xlabel("Hours")
                def plot_matches(matches, description="", xlabel_hours=False, ylim=False):

                    plt.plot(matches, linewidth=0.6)
                    plt.xlim(0, len(matches))
                    if ylim:
                        plt.ylim(ylim[0], ylim[1])
                    plt.yticks(fontsize=8)
                    plt.xticks([])
                    plt.ylabel("{}".format(description), fontsize=8)
                    if xlabel_hours:
                        plt.xticks([i * 120 for i in range((hours // 120) + 1)],
                                   labels=["{}".format(i) for i in range((hours // 120) + 1)])
                        plt.xlabel("Hours")
                def next_subplot(num_figures, current_figure):
                    plt.subplot(int("{}1{}".format(num_figures, current_figure)))
                    return current_figure + 1

                num_figures = 4
                current_figure = 1
                plt.figure()
                plt.title("patient_{}".format(patient_num))
                # current_figure = next_subplot(num_figures, current_figure)
                # plot_pred(preds=pred_plus["blip_tm"], targets=target_plus, non_matches_idx=non_matches_idx["blip_tm"], description="blip_tm")
                # current_figure = next_subplot(num_figures, current_figure)
                # plot_pred(preds=pred_plus["vae"], targets=target_plus, non_matches_idx=non_matches_idx["vae"], description="vae")



                # current_figure = next_subplot(num_figures, current_figure)
                # plot_pred(preds=pred_plus["blip_tm_i"], targets=target_plus, non_matches_idx=non_matches_idx["blip_tm_i"], description="Multimodal")
                # current_figure = next_subplot(num_figures, current_figure)
                # plot_pred(preds=pred_plus["blip_tm_eeg"], targets=target_plus, non_matches_idx=non_matches_idx["blip_tm_eeg"], description="EEG")
                # current_figure = next_subplot(num_figures, current_figure)
                # plot_pred(preds=pred_plus["blip_tm_eog"], targets=target_plus, non_matches_idx=non_matches_idx["blip_tm_eog"], description="EOG")



                # current_figure = next_subplot(num_figures, current_figure)
                # plot_pred(preds=pred_plus["std"], targets=target_plus, non_matches_idx=non_matches_idx["std"], description="std")
                # current_figure = next_subplot(num_figures, current_figure)
                # plot_pred(preds=pred_plus["zc"], targets=target_plus, non_matches_idx=non_matches_idx["zc"], description="zc")
                # plt.show()
                # current_figure = next_subplot(num_figures, current_figure)
                # plot_matches(matches=matches, description="AL loss", ylim=[0,10])
                # current_figure = next_subplot(num_figures, current_figure)
                # plot_matches(matches=matches_onlyalign)

                # num_figures = 6
                # current_figure = 1
                # plt.figure()
                # plt.title("patient_{}".format(patient_num))

                # current_figure = next_subplot(num_figures, current_figure)
                # plot_pred(preds=pred_plus["blip"], targets=target_plus, non_matches_idx=non_matches_idx["blip"], description="BLIP")
                # plt.title("Patient {}".format(patient_num))

                # current_figure = next_subplot(num_figures, current_figure)
                # plot_single(var=this_entropy["blip"], description="Entropy BLIP")

                # current_figure = next_subplot(num_figures, current_figure)
                # plot_pred(preds=pred_plus["eeg"], targets=target_plus, non_matches_idx=non_matches_idx["eeg"], description="EEG")

                # current_figure = next_subplot(num_figures, current_figure)
                # plot_single(var=this_entropy["eeg"], description="Entropy EEG", xlabel_hours=True)

                # current_figure = next_subplot(num_figures, current_figure)
                # plot_pred(preds=pred_plus["eog"], targets=target_plus, non_matches_idx=non_matches_idx["eog"], description="EΟG")

                # for i in pred_plus:
                #     if current_figure==0: plt.title("Patient {}".format(patient_num))
                #     current_figure = next_subplot(num_figures, current_figure)
                #     plot_pred(preds=pred_plus[i], targets=target_plus, non_matches_idx=non_matches_idx[i], description=i)
                    # current_figure = next_subplot(num_figures, current_figure)
                    # plot_single(var=this_entropy[i], description="Entropy {}".format(i), xlabel_hours=True)

                # current_figure = next_subplot(num_figures, current_figure)
                # plot_pred(preds=pred_plus["blip"], targets=target_plus, non_matches_idx=non_matches_idx["blip"], description="BLIP")


                # for i in vae_router:
                #     current_figure = next_subplot(num_figures, current_figure)
                #     plot_single(var=vae_router[i].repeat(21), description="{}".format(i), xlabel_hours=False, ylim=[0,10])
                # print(vae_router.keys())
                # current_figure = next_subplot(num_figures, current_figure)
                # plot_single(var=vae_router["vae_eeg"].repeat(21)-vae_router["vae_eog"].repeat(21), description="VAE Diff", xlabel_hours=False, ylim=[-5,5])
                # plot_single(var=np.zeros(len(vae_router["vae_eeg"].repeat(21))), description="VAE Diff", xlabel_hours=False, ylim=[-5,5])
                # plot_single(var=np.zeros(len(vae_router["vae_eeg"].repeat(21)))+vae_threshold, description="VAE Diff", xlabel_hours=False, ylim=[-5,5])
                # plot_single(var=np.zeros(len(vae_router["vae_eeg"].repeat(21)))-vae_threshold, description="VAE Diff", xlabel_hours=False, ylim=[-5,5])
                #
                # i = "std_eeg"
                # current_figure = next_subplot(num_figures, current_figure)
                # plot_single(var=std_router_indications["std_eeg"].unsqueeze(dim=1).repeat(1,21).flatten()-std_router_indications["std_eog"].unsqueeze(dim=1).repeat(1,21).flatten(), description="STD Diff", xlabel_hours=False, ylim=[-100,100])
                # plot_single(var=np.zeros(len(std_router_indications["std_eeg"].repeat(21))), description="STD Diff", xlabel_hours=False, ylim=[-100,100])
                # plot_single(var=np.zeros(len(std_router_indications[i].repeat(21)))+std_threshold, description="STD Diff", xlabel_hours=False, ylim=[-100,100])
                # plot_single(var=np.zeros(len(std_router_indications[i].repeat(21)))-std_threshold, description="STD Diff", xlabel_hours=False, ylim=[-100,100])

                # for i in zc_router_indications:
                #     current_figure = next_subplot(num_figures, current_figure)
                #     plot_single(var=zc_router_indications[i], description="{}".format(i), xlabel_hours=False)

                # for i in std_router_indications:
                #     current_figure = next_subplot(num_figures, current_figure)
                #     plot_single(var=std_router_indications[i], description="{}".format(i), xlabel_hours=False, ylim=[0,100])

                def plot_time_std_on_paper():
                    num_figures = 4
                    current_figure = 1
                    fig = plt.figure(figsize=(20, 8))
                    plt.title("patient_{}".format(patient_num))
                    views = copy.deepcopy(keep_views)
                    only = None
                    xlabel_hours = True
                    fsize = 20
                    ax = plt.subplot(211)
                    blue_line = plt.plot(views["time_eeg"].flatten().numpy(), linewidth=0.5, label="EEG C4-A1")
                    plt.xlim(0, len(views["time_eeg"].flatten().numpy()) - 1)
                    # plot_single(views["time_eeg"].flatten().numpy(), legend="EEG C4-A1")
                    # plt.ylabel("EEG C4-Cz")
                    orange_line = plt.plot(views["time_eog"].flatten().numpy(), linewidth=0.5, label="EOG L-R")
                    plt.xlim(0, len(views["time_eog"].flatten().numpy()) - 1)
                    plt.yticks([-100, 0, 100], fontsize=fsize - 4)
                    plt.xticks([])
                    leg = ax.legend(fontsize=fsize, bbox_to_anchor=(0.5, 0), frameon=False, loc='center', ncol=2)
                    # change the line width for the legend
                    for line in leg.get_lines():
                        line.set_linewidth(5.0)
                    # plt.legend(frameon=False, fontsize=12, )
                    ax.spines['top'].set_visible(False)
                    # ax.spines['left'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    ax.spines['bottom'].set_visible(False)
                    plt.ylabel("Time Signals (mV)", fontsize=fsize)
                    ax = plt.subplot(212)
                    plt.plot(views["time_eeg"].std(dim=1))
                    plt.plot(views["time_eog"].std(dim=1))
                    plt.ylabel("STD", fontsize=fsize)
                    ax.spines['top'].set_visible(False)
                    # ax.spines['left'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    plt.yticks([0, 20, 40, 60, 80, 100], fontsize=fsize - 4)
                    plt.xlim(0, len(views["time_eeg"]))
                    plt.xticks([i * 120 for i in range((views["time_eog"].shape[0] // 120) + 1)],
                               labels=["{}".format(i) for i in range((views["time_eog"].shape[0] // 120) + 1)],
                               fontsize=fsize - 4)
                    plt.xlabel("Hours", fontsize=fsize)
                    fig.align_ylabels()
                    plt.savefig("/users/sista/kkontras/Documents/Sleep_Project/time_std_sample_patient_1243_1.svg")
                    plt.show()

                    num_figures = 4
                    current_figure = 1
                    fig = plt.figure(figsize=(20, 8))
                    plt.title("patient_{}".format(patient_num))
                    views = copy.deepcopy(keep_views)
                    this_pred = copy.deepcopy(preds)
                    this_tts = renumbering(copy.deepcopy(tts)[500:])
                    for i in views: views[i] = views[i][500:]
                    for i in this_pred: this_pred[i] = renumbering(this_pred[i][500:], argmaxit=True)
                    only = None
                    xlabel_hours = True
                    fsize = 20
                    ax = plt.subplot(311)
                    blue_line = plt.plot(views["time_eeg"].flatten().numpy(), linewidth=0.5, label="EEG C4-A1")
                    plt.xlim(0, len(views["time_eeg"].flatten().numpy()) - 1)
                    # plot_single(views["time_eeg"].flatten().numpy(), legend="EEG C4-A1")
                    # plt.ylabel("EEG C4-Cz")
                    orange_line = plt.plot(views["time_eog"].flatten().numpy(), linewidth=0.5, label="EOG L-R")
                    plt.xlim(0, len(views["time_eog"].flatten().numpy()) - 1)
                    plt.yticks([-100, 0, 100], fontsize=fsize - 4)
                    plt.xticks([])
                    leg = ax.legend(fontsize=fsize, bbox_to_anchor=(0.5, -0.2), frameon=False, loc='center', ncol=2)
                    # change the line width for the legend
                    for line in leg.get_lines():
                        line.set_linewidth(5.0)
                    # plt.legend(frameon=False, fontsize=12, )
                    ax.spines['top'].set_visible(False)
                    # ax.spines['left'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    ax.spines['bottom'].set_visible(False)
                    plt.ylabel("Time Signals (mV)", fontsize=fsize)
                    ax = plt.subplot(312)
                    plt.plot(this_pred["eeg"] - 0.02, alpha=0.7)
                    plt.plot(this_pred["blip_tm_eog"] + 0.02, alpha=0.7)
                    plt.plot(this_tts, color="black")
                    plt.scatter((this_pred["eeg"] != this_tts).nonzero()[0],
                                this_pred["eeg"][(this_pred["eeg"] != this_tts).nonzero()[0]],
                                marker='*', edgecolors="blue", label="Mistakes", linewidth=0.8)
                    plt.scatter((this_pred["blip_tm_eog"] != this_tts).nonzero()[0],
                                this_pred["blip_tm_eog"][(this_pred["blip_tm_eog"] != this_tts).nonzero()[0]],
                                marker='*', edgecolors="orange", label="Mistakes", linewidth=0.8)
                    plt.ylabel("Classes", fontsize=fsize)
                    ax.spines['top'].set_visible(False)
                    ax.spines['bottom'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    plt.xticks([])
                    plt.yticks([0, 1, 2, 3, 4], ["W", "REM", "N1", "N2", "N3"], fontsize=fsize - 4)
                    plt.xlim(0, len(this_pred["blip_tm_i"]))
                    ax = plt.subplot(313)
                    plt.plot(this_pred["blip_tm_i"], color="darksalmon", label="EEG-EOG")
                    plt.plot(this_tts + 0.02, color="black", label="Labels")
                    plt.scatter((this_pred["blip_tm_i"] != this_tts).nonzero()[0],
                                this_pred["blip_tm_i"][(this_pred["blip_tm_i"] != this_tts).nonzero()[0]],
                                marker='*', edgecolors="darksalmon", linewidth=0.6)
                    # plt.plot(views["time_eog"].std(dim=1))
                    plt.ylabel("Classes", fontsize=fsize)
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    plt.yticks([0, 1, 2, 3, 4], ["W", "REM", "N1", "N2", "N3"], fontsize=fsize - 4)
                    plt.xlim(0, len(this_pred["blip_tm_eeg"]))
                    leg = ax.legend(fontsize=fsize, bbox_to_anchor=(0.5, 1.1), frameon=False, loc='center', ncol=2)
                    for line in leg.get_lines():
                        line.set_linewidth(5.0)
                    plt.xticks([i * 120 for i in range((views["time_eog"].shape[0] // 120) + 1)],
                               labels=["{}".format(i) for i in range((views["time_eog"].shape[0] // 120) + 1)],
                               fontsize=fsize - 4)
                    plt.xlabel("Hours", fontsize=fsize)
                    fig.align_ylabels()
                    plt.subplots_adjust(hspace=0.4)
                    plt.savefig("/users/sista/kkontras/Documents/Sleep_Project/time_std_preds_sample_patient_1243.svg")
                    plt.show()

                # current_figure = plot_signals(views = keep_views, current_figure=current_figure, num_figures=num_figures, only="time", xlabel_hours=True)
                # current_figure = plot_signals(views = keep_views, current_figure=current_figure, num_figures=num_figures, only="fourier", xlabel_hours=True)
                current_figure = plot_signals(input_views = keep_views, current_figure=current_figure, num_figures=num_figures, only=None, xlabel_hours=True)
                # plt.xlabel("Patient {}".format(patient_num))

                # if torch.max(results["std_est"]) > 40 or np.max(results["vae_est"]) > 1.5:
                #     count_bad_segments_std += (results["std_est"] > 40).sum()
                #     count_bad_segments_vae += (results["vae_est"] > 1.5).sum()
                #     bad_patients.append(patient_num)
                #     print(bad_patients)
                #     print("Added now std: {} and vae: {}".format((results["std_est"] > 40).sum(),
                #                                                  (results["vae_est"] > 1.5).sum()))
                #     print("Total std: {} and vae {}".format(count_bad_segments_std, count_bad_segments_vae))


                # print("Mean std is {}".format(results["mean_std"]))
                # print("Mean vae is {}".format(results["mean_vae"]))
                # plt.subplot(414)
                # plt.plot(kappa_window.flatten(), color="k", label="MM", linewidth=0.7)
                # plt.plot(kappa_window_eeg.flatten(), color="b", label="EEG", linewidth=0.7)
                # plt.plot(kappa_window_eog.flatten(), color="y", label="EOG", linewidth=0.7)
                # # plt.xticks([i * 120 for i in range((hours // 120) + 1)],
                # #            labels=["{}".format(i) for i in range((hours // 120) + 1)])
                # plt.xlim(0, len(kappa_window.flatten()))
                # plt.ylabel("Kappa", fontsize=8)
                # plt.xticks([])
                # plt.yticks(fontsize=8)
                # plt.legend(prop={'size': 8})
                # plt.show()

                # plt.subplot(712)
                # plt.plot(inter_eeg_mean_floor.flatten(), label="EEG_W_Mean", linewidth=0.3)
                # # plt.xticks([i * 120 for i in range((hours // 120) + 1)],
                # #            labels=["{}".format(i) for i in range((hours // 120) + 1)])
                # plt.xlim(0,len(inter_eeg_mean_floor.flatten()))
                # plt.ylabel("EEG W Mean", fontsize=8)
                # plt.xticks([])
                # plt.yticks(fontsize=8)

                # plt.subplot(713)
                # plt.plot(inter_eeg_std_floor.flatten(), label="EEG_W_STD", linewidth=0.3)
                # # plt.xticks([i * 120 for i in range((hours // 120) + 1)],
                # #            labels=["{}".format(i) for i in range((hours // 120) + 1)])
                # plt.xlim(0,len(inter_eeg_std_floor.flatten()))
                # plt.ylabel("EEG W STD", fontsize=8)
                # plt.xticks([])
                # plt.yticks(fontsize=8)
                #
                # plt.subplot(714)
                # plt.plot(inter_eog_mean_floor.flatten(), label="EEG_W_Mean", linewidth=0.3)
                # # plt.xticks([i * 120 for i in range((hours // 120) + 1)],
                # #            labels=["{}".format(i) for i in range((hours // 120) + 1)])
                # plt.xlim(0,len(inter_eog_mean_floor.flatten()))
                # plt.ylabel("EOG W Mean", fontsize=8)
                # plt.xticks([])
                # plt.yticks(fontsize=8)
                #
                #
                # plt.subplot(715)
                # plt.plot(inter_eog_std_floor.flatten(), label="EEG_W_STD", linewidth=0.3)
                # # plt.xticks([i * 120 for i in range((hours // 120) + 1)],
                # #            labels=["{}".format(i) for i in range((hours // 120) + 1)])
                # plt.xlim(0,len(inter_eog_std_floor.flatten()))
                # plt.ylabel("EOG W STD", fontsize=8)
                # plt.xticks([])
                # plt.yticks(fontsize=8)

                # plt.subplot(412)
                # plt.plot((views_eeg_time.flatten() - views_eeg_time.flatten().mean() )/views_eeg_time.flatten().std(), label="EEG_Time", linewidth=0.3)
                # # plt.xticks([i * 120 for i in range((hours // 120) + 1)],
                # #            labels=["{}".format(i) for i in range((hours // 120) + 1)])
                # plt.xlim(0,len(views_eeg_time.flatten()))
                #
                # plt.ylabel("EEG Time", fontsize=8)
                # plt.xticks([])
                #
                # plt.subplot(413)
                # plt.plot((views_eog_time.flatten() - views_eog_time.flatten().mean() )/views_eog_time.flatten().std(), label="EOG_Time", linewidth=0.3)
                # # plt.xticks([i * 120 for i in range((hours // 120) + 1)],
                # #            labels=["{}".format(i) for i in range((hours // 120) + 1)])
                # plt.xlim(0,len(views_eog_time.flatten()))
                #
                # plt.ylabel("EOG Time", fontsize=8)
                # plt.xticks([])

                # plt.subplot(413)
                # plt.plot(inter_distance, 'b', label='high', linewidth=0.6)
                # # plt.plot(x[low_idx_inter_dist], inter_distance[low_idx_inter_dist], 'b', label='high')
                # plt.scatter(non_matches_idx, inter_distance[non_matches_idx], marker='*', edgecolors="r", label="Mistakes", linewidth=0.4)
                # plt.xticks([])
                # plt.ylabel("Distance EEG-EOG", fontsize=8)
                # plt.xlim(0,hours)

                # plt.subplot(716)
                # t = np.linspace(0, len(views_eeg) - 1, len(views_eeg))
                # # t = np.linspace(0, len(inter_eeg) - 1, len(inter_eeg))
                # f = np.linspace(0, 129 - 1, 129)
                # plt.pcolormesh(t, f, views_eeg.transpose(), shading='auto')
                # # plt.pcolormesh(t, f, inter_eeg.transpose())
                # # plt.xticks([])
                # # plt.xticks([i * 120 for i in range((hours // 120) + 1)],
                # #            labels=["{}".format(i) for i in range((hours // 120) + 1)])
                # plt.yticks(fontsize=8)
                # plt.ylabel("F EEG")
                # plt.xlabel("Hours")
                # # plt.show()
                #
                # plt.subplot(717)
                # t = np.linspace(0, len(views_eog) - 1, len(views_eog))
                # # t = np.linspace(0, len(inter_eeg) - 1, len(inter_eeg))
                # f = np.linspace(0, 129 - 1, 129)
                # plt.pcolormesh(t, f, views_eog.transpose(), shading='auto')
                # # plt.pcolormesh(t, f, inter_eeg.transpose())
                # # plt.xticks([])
                # # plt.xticks([i * 120 for i in range((hours // 120) + 1)],
                # #            labels=["{}".format(i) for i in range((hours // 120) + 1)])
                # plt.yticks(fontsize=8)
                # plt.ylabel("F EOG")
                # plt.xlabel("Hours")
                # plt.show()

                #
                # plt.subplot(413)
                # t = np.linspace(0, len(views_eog) - 1, len(views_eog))
                # # t = np.linspace(0, len(inter_eog) - 1, len(inter_eog))
                # f = np.linspace(0, 129 - 1, 129)
                # plt.pcolormesh(t, f, views_eog.transpose(), shading='auto')
                # # plt.pcolormesh(t, f, inter_eog.transpose())
                # plt.xticks([])
                # plt.ylabel("F EOG")

                # plt.subplot(413)
                # # plt.plot(ce_loss, "r", label="CE_Loss")
                # plt.plot(x[low_idx_ce], ce_loss[low_idx_ce], 'lightblue', label='high', linewidth=0.6)
                # plt.scatter(non_matches_idx, ce_loss[non_matches_idx], marker='*', edgecolors="r", label="Mistakes", linewidth=0.4)
                # # plt.xticks([i * 120 for i in range((hours // 120) + 1)],
                # #            labels=["{}".format(i) for i in range((hours // 120) + 1)])
                # plt.ylabel("CE_Loss", fontsize=8)
                # plt.xticks([])
                # plt.xlim(0,hours)

                # plt.subplot(414)
                # # x = np.linspace(0, len(match_loss) - 1, len(match_loss))
                # # plt.plot(match_loss.numpy(), "g", label="Matches")
                # # plt.plot(entropy_pred, label="Entropy")
                # plt.plot(match_loss, label="Match Loss", linewidth=0.6)
                # # plt.plot(x[low_idx_match], match_loss[low_idx_match], 'g', label='high')
                # plt.scatter(non_matches_idx, match_loss[non_matches_idx], marker='*', edgecolors="r", label="Mistakes", linewidth=0.4)
                #
                # plt.xticks([i * 120 for i in range((hours // 120) + 1)],
                #            labels=["{}".format(i) for i in range((hours // 120) + 1)])
                #
                # # plt.xticks([])
                # plt.xlim(0,hours)
                #
                # # plt.title("Patient {}".format(patient_num))
                # plt.ylabel("Align_loss", fontsize=8)
                # # # plt.ylabel("Entropy Envelope")
                #
                # # plt.subplot(414)
                # # # plt.plot(combined_loss, "k", label="Combined_Loss")
                # # plt.plot(x[low_idx_combined], combined_loss[low_idx_combined], 'k', label='high')
                # # plt.scatter(non_matches_idx, combined_loss[non_matches_idx], marker='*', edgecolors="r", label="Mistakes")
                # # plt.xticks([i * 120 for i in range((hours // 120) + 1)],
                # #            labels=["{}".format(i) for i in range((hours // 120) + 1)])
                # # plt.ylabel("Combined_Loss")
                # plt.xlabel("Hours")
                # plt.show()

                # plt.figure()
                # plt.subplot(411)
                # plt.plot(pred_plus, label="Prediction", linewidth=0.6)
                # plt.plot(target_plus, label="True Label", linewidth=0.6)
                # plt.scatter(non_matches_idx, pred_plus[non_matches_idx], marker='*', edgecolors="r", label="Mistakes", linewidth=0.6)
                # # plt.plot(non_matches_idx,"*")
                # plt.yticks([0, 1, 2, 3, 4], labels=["Wake", "N1", "REM", "N2", "N3"])
                # # plt.xticks([i * 120 for i in range((hours // 120) + 1)],
                # #            labels=["{}".format(i) for i in range((hours // 120) + 1)])
                # plt.xticks([])
                # plt.xlim(0,hours)
                #
                # # plt.legend()
                # plt.ylabel("Labels", fontsize=8)
                # # plt.xlabel("Hours")
                # plt.title("Patient {}".format(patient_num))
                #
                # plt.subplot(412)
                # plt.plot((views_eeg_time.flatten() - views_eeg_time.flatten().mean() )/views_eeg_time.flatten().std(), label="EEG_Time", linewidth=0.3)
                # # plt.xticks([i * 120 for i in range((hours // 120) + 1)],
                # #            labels=["{}".format(i) for i in range((hours // 120) + 1)])
                # plt.xlim(0,len(views_eeg_time.flatten()))
                #
                # plt.ylabel("EEG Time", fontsize=8)
                # plt.xticks([])
                #
                # plt.subplot(413)
                # plt.plot((views_eog_time.flatten() - views_eog_time.flatten().mean() )/views_eog_time.flatten().std(), label="EOG_Time", linewidth=0.3)
                # # plt.xticks([i * 120 for i in range((hours // 120) + 1)],
                # #            labels=["{}".format(i) for i in range((hours // 120) + 1)])
                # plt.xlim(0,len(views_eog_time.flatten()))
                #
                # plt.ylabel("EOG Time", fontsize=8)
                # plt.xticks([])
                #
                # plt.subplot(414)
                # # x = np.linspace(0, len(match_loss) - 1, len(match_loss))
                # # plt.plot(match_loss.numpy(), "g", label="Matches")
                # # plt.plot(entropy_pred, label="Entropy")
                # plt.plot(match_loss, label="Match Loss", linewidth=0.6)
                # # plt.plot(x[low_idx_match], match_loss[low_idx_match], 'g', label='high')
                # plt.scatter(non_matches_idx, match_loss[non_matches_idx], marker='*', edgecolors="r", label="Mistakes", linewidth=0.4)
                #
                # plt.xticks([i * 120 for i in range((hours // 120) + 1)],
                #            labels=["{}".format(i) for i in range((hours // 120) + 1)])
                #
                # # plt.xticks([])
                # plt.xlim(0,hours)
                #
                # # plt.title("Patient {}".format(patient_num))
                # plt.ylabel("Align_loss", fontsize=8)
                # # # plt.ylabel("Entropy Envelope")
                #
                # # plt.subplot(414)
                # # # plt.plot(combined_loss, "k", label="Combined_Loss")
                # # plt.plot(x[low_idx_combined], combined_loss[low_idx_combined], 'k', label='high')
                # # plt.scatter(non_matches_idx, combined_loss[non_matches_idx], marker='*', edgecolors="r", label="Mistakes")
                # # plt.xticks([i * 120 for i in range((hours // 120) + 1)],
                # #            labels=["{}".format(i) for i in range((hours // 120) + 1)])
                # # plt.ylabel("Combined_Loss")
                # plt.xlabel("Hours")
                plt.savefig("/users/sista/kkontras/Documents/Sleep_Project/experiments/Broken_Mod_Imgs/full_{}.jpg".format(patient_num))
                plt.show()

                return results
            except StopIteration:
                pass

    def find_patient_list(self, data_loader):
        # for view in data_loader.dataset:
        #     for file in data_loader.dataset[view]["dataset"]:
        # patient_list = []
        # for view in data_loader.dataset.dataset:
        #     for file in data_loader.dataset.dataset[view]["dataset"]:
        #         print(file)
        #         if file["filename"].split("/")[-1] != "empty":
        #             patient_list.append(int(file["filename"].split("/")[-1][1:5]))
        # print(patient_list)
        patient_list = [data_loader.dataset.cumulatives["files"][i]["patient_num"] for i in data_loader.dataset.cumulatives["files"]]
        patient_list = np.unique(np.array(patient_list))
        print(len(patient_list))
        print(patient_list)
        # patient_list = [ int(file["filename"].split("/")[-1][1:5]) for view in data_loader.dataset.dataset for file in data_loader.dataset.dataset[view]["dataset"] if file["filename"].split("/")[-1] != "empty"]
        # patient_list = [int(data.split("/")[-1][1:5]) for data in data_loader.dataset.dataset[0]  if
        #                 data.split("/")[-1] != "empty"]
        return list(patient_list)
    #   91,  229,  251,  351,  449,  620, 1030, 1129, 1210, 1288, 1627,
    #        1865, 2072, 2172, 2246, 2272, 2347, 2436, 2474, 2515, 2531, 2595,
    #        2847, 2908, 3338, 3435, 3455, 3761, 3906, 3931, 4007, 4074, 4578,
    #        5373, 5425, 5724]
    def _perf_measure(self, y_actual, y_hat):
        TP = 0
        FP = 0
        TN = 0
        FN = 0

        for i in range(len(y_hat)):
            if y_actual[i] == y_hat[i] == 1:
                TP += 1
            if y_hat[i] == 1 and y_actual[i] != y_hat[i]:
                FP += 1
            if y_actual[i] == y_hat[i] == 0:
                TN += 1
            if y_hat[i] == 0 and y_actual[i] != y_hat[i]:
                FN += 1

        return (TP, FP, TN, FN)

    def gather_comparisons(self, set, models, multi_fold_results, only_align_model=False, router_models=None, plot_hypnograms=True, once_whole_set=False):
        # with open("/esat/smcdata/users/kkontras/Image_Dataset/no_backup/Sleep_SHHS_2/patient_map_shhs1to2.pkl", "rb") as f:
        #     patient_map_shhs1to2 = pickle.load(f)
        if set == "Training": this_dataloader = self.data_loader.train_loader
        elif set == "Validation": this_dataloader = self.data_loader.valid_loader
        elif set == "Test": this_dataloader = self.data_loader.test_loader
        elif set == "Total": this_dataloader = self.data_loader.total_loader
        else: raise ValueError('Set should be one of the "Training", "Validation", "Test" or "Total" ')

        # train_bad_perf_eeg_patients = ['0008', '0035', '0065', '0095', '0109', '0111', '0128', '0152', '0160', '0186', '0200', '0240', '0264', '0270',
        #  '0271', '0275', '0409', '0461', '0468', '0496', '0527', '0584', '0658', '0746', '0995', '0996', '0997', '1048',
        #  '1133', '1255', '1325', '1345', '1351', '1352', '1357', '1394', '1428', '1432', '1449', '1450', '1454', '1474',
        #  '1500', '1501', '1548', '1566', '1587', '1597', '1638', '1682', '1696', '1706', '1709', '1732', '1822', '1874',
        #  '1950', '2040', '2041', '2060', '2092', '2132', '2155', '2176', '2223', '2232', '2254', '2277', '2278', '2302',
        #  '2332', '2372', '2447', '2470', '2504', '2543', '2601', '2607', '2610', '2619', '2684', '2715', '2769', '2851',
        #  '2879', '2938', '2944', '2946', '3044', '3090', '3099', '3103', '3107', '3114', '3145', '3163', '3173', '3191',
        #  '3200', '3215', '3228', '3256', '3366', '3409', '3631', '3633', '3735', '3745', '3766', '3815', '3886', '3951',
        #  '4071', '4174', '4214', '4224', '4254', '4398', '4405', '4448', '4482', '4560', '4699', '4815', '4820', '4898',
        #  '4937', '4970', '4999', '5068', '5098', '5134', '5154', '5226', '5253', '5316', '5361', '5371', '5375', '5508',
        #  '5534', '5544', '5697', '5729', '5762']

        # train_bad_perf_eeg_patients = [int(i) for i in train_bad_perf_eeg_patients]
        # this_dataloader.dataset.choose_specific_patient(train_bad_perf_eeg_patients, include_chosen=True)
        # test_bad_perf_eeg_patients= [55,2,91]
        test_bad_perf_eeg_patients= [2, 91, 229, 251, 304, 351, 377, 449, 523, 597, 620, 625, 912, 922, 1030, 1047, 1210,
                                     1229, 1243, 1282, 1346, 1480, 1498, 1518, 1544, 1725, 1816, 1865, 1967, 1986, 2077,
                                     2172, 2211, 2240, 2246, 2271, 2272, 2340, 2347, 2474, 2496, 2531, 2587, 2595, 2731,
                                     2748, 2777, 2835, 2839, 2847, 2868, 2899, 2908, 2978, 3007, 3079, 3178, 3207, 3208,
                                     3238, 3435, 3508, 3753, 3761, 3765, 3906, 3931, 4007, 4074, 4403, 4985, 5425, 5597,
                                     5632, 5724,
                                     304, 377, 597, 625, 701, 912, 1008, 1129, 1161, 1213, 1282, 1346, 1627, 1725,
                                     2072, 2077, 2240, 2271, 2272, 2388, 2419, 2436, 2474, 2515, 2587, 2611, 2731,
                                     2777, 2835, 2847, 2899, 3007, 3150, 3338, 3508, 4199, 4248, 4403, 4543, 4985,
                                     5373, 5632]

        test_bad_perf_eeg_patients= [2, 91, 1865]
        test_bad_perf_eeg_patients= [91, 819, 391, 277, 175]
        test_bad_perf_eeg_patients= [290, 598, 620, 1243, 1288, 3207, 3238, 3455, 5626, 5724, 3435, 3438]
        # test_bad_perf_eeg_patients= [1288]
        test_bad_perf_eeg_patients= [1243]
        # test_bad_perf_eeg_patients= [1047, 1243, 2172, 2246, 2340, 2531, 2595, 2777, 2908, 3178, 3207, 5425]

        this_dataloader.dataset.choose_specific_patient(test_bad_perf_eeg_patients, include_chosen=True)

        patient_list = self.find_patient_list(data_loader=this_dataloader)

        # bad_patients = []
        # count_bad_segments_std, count_bad_segments_vae = 0, 0
        for i, patient_num in enumerate(patient_list):
            if once_whole_set: patient_num="{}_whole".format(set)
            results = self.validate_specific_patient( set=set,
                models=models, data_loader=this_dataloader, device=self.device, only_align_model=only_align_model,
                patient_num=patient_num, plot_hypnograms=plot_hypnograms, plot_matches=True, return_matches=False,
                                                      router_models=router_models, once_whole_set=once_whole_set)
            for model_keys in results:
                k = "patient_{}".format(f'{patient_num:04}') if patient_num is int else patient_num
                multi_fold_results[model_keys][k] = results[model_keys]

            if once_whole_set: break

            # multi_fold_results["vae"]["patient_{}".format(f'{patient_num:04}')] = results["vae"]
            # if results["blip_tm_eog"]["acc"]<0.7:
        # print(bad_patients)
        return multi_fold_results

    def plot_comparisons(self, models, results, print_entropy=False):

        config_list = list(models.keys())
        comparisons = {"k_comparisons":{}, "f1_comparisons":{}, "acc_comparisons":{}, "entropy_comparisons":{}, "entropy_correct_comparisons":{}, "entropy_wrong_comparisons":{}}
        for conf in config_list:

            comparisons["k_comparisons"][conf] = [results[conf][i]["k"] for i in results[conf].keys()]
            comparisons["f1_comparisons"][conf] = [results[conf][i]["f1"] for i in results[conf].keys()]
            comparisons["acc_comparisons"][conf] = [results[conf][i]["acc"] for i in results[conf].keys()]
            comparisons["entropy_comparisons"][conf] = [results[conf][i]["entropy"] for i in results[conf].keys()]
            comparisons["entropy_correct_comparisons"][conf] = [results[conf][i]["entropy_correct"] for i in results[conf].keys()]
            comparisons["entropy_wrong_comparisons"][conf] = [results[conf][i]["entropy_wrong"] for i in results[conf].keys()]

        config_list = list(models.keys())
        comparisons = {"k_comparisons": {}, "f1_comparisons": {}, "acc_comparisons": {}, "entropy_comparisons": {},
                       "entropy_correct_comparisons": {}, "entropy_wrong_comparisons": {}}
        for conf in config_list:
            comparisons["k_comparisons"][conf] = [results[conf][i]["k"] for i in results[conf].keys()]
            comparisons["f1_comparisons"][conf] = [results[conf][i]["f1"] for i in results[conf].keys()]
            comparisons["acc_comparisons"][conf] = [results[conf][i]["acc"] for i in results[conf].keys()]
            comparisons["entropy_comparisons"][conf] = [results[conf][i]["entropy"] for i in results[conf].keys()]
            comparisons["entropy_correct_comparisons"][conf] = [results[conf][i]["entropy_correct"] for i in
                                                                results[conf].keys()]
            comparisons["entropy_wrong_comparisons"][conf] = [results[conf][i]["entropy_wrong"] for i in
                                                              results[conf].keys()]
        fsize = 20
        ylabel = {"k_comparisons": "Cohens Kappa", "f1_comparisons": "MF1", "acc_comparisons": "Accuracy",
                  "entropy_comparisons": "Entropy",
                  "entropy_correct_comparisons": "Entropy", "entropy_wrong_comparisons": "Entropy"}
        legend_titles = {"eeg": "EEG", "blip_tm_eeg": "CoRe-Sleep EEG", "blip_tm_i": "CoRe-Sleep EEG-EOG", }
        for metric in comparisons:
            if not print_entropy and "entropy" in metric: continue
            print(metric)
            patient_sortargs = np.argsort(comparisons[metric][config_list[0]])
            for conf in config_list:
                comparisons[metric][conf] = [comparisons[metric][conf][i] for i in patient_sortargs]
            models_colors = ["orange", "lightblue", "green", "purple", "red"]
            colors = np.array(comparisons[metric][config_list[0]]) > np.array(comparisons[metric][config_list[1]])
            colors = ["orange" if i else "lightblue" for i in colors]
            plt.figure(figsize=(25, 5))
            ax = plt.subplot(111)
            x = np.linspace(0, len(comparisons[metric][config_list[0]]) - 1, len(comparisons[metric][config_list[0]]))
            plt.xlabel("Patients", fontsize=fsize)
            plt.ylabel(ylabel[metric], fontsize=fsize)
            # plt.title("{} Comparison".format(metric), fontsize = fsize)
            for i, conf in enumerate(config_list):
                plt.plot(x, comparisons[metric][conf], 'o', markersize=11, alpha=0.8, color=models_colors[i],
                         label=legend_titles[conf])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            plt.xticks(fontsize=fsize - 4)
            plt.yticks(fontsize=fsize - 4)
            for i in range(2, 11):
                plt.axhline(i / 10, 0, len(x) - 1, linewidth=0.4, linestyle='--', color="gray", alpha=0.7)
            # for i in range(len(x)):
            #     plt.axvline(x[i], 0, 1, color=colors[i])
            plt.legend(fontsize=fsize, frameon=False, ncol=2, bbox_to_anchor=(0.38, 0.95))
            plt.savefig("/users/sista/kkontras/Documents/Sleep_Project/patient_wise_analysis_test.svg")
            plt.show()

        # colors = np.array(k_comparisons[config_list[0]]) > np.array(k_comparisons[config_list[1]])
        # colors = ["orange" if i else "lightblue" for i in colors]
        # plt.figure(figsize=(25, 5))
        # x = np.linspace(0, len(k_comparisons[config_list[0]]) - 1, len(k_comparisons[config_list[0]]))
        # plt.xlabel("Patient")
        # plt.ylabel("Cohens Kappa")
        # plt.title("K Comparison")
        # plt.plot(x, k_comparisons[config_list[0]], 'o', color='orange', label=config_list[0])
        # plt.plot(x, k_comparisons[config_list[1]], 'o', color='lightblue', label=config_list[1])
        # # for i in range(len(x)):
        # #     plt.axvline(x[i], 0, 1, color=colors[i])
        # plt.legend()
        # plt.show()
        #
        # colors = np.array(acc_comparisons[config_list[0]]) > np.array(acc_comparisons[config_list[1]])
        # colors = ["orange" if i else "lightblue" for i in colors]
        # plt.figure(figsize=(25, 5))
        # x = np.linspace(0, len(acc_comparisons[config_list[0]]) - 1, len(acc_comparisons[config_list[0]]))
        # plt.xlabel("Patient")
        # plt.ylabel("Accuracy")
        # plt.title("Accuracy Comparison")
        # plt.plot(x, acc_comparisons[config_list[0]], 'o', color='orange', label=config_list[0])
        # plt.plot(x, acc_comparisons[config_list[1]], 'o', color='lightblue', label=config_list[1])
        # # for i in range(len(x)):
        # #     plt.axvline(x[i], 0, 1, color=colors[i])
        #
        # plt.legend()
        # plt.show()
        #
        # colors = np.array(entropy_comparisons[config_list[0]]) > np.array(entropy_comparisons[config_list[1]])
        # colors = ["orange" if i else "lightblue" for i in colors]
        # plt.figure(figsize=(25, 5))
        # x = np.linspace(0, len(entropy_comparisons[config_list[0]]) - 1, len(entropy_comparisons[config_list[0]]))
        # plt.xlabel("Patient")
        # plt.ylabel("Entropy")
        # plt.title("Entropy Comparison")
        # plt.plot(x, entropy_comparisons[config_list[0]], 'o', color='orange', label=config_list[0])
        # plt.plot(x, entropy_comparisons[config_list[1]], 'o', color='lightblue', label=config_list[1])
        # # for i in range(len(x)):
        # #     plt.axvline(x[i], 0, 1, color=colors[i])
        # plt.legend()
        # plt.show()
        #
        # colors = np.array(entropy_correct_comparisons[config_list[0]]) > np.array(entropy_correct_comparisons[config_list[1]])
        # colors = ["orange" if i else "lightblue" for i in colors]
        # plt.figure(figsize=(25, 5))
        # x = np.linspace(0, len(entropy_correct_comparisons[config_list[0]]) - 1, len(entropy_correct_comparisons[config_list[0]]))
        # plt.xlabel("Patient")
        # plt.ylabel("Entropy_Correct")
        # plt.title("Entropy_Correct Comparison")
        # plt.plot(x, entropy_correct_comparisons[config_list[0]], 'o', color='orange', label=config_list[0])
        # plt.plot(x, entropy_correct_comparisons[config_list[1]], 'o', color='lightblue', label=config_list[1])
        # # for i in range(len(x)):
        # #     plt.axvline(x[i], 0, 1, color=colors[i])
        # plt.legend()
        # plt.show()
        #
        # colors = np.array(entropy_wrong_comparisons[config_list[0]]) > np.array(entropy_wrong_comparisons[config_list[1]])
        # colors = ["orange" if i else "lightblue" for i in colors]
        # plt.figure(figsize=(25, 5))
        # x = np.linspace(0, len(entropy_wrong_comparisons[config_list[0]]) - 1, len(entropy_wrong_comparisons[config_list[0]]))
        # plt.xlabel("Patient")
        # plt.ylabel("Entropy_Correct")
        # plt.title("Entropy_Correct Comparison")
        # plt.plot(x, entropy_wrong_comparisons[config_list[0]], 'o', color='orange', label=config_list[0])
        # plt.plot(x, entropy_wrong_comparisons[config_list[1]], 'o', color='lightblue', label=config_list[1])
        # # for i in range(len(x)):
        # #     plt.axvline(x[i], 0, 1, color=colors[i])
        # plt.legend()
        # plt.show()

    def load_results(self, filename:str, prev_results:dict={}):
        metrics_file = open(filename, "rb")
        results = pickle.load(metrics_file)
        prev_results.update(results)
        print("Available results:")
        for i in prev_results: print(i)
        return prev_results

    def save_results(self, filename:str, results:dict):

        results = {i: dict(v) for i, v in results.items()} #from defaultdict to dict
        try:
            file = open(filename, "wb")
            pickle.dump(results, file)
            file.close()
            print("Results saved!")
        except:
            print("Error on saving metrics")

    def get_predictions_time_series(self, model, views, inits, skip_modality):
        """
        This is a function to exploit the fact that time series are not always continuous. We dont want to correlate signals from different patients/recordings just because the batch is not fully dividing the number of recording imgs.
        :param views: List of tensors, data views/modalities
        :param inits: Tensor indicating with value one, when there incontinuities.
        :return: predictions of the self.model on the batch
        """
        this_inits = inits[list(inits.keys())[0]]
        this_view = views[list(inits.keys())[0]]
        skip_modality_temp = torch.zeros(this_view.shape[0], this_view.shape[1])
        skip_modality_temp[skip_modality["stft_eeg"] == 1] = 1
        skip_modality_temp[skip_modality["stft_eog"] == 1] = 2

        a = ((skip_modality_temp - skip_modality_temp.mean(dim=1).unsqueeze(dim=1).repeat(1, 21)).mean(dim=1) != 0).nonzero(as_tuple=True)[0]
        for batch_idx in a:
            for i in range(len(skip_modality_temp[batch_idx]) - 1):
                if skip_modality_temp[batch_idx][i] != skip_modality_temp[batch_idx][i + 1]:
                    this_inits[batch_idx][i] = 1
                    this_inits[batch_idx][i + 1] = 1

        inits_sum_batch = (this_inits.sum(dim=1) > 1).nonzero(as_tuple=True)[0]
        if len(inits_sum_batch) > 0:
            batch, outer = this_view.shape[0], this_view.shape[1]
            batch_idx_checked = torch.ones(batch, dtype=torch.bool)
            pred ={"preds":{"combined": torch.zeros(batch * outer, 5).to(this_view.device)}}
            for batch_idx in inits_sum_batch:
                ones_idx = (this_inits[batch_idx] > 0).nonzero(as_tuple=True)[0]
                if (ones_idx[0] + 1 == ones_idx[1]  ): #and ones_idx[0]!=0 and ones_idx[1]!= len(inits[idx])
                    if ones_idx[0] == 0:
                        pred_split_0 = model({view: views[view][batch_idx, ones_idx[0]].unsqueeze(dim=0).unsqueeze(dim=1) for view in views}, skip_modality={view: skip_modality[view][batch_idx, ones_idx[0]].unsqueeze(dim=0).unsqueeze(dim=1) for view in views})
                    else:
                        pred_split_0 = model({view: views[view][batch_idx, :ones_idx[0] + 1].unsqueeze(dim=0) for view in views}, skip_modality={view: skip_modality[view][batch_idx, :ones_idx[0] + 1].unsqueeze(dim=0) for view in views})

                    if ones_idx[1] == len(this_inits[batch_idx]):
                        pred_split_1 = model({view: views[view][batch_idx, -1].unsqueeze(dim=0).unsqueeze(dim=1) for view in views}, skip_modality={view: skip_modality[view][batch_idx, -1].unsqueeze(dim=0).unsqueeze(dim=1) for view in views})
                    else:
                        pred_split_1 = model({view: views[view][batch_idx, ones_idx[1]:].unsqueeze(dim=0) for view in views}, skip_modality={view: skip_modality[view][batch_idx, ones_idx[1]:].unsqueeze(dim=0) for view in views})

                    pred["preds"]["combined"][batch_idx * outer:(batch_idx + 1) * outer] = torch.cat([pred_split_0["preds"]["combined"], pred_split_1["preds"]["combined"]], dim=0)
                else:
                    pred["preds"]["combined"][batch_idx * outer:(batch_idx + 1) * outer] = model({view: views[view][batch_idx].unsqueeze(dim=0) for view in views}, skip_modality={view: skip_modality[view][batch_idx].unsqueeze(dim=0) for view in views})["preds"]["combined"]

                batch_idx_checked[batch_idx] = False
            pred["preds"]["combined"][batch_idx_checked.repeat_interleave(outer)] = model({view: views[view][batch_idx_checked] for view in views}, skip_modality={view: skip_modality[view][batch_idx_checked] for view in views})["preds"]["combined"]

        else:
            pred = model(views, skip_modality=skip_modality)

        return pred

    def plot_spider(self, results):

        categories = ['W', 'N1', 'N2',
                      'N3', 'REM']

        fig = go.Figure()

        fig.add_trace(go.Scatterpolar(
            r=[93.78, 47.57, 89.74, 86.05, 91.38],
            theta=categories,
            fill='toself',
            name='BLIP Limited'
        ))
        fig.add_trace(go.Scatterpolar(
            r=[90.96, 44.9, 86.84, 85.86, 85.05],
            theta=categories,
            fill='toself',
            name='EEG'
        ))

        fig.add_trace(go.Scatterpolar(
            r=[89.03, 23.91, 83.25, 76.23, 87.63],
            theta=categories,
            fill='toself',
            name='EOG'
        ))


        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )),
            legend= dict(
                font = dict(size=30),
                ),
            title = dict(text = "Mode Comparison on BLIP", font = dict(size=30)),

            showlegend=True
        )

        fig.show()



