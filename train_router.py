#%%

import copy
import sys

import einops
import numpy
import numpy as np
import torch

sys.exc_info()
from utils.config import process_config
from datasets.sleepset import *
from graphs.models.attention_models.windowFeature_base import *
from graphs.models.attention_models.windowFeature_base import SleepEnc_Merged_EEG_EOG
from sklearn.metrics import f1_score, cohen_kappa_score, roc_auc_score, confusion_matrix
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from utils.deterministic_pytorch import deterministic
# import umap
from scipy.stats import entropy
import os
from torchdistill.core.forward_hook import ForwardHookManager
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn as sns
import pandas as pd

print("Done Loading Libraries")


#%%

def sleep_load_encoder(encoders):
    encs = []
    for num_enc in range(len(encoders)):
        if encoders[num_enc]["model"] == "TF":
            layers = ["huy_pos_inner", "inner_att", "aggregation_att_contx_inner", "huy_pos_outer", "outer_att"]
            enc = Multi_Transformer(128, inner= 29, outer = 21, modalities=1, heads=8,
                                 layers = layers, num_layers=4, pos = False)
        else:
            enc_class = globals()[encoders[num_enc]["model"]]
            args = encoders[num_enc]["args"]
            enc = enc_class(args = args)
            enc = nn.DataParallel(enc, device_ids=[torch.device(0)])

        if encoders[num_enc]["pretrainedEncoder"]["use"]:
            print("Loading encoder from {}".format(encoders[num_enc]["pretrainedEncoder"]["dir"]))
            checkpoint = torch.load(encoders[num_enc]["pretrainedEncoder"]["dir"])
            enc.load_state_dict(checkpoint["encoder_state_dict"])
        encs.append(enc)
    return encs

def load_models(config, device, checkpoint, only_model=False):

    model_class = globals()[config.model.model_class]
    # config.pretrainedEncoder = [False]
    enc = sleep_load_encoder(encoders=config.model.encoders)
    model = model_class(enc, args = config.model.args)
    # model = model.to('cpu')
    # model = nn.DataParallel(model, device_ids='cpu')
    model = model.to(device)
    model = nn.DataParallel(model, device_ids=[torch.device(i) for i in config.gpu_device])
    # print(device)
    # model = nn.DataParallel(model, device="cpu")

    #
    if only_model:
        return model

    # config.pretrainedEncoder = [True]
    # enc = sleep_load_encoder(encoder_models=config.encoder_models,pretrainedEncoder=config.pretrainedEncoder,save_dir_encoder=config.savetrainedEncoder)
    # best_model = model_class(enc, channel = config.channel)
    # best_model = best_model.to(device)
    # best_model = nn.DataParallel(best_model, device_ids=[torch.device(i) for i in config.gpu_device])

    best_model = copy.deepcopy(model)
    # best_model = best_model.to('cpu')
    # best_model = nn.DataParallel(best_model, device_ids='cpu')
    # model.load_state_dict(checkpoint["model_state_dict"])
    best_model.load_state_dict(checkpoint["best_model_state_dict"])

    return model, best_model

def find_patient_list(data_loader):
    patient_list = [int(data.split("/")[-1][1:5]) for data in data_loader.dataset.dataset[0] if data.split("/")[-1]!="empty"]
    return patient_list

def perf_measure(y_actual, y_hat):
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

def print_perf(model_name, patient_num, preds, tts, multiclass=True):
    if len(preds.shape)==2:
        preds_eeg = preds.argmax(-1)
    else:
        preds_eeg = preds

    test_acc = np.equal(tts, preds_eeg).sum() / len(tts)
    test_f1 = f1_score(preds_eeg, tts) if not multiclass else f1_score(preds_eeg, tts, average="macro")
    test_perclass_f1 = f1_score(preds_eeg, tts) if not multiclass else f1_score(preds_eeg, tts, average=None)
    test_k = cohen_kappa_score(tts, preds_eeg)
    test_auc = roc_auc_score(tts, preds_eeg) if not multiclass else 0
    test_conf = confusion_matrix(tts, preds_eeg)
    tp, fp, tn, fn = perf_measure(tts, preds_eeg)
    test_spec = tn / (tn + fp) if (tn + fp) != 0 else 0
    test_sens = tp / (tp + fn) if (tp + fn) != 0 else 0
    print("{} Patient {} has acc: {}, f1: {}, k:{} and f1_per_class: {}".format(model_name, patient_num,
                                                                             round(test_acc * 100, 1),
                                                                             round(test_f1 * 100, 1),
                                                                             round(test_k, 3),
                                                                             np.round(test_perclass_f1 * 100,
                                                                                      1)))
    return test_k

def get_performance_windows(preds, tts, print_it=True, window=40, type="f1"):

    tts_unfolded = torch.from_numpy(tts).unfold(0, window, window).numpy()
    preds_unfolded = torch.from_numpy(preds).unfold(0, window, window).numpy()

    if type == "accuracy":
        perf_window = np.array([np.equal(tts_unfolded[i], preds_unfolded[i]).sum() / len(tts_unfolded[i]) for i in range(tts_unfolded.shape[0])])
    elif type=="k":
        perf_window = np.array([cohen_kappa_score(tts_unfolded[i], preds_unfolded[i]) for i in range(tts_unfolded.shape[0])])
    elif type=="f1":
        perf_window = np.array([f1_score(tts_unfolded[i], preds_unfolded[i], average="macro") for i in range(tts_unfolded.shape[0])])
    else:
        raise ValueError("This type of performance does not extst, 'accuracy', 'k' and 'f1'")
    perf_window[perf_window != perf_window] = 1
    if print_it:
        for i in perf_window: print("{:.3f}".format(i), end=" ")
        print()
    perf_window = perf_window
    return perf_window

def get_windows(input, window=40):
    input_unfolded = torch.from_numpy(input).unfold(0, window, window).numpy()
    if (input_unfolded != input_unfolded).any(): print("Τhere are nan")
    input_unfolded = input_unfolded.mean(axis=-1)
    # input_unfolded = input_unfolded.repeat(window)
    return input_unfolded

def change_numbers_preds(preds, tts, argmax=True):
    if argmax: pred_plus = copy.deepcopy(preds).argmax(-1)
    else: pred_plus = copy.deepcopy(preds)
    pred_plus[pred_plus == 4] = 5
    pred_plus[pred_plus == 3] = 4
    pred_plus[pred_plus == 2] = 3
    pred_plus[pred_plus == 5] = 2

    target_plus = copy.deepcopy(tts)
    target_plus[target_plus == 4] = 5
    target_plus[target_plus == 3] = 4
    target_plus[target_plus == 2] = 3
    target_plus[target_plus == 5] = 2

    return pred_plus, target_plus

def find_matches(pred_plus, target_plus):

    non_matches = (pred_plus != target_plus).astype(int)
    non_matches_idx = non_matches.nonzero()[0]
    return non_matches_idx

def print_max_perf(predictors, performs, tts, window_floor):
    max_preds = []
    for i in range(len(performs[0])):
        max_mod = np.argmax(np.array([perf[i] for perf in performs]))
        max_preds.append(predictors[max_mod][i*window_floor:(i+1)*window_floor])
    max_preds = np.array(max_preds).flatten()

    multiclass = True
    model_name = "Max"
    test_acc = np.equal(tts, max_preds).sum() / len(tts)
    test_f1 = f1_score(max_preds, tts) if not multiclass else f1_score(max_preds, tts, average="macro")
    test_perclass_f1 = f1_score(max_preds, tts) if not multiclass else f1_score(max_preds, tts, average=None)
    test_k = cohen_kappa_score(tts, max_preds)
    test_auc = roc_auc_score(tts, max_preds) if not multiclass else 0
    test_conf = confusion_matrix(tts, max_preds)
    tp, fp, tn, fn = perf_measure(tts, max_preds)
    test_spec = tn / (tn + fp) if (tn + fp) != 0 else 0
    test_sens = tp / (tp + fn) if (tp + fn) != 0 else 0
    print("{} has acc: {}, f1: {}, k:{} and f1_per_class: {}".format(model_name,
                                                                             round(test_acc * 100, 1),
                                                                             round(test_f1 * 100, 1),
                                                                             round(test_k, 3),
                                                                             np.round(test_perclass_f1 * 100,
                                                                                      1)))
    return test_k, max_preds

def routing_predictor_std(t_eeg, t_eog, views_eeg_time, views_eog_time, predictors):

    #STD Router
    print(views_eeg_time.shape)
    views_eeg_time_hour_std = torch.from_numpy(views_eeg_time).unfold(0,21,21).flatten(start_dim=1).numpy().std(axis=-1)
    views_eog_time_hour_std = torch.from_numpy(views_eog_time).unfold(0,21,21).flatten(start_dim=1).numpy().std(axis=-1)

    # views_eeg_time_hour_std = torch.from_numpy(views_eeg_time_hour_std).numpy()
    # views_eog_time_hour_std = torch.from_numpy(views_eog_time_hour_std).unfold(0,21,21).numpy()

    skip_eeg, skip_eog = np.zeros(len(views_eeg_time_hour_std)), np.zeros(len(views_eog_time_hour_std))

    skip_eeg[views_eeg_time_hour_std.squeeze()> t_eeg] = 1
    skip_eog[views_eog_time_hour_std.squeeze()> t_eog] = 2
    skip_flags = skip_eeg + skip_eog
    skip_flags[skip_flags==3]=0
    counter, router_preds = 0, []
    window_floor = 21
    for i, v in enumerate(np.array(skip_flags).astype(int)):
        router_pred = np.array([np.array(predictors)[v][i*window_floor:(i+1)*window_floor]])
        router_preds.append(router_pred)
    router_preds = np.array(router_preds)
    return router_preds, np.array(skip_flags)

def routing_predictor_zero_crossing(t_eeg, t_eog, views_eeg_time, views_eog_time, predictors):

    views_eeg_time_hour_zero_crossings = einops.rearrange(torch.from_numpy(views_eeg_time).unfold(0,21,21), "b time outer -> b (outer time)")
    views_eeg_time_hour_zero_crossings = einops.rearrange(einops.rearrange(views_eeg_time_hour_zero_crossings, "b t -> t b") - views_eeg_time_hour_zero_crossings.mean(dim=1), "b t -> t b")
    views_eeg_time_hour_zero_crossings = np.array([len(numpy.where(numpy.diff(numpy.sign(views_eeg_time_hour_zero_crossings[i])))[0]) for i in range(0,len(views_eeg_time_hour_zero_crossings))])/21

    views_eog_time_hour_zero_crossings = einops.rearrange(torch.from_numpy(views_eog_time).unfold(0,21,21), "b time outer -> b (outer time)")
    views_eog_time_hour_zero_crossings = einops.rearrange(einops.rearrange(views_eog_time_hour_zero_crossings, "b t -> t b") - views_eog_time_hour_zero_crossings.mean(dim=1), "b t -> t b")
    views_eog_time_hour_zero_crossings = np.array([len(numpy.where(numpy.diff(numpy.sign(views_eog_time_hour_zero_crossings[i])))[0]) for i in range(0,len(views_eog_time_hour_zero_crossings))])/21

    # print(views_eeg_time_hour_zero_crossings.max())
    # print(views_eeg_time_hour_zero_crossings.min())
    #
    # print(views_eog_time_hour_zero_crossings.max())
    # print(views_eog_time_hour_zero_crossings.min())

    skip_eeg, skip_eog = np.zeros(len(views_eeg_time_hour_zero_crossings)), np.zeros(len(views_eog_time_hour_zero_crossings))

    skip_eeg[views_eeg_time_hour_zero_crossings > t_eeg] = 1
    skip_eog[views_eog_time_hour_zero_crossings > t_eog] = 2
    skip_flags = skip_eeg + skip_eog
    skip_flags[skip_flags==3]=0
    counter, router_preds = 0, []
    window_floor = 21
    for i, v in enumerate(np.array(skip_flags).astype(int)):
        router_pred = np.array([np.array(predictors)[v][i*window_floor:(i+1)*window_floor]])
        router_preds.append(router_pred)
    router_preds = np.array(router_preds)
    return router_preds, np.array(skip_flags)
def routing_predictor_final_reps(t_eeg, t_eog, final_reps, predictors):

    a = torch.from_numpy(copy.deepcopy(final_reps)).squeeze()
    b = torch.einsum('bomf,bomf->mb', a, a)

    skip_eeg, skip_eog = np.zeros(len(b[0])), np.zeros(len(b[0]))

    skip_eeg[b[0] > t_eeg] = 1
    skip_eog[b[1] > t_eog] = 2
    skip_flags = skip_eeg + skip_eog
    skip_flags[skip_flags==3]=0
    counter, router_preds = 0, []
    window_floor = 21
    for i, v in enumerate(np.array(skip_flags).astype(int)):
        router_pred = np.array([np.array(predictors)[v][i*window_floor:(i+1)*window_floor]])
        router_preds.append(router_pred)
    router_preds = np.array(router_preds)
    return router_preds, np.array(skip_flags)

def test_metrics(test_preds, tts):
    results = {}
    results["acc"] = np.equal(tts, test_preds).sum() / len(tts)
    results["f1"] = f1_score(test_preds, tts, average="macro")
    results["f1_perclass"] = f1_score(test_preds, tts, average=None)
    results["k"] = cohen_kappa_score(tts, test_preds)
    results["conf"] = confusion_matrix(tts, test_preds)
    tp, fp, tn, fn = perf_measure(tts, test_preds)
    results["spec"] = tn / (tn + fp) if (tn + fp)!=0 else 0
    results["sens"] = tp / (tp + fn) if (tp + fn)!=0 else 0
    return results

print("Done Loading Functions")

#%%

multimodal_merged_config_name = "./configs/shhs/multi_modal/eeg_eog/established_models/fourier_transformer_eeg_eog_mat_merged_rpos_adv.json"
multimodal_config_name = "./configs/shhs/multi_modal/eeg_eog/established_models/fourier_transformer_eeg_eog_mat_BIOBLIP_lossw.json"
eeg_config_name = "./configs/shhs/single_channel/fourier_transformer_cls_eeg_mat_adv.json"
eog_config_name = "./configs/shhs/single_channel/fourier_transformer_cls_eog_mat.json"
emg_config_name = "./configs/shhs/single_channel/fourier_transformer_cls_emg_mat.json"
router_config_name = "./configs/shhs/router/router_fourier_tf_eeg_eog.json"

multimodal_config = process_config(multimodal_config_name, False)
multimodal_merged_config = process_config(multimodal_merged_config_name, False)
eeg_config = process_config(eeg_config_name, False)
eog_config = process_config(eog_config_name, False)
emg_config = process_config(emg_config_name, False)
router_config = process_config(router_config_name, False)

multimodal_config.data_view_dir = [
            {"list_dir" : "patient_mat_list.txt", "data_type": "stft", "mod": "eeg", "num_ch": 1},
            {"list_dir" :"patient_eog_mat_list.txt", "data_type": "stft", "mod": "eog", "num_ch": 1},
            {"list_dir" : "patient_mat_list.txt", "data_type": "time", "mod": "eeg", "num_ch": 1},
            {"list_dir" : "patient_eog_mat_list.txt", "data_type": "time", "mod": "eog", "num_ch": 1},
        ]

#%%

device = "cuda:0"
# device = "cpu"
multimodal_config.test_batch_size=256
#Load the models
checkpoint_multimodal = torch.load(multimodal_config.model.save_dir, map_location="cpu")
checkpoint_multimodal_merged = torch.load(multimodal_merged_config.model.save_dir, map_location="cpu")
checkpoint_eeg = torch.load(eeg_config.model.save_dir, map_location="cpu")
checkpoint_eog = torch.load(eog_config.model.save_dir, map_location="cpu")
checkpoint_emg = torch.load(emg_config.model.save_dir, map_location="cpu")
checkpoint_router = torch.load(router_config.model.save_dir, map_location="cpu")
dataloader = globals()[multimodal_config.dataloader_class]
data_loader = dataloader(config=multimodal_config)
data_loader.load_metrics_ongoing(checkpoint_multimodal["metrics"])
data_loader.weights = checkpoint_multimodal["logs"]["weights"]

_, best_model_multimodal = load_models(config=multimodal_config, device=device, checkpoint=checkpoint_multimodal)
_, best_model_multimodal_merged = load_models(config=multimodal_merged_config, device=device, checkpoint=checkpoint_multimodal_merged)
_, best_model_eeg = load_models(config=eeg_config, device=device, checkpoint=checkpoint_eeg)
_, best_model_eog = load_models(config=eog_config, device=device, checkpoint=checkpoint_eog)
_, best_model_emg = load_models(config=emg_config, device=device, checkpoint=checkpoint_emg)
_, best_model_router = load_models(config=router_config, device=device, checkpoint=checkpoint_router)


with torch.no_grad():
    best_model_multimodal.eval()
    best_model_eeg.eval()
    best_model_eog.eval()
    # best_model_emg.eval()
    best_model_router.eval()

    tts, preds, matches, inits, views_eeg, views_eog, views_emg, inter_eeg, inter_eog = [], [], [], [], [], [], [], [], []
    views_eeg_time, views_eog_time, views_emg_time = [], [], []
    router_output, std_choices = [], []
    final_reps = []
    preds_eeg, preds_eog, preds_emg, preds_merged = [], [], [], []
    patient_list_test = [2, 36, 51, 53, 55, 78, 91, 108, 113, 119, 120, 139, 170, 193, 202, 229, 231, 251, 289, 304, 320, 324, 344, 351, 377, 378, 385, 442, 449, 450, 474, 512, 523, 526, 597, 607, 620, 625, 630, 712, 717, 718, 912, 922, 937, 972, 977, 981, 1008, 1030, 1047, 1049, 1058, 1098, 1138, 1150, 1161, 1184, 1198, 1210, 1213, 1229, 1235, 1240, 1243, 1282, 1288, 1300, 1310, 1311, 1319, 1337, 1340, 1346, 1392, 1414, 1424, 1442, 1445, 1459, 1461, 1469, 1480, 1496, 1498, 1513, 1518, 1532, 1544, 1558, 1565, 1585, 1611, 1624, 1627, 1653, 1674, 1725, 1786, 1813, 1816, 1823, 1837, 1838, 1842, 1865, 1873, 1895, 1898, 1967, 1972, 1986, 2023, 2044, 2073, 2077, 2078, 2080, 2090, 2110, 2128, 2129, 2143, 2153, 2160, 2169, 2172, 2173, 2211, 2213, 2228, 2236, 2240, 2246, 2271, 2272, 2287, 2292, 2340, 2347, 2354, 2363, 2388, 2393, 2409, 2417, 2462, 2474, 2481, 2484, 2496, 2531, 2575, 2586, 2587, 2594, 2595, 2604, 2608, 2621, 2628, 2655, 2697, 2731, 2746, 2748, 2771, 2777, 2792, 2795, 2821, 2835, 2839, 2844, 2847, 2852, 2866, 2868, 2873, 2876, 2885, 2891, 2899, 2908, 2937, 2953, 2978, 3011, 3021, 3038, 3079, 3140, 3157, 3176, 3178, 3183, 3206, 3207, 3208, 3238, 3269, 3320, 3343, 3356, 3386, 3407, 3428, 3431, 3435, 3455, 3466, 3508, 3567, 3598, 3656, 3665, 3753, 3761, 3765, 3851, 3906, 3931, 3958, 4007, 4074, 4113, 4151, 4250, 4290, 4324, 4335, 4337, 4351, 4364, 4403, 4427, 4454, 4535, 4540, 4643, 4651, 4679, 4697, 4728]

    # patient_list = [149, 177, 494, 723, 787, 832, 885, 1178, 1180, 1202, 1224, 1336, 1444, 1506, 1521, 1549, 1602, 1675, 1971, 2395, 2519, 2780, 2846, 2930, 4003, 4011, 4057, 4132, 4316, 4951, 5049, 5179, 5368]
    # data_loader.valid_loader.dataset.choose_specific_patient(patient_list)
    data_loader.test_loader.dataset.choose_specific_patient(patient_list_test)
    for batch_idx, (data, target, init, _) in tqdm(enumerate(data_loader.test_loader)):
        views = [data[i].float().to(device) for i in range(len(data))]
        target = target.to(device)

        output = best_model_multimodal(views, return_final_reps=True)
        output_merged = best_model_multimodal_merged(views)
        output_eeg = best_model_eeg([views[0]])
        output_eog = best_model_eog([views[1]])
        router_output.append(best_model_router(views)["preds"]["combined"].cpu())
        final_reps.append(output["final_reps"].cpu())


        tts.append(target.cpu())
        preds.append(output["preds"]["combined"].cpu())
        # preds.append(output)
        preds_merged.append(output_merged["preds"]["combined"].cpu())
        preds_eeg.append(output_eeg.cpu())
        preds_eog.append(output_eog.cpu())

        views_eeg.append(views[0].cpu())
        views_eog.append(views[1].cpu())
        views_eeg_time.append(views[2].cpu())
        views_eog_time.append(views[3].cpu())

    tts = torch.cat(tts).cpu().numpy().flatten()
    preds = torch.cat(preds).cpu().numpy()
    final_reps = torch.cat(final_reps).cpu().numpy()
    preds_eeg = torch.cat(preds_eeg).cpu().numpy()
    preds_eog = torch.cat(preds_eog).cpu().numpy()
    preds_merged = torch.cat(preds_merged).cpu().numpy()
    router_output = torch.cat(router_output).cpu().numpy()


    views_eeg = torch.cat(views_eeg).cpu().numpy()
    views_eog = torch.cat(views_eog).cpu().numpy()

    views_eeg_time = torch.cat(views_eeg_time).cpu().flatten(start_dim=0,end_dim=1).squeeze().numpy()
    views_eog_time = torch.cat(views_eog_time).cpu().flatten(start_dim=0,end_dim=1).squeeze().numpy()

