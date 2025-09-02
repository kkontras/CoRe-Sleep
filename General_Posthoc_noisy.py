import sys
import os

sys.exc_info()
os.chdir('/users/sista/kkontras/Documents/Sleep_Project/')

import numpy as np
from posthoc.Helpers.Helper_Importer import Importer
from posthoc.Helpers.Helper_Validator import Validator
from posthoc.Helpers.Helper_LogsPlotter import LogsPlotter
from posthoc.Helpers.Helper_Generate_n_Compare import Generate_n_Compare

config_list = [

    # "./configs/paper_finals/shhs/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_aligninner_trial1.json",
    # "./configs/paper_finals/shhs/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_aligninner_trial2.json",
    # "./configs/paper_finals/shhs/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_aligninner_trial1_pt.json",
    # "./configs/paper_finals/shhs/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_aligninner_trial2_pt.json",

    # "./configs/paper_finals/shhs/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_nomult_aligninner_trial1.json",
    # "./configs/paper_finals/shhs/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_nomult_aligninner_trial1_pt.json",
    "./configs/paper_finals/shhs/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_nomult_aligninner_trial2_pt.json",

    # "./configs/paper_finals/shhs/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_Late_aligninner_trial1.json",
    # "./configs/paper_finals/shhs/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_Late_aligninner_trial2.json",
    # "./configs/paper_finals/shhs/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_Late_aligninner_trial1_pt.json",
    # "./configs/paper_finals/shhs/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_Late_aligninner_trial2_pt.json",

    # "./configs/paper_finals/shhs/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_Late_nomult_aligninner_trial1.json",
    # "./configs/paper_finals/shhs/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_Late_nomult_aligninner_trial1_pt.json",
    # "./configs/paper_finals/shhs/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_Late_nomult_aligninner_trial2_pt.json",

    # "./configs/paper_finals/shhs/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_Early_aligninner_v2_trial1.json",
    # "./configs/paper_finals/shhs/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_Early_aligninner_v2_trial1_pt.json",
    # "./configs/paper_finals/shhs/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_Early_aligninner_v2_trial2_pt.json",

    # "./configs/paper_finals/shhs/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_Early_nomult_aligninner_trial1.json",
    # "./configs/paper_finals/shhs/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_Early_nomult_aligninner_trial1_pt.json",
    # "./configs/paper_finals/shhs/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_Early_nomult_aligninner_trial2_pt.json",

    # "./configs/paper_finals/shhs/fourier_transformer_eeg_drop03_trial1.json",
    # "./configs/paper_finals/shhs/fourier_transformer_eeg_drop03_trial1_pt.json",
    # "./configs/paper_finals/shhs/fourier_transformer_eeg_drop03_trial2_pt.json",
    #
    # "./configs/paper_finals/shhs/fourier_transformer_eog_trial1.json",
    # "./configs/paper_finals/shhs/fourier_transformer_eog_trial1_pt.json",
    # "./configs/paper_finals/shhs/fourier_transformer_eog_trial2_pt.json",
]


# device = "cpu"
device = "cuda:0"
a =         [
        # "fourier_transformer_cls_eeg_eog_BIOBLIP_1s_twomode.pth.tar",
        # "fourier_transformer_cls_eeg_eog_BIOBLIP_1s_twomode_sepouter_al1.pth.tar",
        # "fourier_transformer_cls_eeg_eog_BIOBLIP_1s_twomode_t.pth.tar",
        # "fourier_transformer_cls_eeg_eog_BIOBLIP_1s_twomode_tt_al1.pth.tar",
        # "fourier_transformer_cls_eeg_eog_BIOBLIP_1s_twomode_tt.pth.tar"
]
multi_fold_results, count = {}, 0
mm_acc, mm_acc_std, mm_f1, mm_f1_std, mm_k, mm_k_std = [], [], [], [], [], []
for i, config_name in enumerate(config_list):
    for fold in range(1):

        importer = Importer(config_name=config_name, device=device, fold=fold)

        importer.load_checkpoint()
        # plotter = LogsPlotter(config=importer.config, logs=importer.checkpoint["logs"])

        # multi_fold_results = importer.print_progress(multi_fold_results=multi_fold_results, latex_version=False)
        # importer.print_progress_aggregated(multi_fold_results=multi_fold_results)

        # plotter.sleep_plot_losses()
        # plotter.sleep_plot_performance()

        #Import models
        # model = importer.get_model(return_model="untrained_model")
        # running_model = importer.get_model(return_model="running_model")
        best_model = importer.get_model(return_model="best_model")

        # best_model.module.shared_pred = True

        importer.change_config(attr="dataset.filter_windows", value={
                                            "train": {"use_type": False, "include_skipped": False},
                                            "val": {"use_type": False, "include_skipped": False},
                                            # "test": {"use_type": "full", "skip_skips": True, "whole_patient": True, "std_threshold": 41, "perc_threshold": 0.4}})
                                            "test": {"use_type": "include_only_skipped", "skip_skips":True, "whole_patient": True, "std_threshold": 41, "perc_threshold": 0.4}})
                                            # "test": {"use_type": "include_only_skipped", "skip_skips":True, "whole_patient": False, "std_threshold": 41, "perc_threshold": 0.4}})

        importer.change_config(attr="dataset.data_split.fold", value=fold)
        # # # # importer.change_config(attr="dataset.seq_length", value=[3,0])
        # importer.change_config(attr="training_params.batch_size", value=64)
        importer.change_config(attr="training_params.test_batch_size", value=16)
        #
        data_loader = importer.get_dataloaders()

        # print(data_loader.test)
        # data_loader.valid_loader.dataset.choose_specific_patient(include_chosen=False, patient_nums=[149, 177, 2435, 2783, 2892, 4003, 4011, 4316, 5368])
        # generator = Generate_n_Compare(model=best_model, data_loader=data_loader, config=importer.config, device=device)
        # generator.reconstruct(set="Validation", plot_comparison=True)

        validator = Validator(model=best_model, data_loader=data_loader, config=importer.config, device=device)
        test_results = validator.get_results(set="Test", print_results=False)
        multi_fold_results[count] = test_results
        # importer.print_progress_aggregated(multi_fold_results=multi_fold_results, latex_version=False)

        if (i+1)%3 == 0:
            mm_acc.append(np.array([multi_fold_results[i]["acc"]["combined"] for i in multi_fold_results]).mean())
            mm_f1.append(np.array([multi_fold_results[i]["f1"]["combined"] for i in multi_fold_results]).mean())
            mm_k.append(np.array([multi_fold_results[i]["k"]["combined"] for i in multi_fold_results]).mean())

            mm_acc_std.append(np.array([multi_fold_results[i]["acc"]["combined"] for i in multi_fold_results]).std())
            mm_f1_std.append(np.array([multi_fold_results[i]["f1"]["combined"] for i in multi_fold_results]).std())
            mm_k_std.append(np.array([multi_fold_results[i]["k"]["combined"] for i in multi_fold_results]).std())

            multi_fold_results = {}

        count = count+1

for i in range(len(mm_acc)):
    print(round(mm_acc[i]*100, 1), end="{\\ tiny$\\pm$")
    print(round(mm_acc_std[i]*100, 1), end="} & ")
    print(round(mm_f1[i]*100, 1), end="{\\ tiny$\\pm$")
    print(round(mm_f1_std[i]*100, 1), end=" & ")
    print(round(mm_k[i], 3), end="{\\ tiny$\\pm$")
    print(round(mm_k_std[i], 3), end="\\\\ \n")

