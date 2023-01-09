import sys
sys.exc_info()

from posthoc.Helpers.Helper_Importer import Importer
from posthoc.Helpers.Helper_Validator import Validator
from posthoc.Helpers.Helper_LogsPlotter import LogsPlotter
from posthoc.Helpers.Helper_PatientWise import PatientWise_Analyser
from collections import defaultdict

import os
os.chdir('/users/sista/kkontras/Documents/Sleep_Project/')

config_list = [
    "./configs/shhs/single_channel/fourier_transformer_cls_eeg_mat_adv.json",
    # "./configs/shhs/multi_modal/eeg_eog/established_models/fourier_transformer_eeg_eog_mat_BIOBLIP.json",
    # "./configs/shhs/multi_modal/eeg_eog/established_models/fourier_transformer_eeg_eog_mat_BIOBLIP_lossw_masked.json",
]

nested_dict = lambda : defaultdict(nested_dict)
multi_fold_results = nested_dict()
# device = "cpu"
device = "cuda:0"
for i, config_name in enumerate(config_list):
    for fold in range(1):

        # config_name = "./configs/shhs/single_channel/fourier_transformer_cls_eeg_mat_adv.json"
        config_name ="./configs/paper_finals/shhs/fourier_transformer_eeg_trial1.json"
        importer_eeg = Importer(config_name=config_name, device=device, fold=fold)
        importer_eeg.load_checkpoint()

        # config_name = "./configs/shhs/multi_modal/eeg_eog/established_models/fourier_transformer_eeg_eog_mat_BIOBLIP.json"
        # importer_blip = Importer(config_name=config_name, device=device, fold=fold)
        # importer_blip.load_checkpoint()

        # config_name = "./configs/shhs/multi_modal/eeg_eog/established_models/fourier_transformer_eeg_eog_mat_BIOBLIP_twomode.json"
        # config_name = "./configs/shhs/multi_modal/eeg_eog/established_models/fourier_transformer_eeg_eog_mat_BIOBLIP_twomode_sepouter.json"
        # config_name = "./configs/shhs/multi_modal/eeg_eog/established_models/fourier_transformer_eeg_eog_mat_BIOBLIP_twomode_caouter_al1.json"
        # config_name = "./configs/shhs/multi_modal/eeg_eog/established_models/fourier_transformer_eeg_eog_mat_BIOBLIP_twomode_caouter_al1_shared_limited.json"
        # config_name = "./configs/shhs/multi_modal/eeg_eog/established_models/fourier_transformer_eeg_eog_mat_BIOBLIP_twomode_caouter_al1_shared_b16_mult.json"
        # config_name = "./configs/shhs/multi_modal/eeg_eog/established_models/fourier_transformer_eeg_eog_mat_BIOBLIP_twomode_caouter_al1_shared_b16_mult_freetrain_lr3.json"
        # config_name = "./configs/shhs/multi_modal/eeg_eog/established_models/fourier_transformer_eeg_eog_mat_BIOBLIP_twomode_caouter_al1_shared_b16_mult_freetrain_lr3_cliplike.json"
        # config_name = "./configs/shhs/multi_modal/eeg_eog/established_models/fourier_transformer_eeg_eog_mat_BIOBLIP_twomode_caouter_al01_shared_b16_mult_freetrain_lr3_cliplike.json"
        # config_name = "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos.json"
        # config_name = "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos.json"
        # config_name = "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_bigal01_shared_b16_freetrain.json"
        # config_name = "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_drop03_aligninner_trial2.json"
        config_name = "./configs/paper_finals/shhs/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_aligninner_trial1.json"

        importer_blip_twomode = Importer(config_name=config_name, device=device, fold=fold)
        importer_blip_twomode.load_checkpoint()
        multi_fold_results = importer_blip_twomode.print_progress(multi_fold_results=multi_fold_results)

        # importer_blip_twomode.change_config(attr="dataset.discard_nonskipped", value=True)
        importer_blip_twomode.change_config(attr="training_params.test_batch_size", value=256)
        importer_blip_twomode.change_config(attr="dataset.data_view_dir", value=[
        {"list_dir": "patient_mat_list.txt", "data_type": "stft", "mod":  "eeg", "num_ch": 1},
        {"list_dir": "patient_mat_list.txt", "data_type": "time", "mod":  "eeg", "num_ch": 1},
        {"list_dir": "patient_eog_mat_list.txt", "data_type": "stft", "mod":  "eog", "num_ch": 1},
        {"list_dir": "patient_eog_mat_list.txt", "data_type": "time", "mod":  "eog", "num_ch": 1}
        ])
        importer_blip_twomode.change_config(attr="dataset.filter_windows", value={
                                            "train": {"use_type": False, "skip_skips": True},
                                            "val": {"use_type": False, "skip_skips": True},
                                            "test": {"use_type": False, "skip_skips": True}})
                                            # "test": {"use_type": "include_skipped", "whole_patient": True, "std_threshold": 41, "perc_threshold": 0.4}})

        config_name = "./configs/paper_finals/shhs/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_Early_aligninner_trial2_pt.json"
        importer_early_twomode = Importer(config_name=config_name, device=device, fold=fold)
        importer_early_twomode.load_checkpoint()

        config_name = "./configs/shhs/single_channel/fourier_transformer_cls_eog_mat.json"
        importer_eog = Importer(config_name=config_name, device=device, fold=fold)
        importer_eog.load_checkpoint()


        # config_name = "configs/shhs/multi_modal/eeg_eog/established_models/fourier_transformer_eeg_eog_mat_BIOBLIP_twomode_caouter_onlyal.json"
        # importer_onlyalign = Importer(config_name=config_name, device=device, fold=fold)
        # importer_onlyalign.load_checkpoint()
        #
        # config_name = "./configs/shhs/reconstruction/vae_eeg.json"
        # importer_vae_eeg = Importer(config_name=config_name, device=device, fold=fold)
        # importer_vae_eeg.load_checkpoint()
        #
        # config_name = "./configs/shhs/reconstruction/vae_eοg.json"
        # importer_vae_eog = Importer(config_name=config_name, device=device, fold=fold)
        # importer_vae_eog.load_checkpoint()

        # data_loader = importer_eeg.get_dataloaders()
        # data_loader = importer_blip.get_dataloaders()
        data_loader = importer_blip_twomode.get_dataloaders()

        best_model_eeg = importer_eeg.get_model(return_model="best_model")
        # best_model_blip = importer_blip.get_model(return_model="best_model")
        best_model_eog = importer_eog.get_model(return_model="best_model")
        best_model_blip_twomode = importer_blip_twomode.get_model(return_model="best_model")
        best_model_early_twomode = importer_early_twomode.get_model(return_model="best_model")


        # best_model_onlyalign = importer_onlyalign.get_model(return_model="best_model")
        # best_model_vae_eeg = importer_vae_eeg.get_model(return_model="best_model")
        # best_model_vae_eog = importer_vae_eog.get_model(return_model="best_model")

        # multi_fold_results = patient_analyzer.gather_comparisons(set="Test", models={"eeg": best_model_eeg, "blip":best_model_blip, "eog": best_model_eog },
        #                                                          multi_fold_results=multi_fold_results, router_model=best_model_vae, plot_hypnograms=True)
        # patient_analyzer.plot_comparisons(models={"eeg": best_model_eeg, "blip":best_model_blip, "eog":best_model_eog, "vae":"yo"}, results=multi_fold_results)
        # print(multi_fold_results)

        # data_loader.valid_loader.dataset.choose_specific_patient(include_chosen=True, patient_nums=[149, 177, 2435, 2783, 2892, 4003, 4011, 4316, 5368])
        # data_loader.test_loader.dataset.choose_specific_patient(include_chosen=True, patient_nums=[55,2,91])

        patient_analyzer = PatientWise_Analyser(data_loader=data_loader, device=device)
        # multi_fold_results = patient_analyzer.gather_comparisons(set="Validation", models={ "eeg":best_model_eeg, "blip":best_model_blip, "eog":best_model_eog },
        #                                                          multi_fold_results=multi_fold_results, router_models=False, plot_hypnograms=True)

        # multi_fold_results = patient_analyzer.load_results(filename="/users/sista/kkontras/Documents/Sleep_Project/experiments/paper_results_all.pkl", prev_results=multi_fold_results)

        multi_fold_results = patient_analyzer.gather_comparisons(set="Test", # Training Validation, Test
                                                                 models={
                                                                     # "blip_tm": best_model_blip_twomode,
                                                                     "blip_tm_i": best_model_blip_twomode,
                                                                     "blip_tm_eeg": best_model_blip_twomode,
                                                                     "blip_tm_eog": best_model_blip_twomode,
                                                                     "early_tm_i": best_model_early_twomode,
                                                                     # "blip_skip": best_model_blip_twomode,
                                                                     "eeg": best_model_eeg,
                                                                     "eog": best_model_eog
                                                                         },
                                                                 # only_align_model = best_model_onlyalign,
                                                                 # router_models={
                                                                 #     # "vae_eeg": best_model_vae_eeg,
                                                                 #     # "vae_eog": best_model_vae_eog
                                                                 # },
                                                                 once_whole_set=False,
                                                                 multi_fold_results=multi_fold_results,
                                                                 plot_hypnograms=False )

        # patient_analyzer.save_results(filename="/users/sista/kkontras/Documents/Sleep_Project/experiments/paper_results.pkl", results = multi_fold_results )

        # patient_analyzer.plot_comparisons(models={"blip_tm_i": {}, "blip_skip": {}}, results=multi_fold_results)
        # patient_analyzer.plot_spider(results=multi_fold_results)
        # patient_analyzer.plot_comparisons(models={"blip_tm_eeg": {}, "blip_tm": {}, "blip_tm_i": {}, "blip_tm_eog": {}}, results=multi_fold_results)
        # patient_analyzer.plot_comparisons(models={"blip_tm_i": {}, "vae": {}, "std": {}, "zc": {}}, results=multi_fold_results)
        # patient_analyzer.plot_comparisons(models={"blip_tm_eeg": {}, "blip_tm_i": {}, "blip_tm_eog": {}}, results=multi_fold_results)
        # patient_analyzer.plot_comparisons(models={"blip_tm_eeg": {}, "blip_tm_i": {}, "blip_tm": {}}, results=multi_fold_results)
        patient_analyzer.plot_comparisons(models={"eeg": {}, "blip_tm_i": {}}, results=multi_fold_results)
        # patient_analyzer.plot_comparisons(models={"blip_tm": {}, "blip_tm_i": {}}, results=multi_fold_results)
        # patient_analyzer.plot_comparisons(models={"eeg": {}, "blip_tm_eeg": {}}, results=multi_fold_results)
        # patient_analyzer.plot_comparisons(models={"eog": {}, "blip_tm_eog": {}}, results=multi_fold_results)
        # patient_analyzer.plot_comparisons(models={"blip_tm_eeg": {}, "blip_tm_eog": {}}, results=multi_fold_results)
        # patient_analyzer.plot_comparisons(models={"blip_tm": {}, "vae": {}}, results=multi_fold_results)
        # patient_analyzer.plot_comparisons(models={"blip_tm_i": {}, "vae": {}}, results=multi_fold_results)

        # print(multi_fold_results)
        # print(multi_fold_results)
        # multi_fold_results = patient_analyzer.gather_comparisons(set="Validation", models={"eeg": best_model_eeg}, multi_fold_results=multi_fold_results, plot_hypnograms=False)
        # # patient_analyzer.plot_comparisons(models={"eeg": best_model_eeg}, results=multi_fold_results)
        # print(multi_fold_results)
        # val_patients = []
        # for patient in multi_fold_results["eeg"]:l
        #     if multi_fold_results["eeg"][patient]["acc"] < 0.75:
        #         val_patients.append(patient.split("_")[1])
        #         print(multi_fold_results["eeg"][patient])
        #
        # print(val_patients)
        # multi_fold_results = {}

        # multi_fold_results = patient_analyzer.gather_comparisons(set="Training", models={"eeg": best_model_eeg}, multi_fold_results=multi_fold_results, plot_hypnograms=False)
        # # patient_analyzer.plot_comparisons(models={"eeg": best_model_eeg}, results=multi_fold_results)
        # print(multi_fold_results)
        # val_patients = []
        # for patient in multi_fold_results["eeg"]:
        #     if multi_fold_results["eeg"][patient]["acc"] < 0.75:
        #         val_patients.append(patient.split("_")[1])
        #         print(multi_fold_results["eeg"][patient])
        #
        # print(val_patients)



        # plotter.sleep_plot_losses()
        # plotter.sleep_plot_performance()
        # plotter.sleep_plot_lr()
