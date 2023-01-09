import sys
sys.exc_info()

from posthoc.Helpers.Helper_Importer import Importer
from posthoc.Helpers.Helper_Validator import Validator
from posthoc.Helpers.Helper_LogsPlotter import LogsPlotter
from posthoc.Helpers.Helper_PatientWise import PatientWise_Analyser
from posthoc.Helpers.Helper_PatientWise_Noisy import PatientWise_Analyser_Noisy
from collections import defaultdict

import os
os.chdir('/users/sista/kkontras/Documents/Sleep_Project/')

fold = 0
device = "cuda:0"

# config_name = "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_nopos.json"
# importer_eeg = Importer(config_name=config_name, device=device, fold=fold)
# importer_eeg.load_checkpoint()

config_name = "./configs/paper_finals/shhs/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_aligninner_trial2.json"
importer_blip_twomode = Importer(config_name=config_name, device=device, fold=fold)
importer_blip_twomode.load_checkpoint()
multi_fold_results = importer_blip_twomode.print_progress(multi_fold_results={})

# importer_blip_twomode.change_config(attr="dataset.discard_nonskipped", value=True)
importer_blip_twomode.change_config(attr="training_params.test_batch_size", value=256)
importer_blip_twomode.change_config(attr="dataset.data_view_dir", value=[
# {"list_dir": "patient_mat_list.txt", "data_type": "stft", "mod":  "eeg", "num_ch": 1},
{"list_dir": "patient_mat_list.txt", "data_type": "time", "mod":  "eeg", "num_ch": 1},
# {"list_dir": "patient_eog_mat_list.txt", "data_type": "stft", "mod":  "eog", "num_ch": 1},
{"list_dir": "patient_eog_mat_list.txt", "data_type": "time", "mod":  "eog", "num_ch": 1}
])

data_loader = importer_blip_twomode.get_dataloaders()

# best_model_eeg = importer_eeg.get_model(return_model="best_model")
# best_model_blip_twomode = importer_blip_twomode.get_model(return_model="best_model")


patient_analyzer = PatientWise_Analyser_Noisy(data_loader=data_loader, device=device)

# multi_fold_results = patient_analyzer.load_results(filename="/users/sista/kkontras/Documents/Sleep_Project/experiments/paper_results_all.pkl", prev_results=multi_fold_results)
filename = "/users/sista/kkontras/Documents/Sleep_Project/experiments/noisy_patients_trial2_pp.pkl"

multi_fold_results = patient_analyzer.gather_comparisons(set="Total", # Training, Validation, Test, Total
                                                        filename=filename,
                                                        std_router = True)

# patient_analyzer.save_results(filename="/users/sista/kkontras/Documents/Sleep_Project/experiments/noisy_patients_trial2.pkl", results = multi_fold_results )
# patient_analyzer.plot_comparisons(models={"eeg": {}, "blip_tm_i": {}}, results=multi_fold_results)

