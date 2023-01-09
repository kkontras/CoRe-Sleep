import sys
import os
import numpy as np

sys.exc_info()
os.chdir('/users/sista/kkontras/Documents/Sleep_Project/')


from posthoc.Helpers.Helper_Importer import Importer
from posthoc.Helpers.Helper_Validator import Validator
from posthoc.Helpers.Helper_LogsPlotter import LogsPlotter
from posthoc.Helpers.Helper_Generate_n_Compare import Generate_n_Compare

config_list = [

    # "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_aligninner_trial1_incomplete_50p_0e_0o.json",
    # "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_aligninner_trial1_pt_incomplete_50p_0e_0o.json",
    # "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_aligninner_trial2_pt_incomplete_50p_0e_0o.json",

    # "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_aligninner_trial1_incomplete_50p_50e_0o.json",
    # "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_aligninner_trial1_pt_incomplete_50p_50e_0o.json",
    # "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_aligninner_trial2_pt_incomplete_50p_50e_0o.json",
    #
    # "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_aligninner_trial1_incomplete_50p_100e_0o.json",
    # "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_aligninner_trial1_pt_incomplete_50p_100e_0o.json",
    # "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_aligninner_trial2_pt_incomplete_50p_100e_0o.json",
    #
    # "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_aligninner_trial1_incomplete_50p_200e_0o.json",
    # "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_aligninner_trial1_pt_incomplete_50p_200e_0o.json",
    # "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_aligninner_trial2_pt_incomplete_50p_200e_0o.json",
    #


    # "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_aligninner_trial2_pt_incomplete_100p_0e_0o.json",
    # "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_aligninner_trial1_pt_incomplete_100p_0e_0o.json",
    # "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_aligninner_trial1_incomplete_100p_0e_0o.json",
    # #
    # "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_aligninner_trial2_pt_incomplete_100p_50e_0o.json",
    # "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_aligninner_trial1_pt_incomplete_100p_50e_0o.json",
    # "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_aligninner_trial1_incomplete_100p_50e_0o.json",
    #
    # "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_aligninner_trial2_pt_incomplete_100p_100e_0o.json",
    # "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_aligninner_trial1_pt_incomplete_100p_100e_0o.json",
    # "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_aligninner_trial1_incomplete_100p_100e_0o.json",
    #
    # "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_aligninner_trial2_pt_incomplete_100p_200e_0o.json",
    # "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_aligninner_trial1_pt_incomplete_100p_200e_0o.json",
    # "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_aligninner_trial1_incomplete_100p_200e_0o.json",
    #
    # "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_aligninner_trial1_incomplete_100p_400e_0o.json",
    # "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_aligninner_trial1_pt_incomplete_100p_400e_0o.json",
    # "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_aligninner_trial2_pt_incomplete_100p_400e_0o.json",
    #
    # "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_aligninner_trial1_incomplete_100p_600e_0o.json",
    # "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_aligninner_trial1_pt_incomplete_100p_600e_0o.json",
    # "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_aligninner_trial2_pt_incomplete_100p_600e_0o.json",
    #
    # "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_aligninner_trial1_incomplete_100p_1000e_0o.json",
    # "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_aligninner_trial1_pt_incomplete_100p_1000e_0o.json",
    # "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_aligninner_trial2_pt_incomplete_100p_1000e_0o.json",
    #
    # "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_aligninner_trial1_incomplete_100p_1500e_0o.json",
    # "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_aligninner_trial1_pt_incomplete_100p_1500e_0o.json",
    # "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_aligninner_trial2_pt_incomplete_100p_1500e_0o.json",
    #
    # "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_aligninner_trial1_incomplete_100p_2000e_0o.json",
    # "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_aligninner_trial1_pt_incomplete_100p_2000e_0o.json",
    # "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_aligninner_trial2_pt_incomplete_100p_2000e_0o.json",



    # "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_aligninner_trial1_incomplete_100p_0e_50o.json",
    # "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_aligninner_trial1_pt_incomplete_100p_0e_50o.json",
    # "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_aligninner_trial2_pt_incomplete_100p_0e_50o.json",
    #
    # "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_aligninner_trial1_incomplete_100p_0e_100o.json",
    # "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_aligninner_trial1_pt_incomplete_100p_0e_100o.json",
    # "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_aligninner_trial2_pt_incomplete_100p_0e_100o.json",
    #
    # "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_aligninner_trial1_incomplete_100p_0e_200o.json",
    # "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_aligninner_trial1_pt_incomplete_100p_0e_200o.json",
    # "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_aligninner_trial2_pt_incomplete_100p_0e_200o.json",
    #
    # "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_aligninner_trial1_incomplete_100p_0e_400o.json",
    # "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_aligninner_trial1_pt_incomplete_100p_0e_400o.json",
    # "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_aligninner_trial2_pt_incomplete_100p_0e_400o.json",
    # #
    # "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_aligninner_trial1_incomplete_100p_0e_600o.json",
    # "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_aligninner_trial1_pt_incomplete_100p_0e_600o.json",
    # "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_aligninner_trial2_pt_incomplete_100p_0e_600o.json",
    #
    # "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_aligninner_trial1_incomplete_100p_0e_1000o.json",
    # "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_aligninner_trial1_pt_incomplete_100p_0e_1000o.json",
    # "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_aligninner_trial2_pt_incomplete_100p_0e_1000o.json",
    #
    # "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_aligninner_trial1_incomplete_100p_0e_1500o.json",
    # "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_aligninner_trial1_pt_incomplete_100p_0e_1500o.json",
    # "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_aligninner_trial2_pt_incomplete_100p_0e_1500o.json",
    #
    # "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_aligninner_trial1_incomplete_100p_0e_2000o.json",
    # "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_aligninner_trial1_pt_incomplete_100p_0e_2000o.json",
    # "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_aligninner_trial2_pt_incomplete_100p_0e_2000o.json",



    # "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_aligninner_trial2_pt_incomplete_100p_50e_50o.json",
    # "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_aligninner_trial1_pt_incomplete_100p_50e_50o.json",
    # "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_aligninner_trial1_incomplete_100p_50e_50o.json",
    #
    # "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_aligninner_trial2_pt_incomplete_100p_100e_100o.json",
    # "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_aligninner_trial1_pt_incomplete_100p_100e_100o.json",
    # "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_aligninner_trial1_incomplete_100p_100e_100o.json",
    #
    # "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_aligninner_trial2_pt_incomplete_100p_200e_200o.json",
    # "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_aligninner_trial1_pt_incomplete_100p_200e_200o.json",
    # "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_aligninner_trial1_incomplete_100p_200e_200o.json",
    #
    # "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_aligninner_trial1_incomplete_100p_400e_400o.json",
    # "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_aligninner_trial1_pt_incomplete_100p_400e_400o.json",
    # "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_aligninner_trial2_pt_incomplete_100p_400e_400o.json",
    #
    # "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_aligninner_trial1_incomplete_100p_600e_600o.json",
    # "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_aligninner_trial1_pt_incomplete_100p_600e_600o.json",
    # "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_aligninner_trial2_pt_incomplete_100p_600e_600o.json",
    #
    # "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_aligninner_trial1_incomplete_100p_1000e_1000o.json",
    # "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_aligninner_trial1_pt_incomplete_100p_1000e_1000o.json",
    # "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_aligninner_trial2_pt_incomplete_100p_1000e_1000o.json",
    #
    # "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_aligninner_trial1_incomplete_100p_1500e_1500o.json",
    # "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_aligninner_trial1_pt_incomplete_100p_1500e_1500o.json",
    # "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_aligninner_trial2_pt_incomplete_100p_1500e_1500o.json",



    # "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_aligninner_trial2_pt_incomplete_unsharedpreds_100p_50e_50o.json",
    # "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_aligninner_trial1_pt_incomplete_unsharedpreds_100p_50e_50o.json",
    # "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_aligninner_trial1_incomplete_unsharedpreds_100p_50e_50o.json",
    #
    # "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_aligninner_trial2_pt_incomplete_unsharedpreds_100p_100e_100o.json",
    # "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_aligninner_trial1_pt_incomplete_unsharedpreds_100p_100e_100o.json",
    # "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_aligninner_trial1_incomplete_unsharedpreds_100p_100e_100o.json",
    #
    # "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_aligninner_trial2_pt_incomplete_unsharedpreds_100p_200e_200o.json",
    # "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_aligninner_trial1_pt_incomplete_unsharedpreds_100p_200e_200o.json",
    # "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_aligninner_trial1_incomplete_unsharedpreds_100p_200e_200o.json",
    #
    # "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_aligninner_trial2_pt_incomplete_unsharedpreds_100p_400e_400o.json",
    # "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_aligninner_trial1_pt_incomplete_unsharedpreds_100p_400e_400o.json",
    # "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_aligninner_trial1_incomplete_unsharedpreds_100p_400e_400o.json",
    #
    # "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_aligninner_trial2_pt_incomplete_unsharedpreds_100p_600e_600o.json",
    # "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_aligninner_trial1_pt_incomplete_unsharedpreds_100p_600e_600o.json",
    # "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_aligninner_trial1_incomplete_unsharedpreds_100p_600e_600o.json",
    #
    # "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_aligninner_trial2_pt_incomplete_unsharedpreds_100p_1000e_1000o.json",
    # "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_aligninner_trial1_pt_incomplete_unsharedpreds_100p_1000e_1000o.json",
    # "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_aligninner_trial1_incomplete_unsharedpreds_100p_1000e_1000o.json",
    #
    # "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_aligninner_trial2_pt_incomplete_unsharedpreds_100p_1500e_1500o.json",
    # "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_aligninner_trial1_pt_incomplete_unsharedpreds_100p_1500e_1500o.json",
    # "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_aligninner_trial1_incomplete_unsharedpreds_100p_1500e_1500o.json",




    "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_aligninner_trial2_pt_incomplete_unsharedpreds_100p_50e_50o.json",
    "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_aligninner_trial1_pt_incomplete_unsharedpreds_100p_50e_50o.json",
    "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_aligninner_trial1_incomplete_unsharedpreds_100p_50e_50o.json",

    "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_aligninner_trial2_pt_incomplete_unsharedpreds_100p_100e_100o.json",
    "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_aligninner_trial1_pt_incomplete_unsharedpreds_100p_100e_100o.json",
    "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_aligninner_trial1_incomplete_unsharedpreds_100p_100e_100o.json",

    "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_aligninner_trial2_pt_incomplete_unsharedpreds_100p_200e_200o.json",
    "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_aligninner_trial1_pt_incomplete_unsharedpreds_100p_200e_200o.json",
    "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_aligninner_trial1_incomplete_unsharedpreds_100p_200e_200o.json",

    "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_aligninner_trial2_pt_incomplete_unsharedpreds_100p_400e_400o.json",
    "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_aligninner_trial1_pt_incomplete_unsharedpreds_100p_400e_400o.json",
    "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_aligninner_trial1_incomplete_unsharedpreds_100p_400e_400o.json",

    "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_aligninner_trial2_pt_incomplete_unsharedpreds_100p_600e_600o.json",
    "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_aligninner_trial1_pt_incomplete_unsharedpreds_100p_600e_600o.json",
    "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_aligninner_trial1_incomplete_unsharedpreds_100p_600e_600o.json",

    "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_aligninner_trial2_pt_incomplete_unsharedpreds_100p_1000e_1000o.json",
    "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_aligninner_trial1_pt_incomplete_unsharedpreds_100p_1000e_1000o.json",
    "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_aligninner_trial1_incomplete_unsharedpreds_100p_1000e_1000o.json",

    "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_aligninner_trial2_pt_incomplete_unsharedpreds_100p_1500e_1500o.json",
    "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_aligninner_trial1_pt_incomplete_unsharedpreds_100p_1500e_1500o.json",
    "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_aligninner_trial1_incomplete_unsharedpreds_100p_1500e_1500o.json",

]
count = 0
multi_fold_results = {}

eeg, eog, mm = [], [], []
eeg_std, eog_std, mm_std = [], [], []
for i, config_name in enumerate(config_list):
    for fold in range(1):
        # fold = len(config_list)*i + fold
        importer = Importer(config_name=config_name, device="cuda:0", fold=fold)
        try:
            importer.load_checkpoint()
        except:
            print("We could not load {}".format(config_name))
            continue
        plotter = LogsPlotter(config=importer.config, logs=importer.checkpoint["logs"])

        importer.fold = count*10 + fold
        # try:
        #     importer.change_config(attr="model.args.shared_preds", value=False)
        #     a = importer.get_model()
        #     importer._my_numel(a)
        # except:
        #     print("We could not load {}".format(config_name))
        #     continue

        multi_fold_results = importer.print_progress(multi_fold_results=multi_fold_results, latex_version=False)
        # importer.print_progress_aggregated(multi_fold_results=multi_fold_results, latex_version=True)


        # if (i+1)%3 == 0:
        #     print(multi_fold_results.keys())
        #     eeg.append(np.array([multi_fold_results[i]["k"]["eeg"] for i in multi_fold_results]).mean())
        #     eog.append(np.array([multi_fold_results[i]["k"]["eog"] for i in multi_fold_results]).mean())
        #     mm.append(np.array([multi_fold_results[i]["k"]["combined"] for i in multi_fold_results]).mean())
        #     eeg_std.append(np.array([multi_fold_results[i]["k"]["eeg"] for i in multi_fold_results]).std())
        #     eog_std.append(np.array([multi_fold_results[i]["k"]["eog"] for i in multi_fold_results]).std())
        #     mm_std.append(np.array([multi_fold_results[i]["k"]["combined"] for i in multi_fold_results]).std())
        #     multi_fold_results = {}

        # plotter.sleep_plot_losses()
        # plotter.sleep_plot_performance()
    count +=1

# eeg, eog, mm = np.array(eeg), np.array(eog), np.array(mm)
# eeg_std, eog_std, mm_std = np.array(eeg_std), np.array(eog_std), np.array(mm_std)
# # x = np.array([  0, 50,  100,  200, 400,  600, 1000, 1500])
# x = np.array([  0, 50,  100,  200, 400,  600, 1000, 1500, 2000])
# import matplotlib.pyplot as plt
# plt.figure(figsize=(10,6))
# ax = plt.subplot(111)
# plt.plot(x, eeg, color="blue", label="EEG")
# plt.plot(x, eog, color="orange", label="EOG")
# plt.plot(x, mm, color="black", label="Multimodal")
# plt.scatter(x, eeg, color="blue")
# plt.scatter(x, eog, color="orange")
# plt.scatter(x, mm, color="black")
# plt.fill_between(x, eeg-eeg_std, eeg+eeg_std, alpha=0.5)
# plt.fill_between(x, eog-eog_std, eog+eog_std, alpha=0.5)
# plt.fill_between(x, mm-mm_std, mm+mm_std, alpha=0.5)
# plt.legend(fontsize=12, frameon=False, ncol=3,  bbox_to_anchor=(0.28, 0.88))
# # plt.xlabel("Extra Single-Modality EEG patients", fontsize=12)
# plt.xlabel("Extra Single-Modality Patients", fontsize=12)
# # plt.xlabel("Extra Single-Modality Patients", fontsize=12)
# plt.ylabel("Cohens Kappa", fontsize=12)
# plt.xticks(x, fontsize = 8)
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# # ax.spines['bottom'].set_visible(False)
# plt.yticks([0.65, 0.70,0.75,0.80,0.85 ], fontsize = 8)
# plt.xlim(-100,2100)
# plt.xlim(-100,1600)
# plt.ylim(0.65,0.85)
# plt.title("Training with 100 EEG-EOG patients and EEG unimodal ones", fontsize=14)
# # plt.title("Training with 100 EEG-EOG patients and EOG unimodal ones", fontsize=14)
# # plt.title("Training with 100 EEG-EOG patients and unimodal ones", fontsize=14)
# # plt.ylim(0.75,0.9)
# # plt.savefig("./incomplete_100c_2000e_0o.svg")
# # plt.savefig("./incomplete_100c_2000e_0o.png")
# plt.show()

# Plot a three different lines with the given standard deviations in matplotlib with legend and all that nicely polished to be a publication ready figure and limited to a specific range of the x-axis.

import numpy as np
import matplotlib.pyplot as plt
def plot_signle_line(x, y, yerr, label, color):
    plt.plot(x, y, color=color, label=label)
    plt.scatter(x, y, color=color)
    plt.fill_between(x, y-yerr, y+yerr, alpha=0.5)
plt.figure(figsize=(10,6))
ax = plt.subplot(111)
x= np.array([  0, 50,  100,  200, 400,  600, 1000, 1500, 2000])
plot_signle_line( x, eeg, eeg_std, "EEG", "blue")
plot_signle_line( x, eog, eog_std, "EOG", "orange")
plot_signle_line( x, mm, mm_std, "Multimodal", "black")
plt.legend(fontsize=12, frameon=False, ncol=3,  bbox_to_anchor=(0.28, 0.88))
plt.xlabel("Extra Single-Modality Patients", fontsize=12)
plt.ylabel("Cohens Kappa", fontsize=12)
plt.xticks(x, fontsize = 8)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.yticks([0.65, 0.70,0.75,0.80,0.85 ], fontsize = 8)
plt.xlim(-100,2100)
plt.ylim(0.65,0.85)
plt.title("Training with 100 EEG-EOG patients and EEG unimodal ones", fontsize=14)
plt.show()


