import argparse
from utils.config import process_config

from agents.sleep_test import *

import matplotlib.pyplot as plt
import numpy as np


def main():
    config_list = [
        # "./configs/paper_finals/shhs/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_Early_aligninner_trial1.json",
        # "./configs/paper_finals/shhs/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_Early_aligninner_trial1_pt.json",
        # "./configs/paper_finals/shhs/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_Early_aligninner_trial2_pt.json",
        # "./configs/paper_finals/shhs/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_Early_nomult_al01_aligninner_trial1.json",
        # "./configs/paper_finals/shhs/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_Early_nomult_al01_aligninner_trial1_pt.json",
        # "./configs/paper_finals/shhs/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_Early_nomult_al01_aligninner_trial2_pt.json",
        # "./configs/paper_finals/shhs/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_Early_nomult_aligninner_trial1.json",
        # "./configs/paper_finals/shhs/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_Early_nomult_aligninner_trial1_pt.json",
        # "./configs/paper_finals/shhs/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_Early_nomult_aligninner_trial2_pt.json",
        # "./configs/paper_finals/shhs/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_Early_mult_aligninner_trial1.json",
        # "./configs/paper_finals/shhs/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_Early_mult_aligninner_trial1_pt.json",
        # "./configs/paper_finals/shhs/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_Early_mult_aligninner_trial2_pt.json",

        # "./configs/paper_finals/shhs/fourier_transformer_eog_trial1.json",
        # "./configs/paper_finals/shhs/fourier_transformer_eog_trial1_pt.json",
        # "./configs/paper_finals/shhs/fourier_transformer_eog_trial2_pt.json",

        # "./configs/paper_finals/shhs/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_Early_aligninner_v2_trial1_pt.json"
        # "./configs/paper_finals/shhs/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_Early_aligninner_v2_trial2_pt.json"
        # "./configs/paper_finals/shhs/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_Early_aligninner_v2_trial1.json"

        # "./configs/paper_finals/shhs/fourier_transformer_eeg_eog_mat_BLIP_al01_sharedall_b16_freetrain_nopos_nonsharedpreds_drop03_aligninner_trial1_pt.json",
        # "./configs/paper_finals/shhs/fourier_transformer_eeg_eog_mat_BLIP_al01_sharedall_b16_freetrain_nopos_nonsharedpreds_drop03_aligninner_trial2_pt.json",
        # "./configs/paper_finals/shhs/fourier_transformer_eeg_eog_mat_BLIP_al01_sharedall_b16_freetrain_nopos_nonsharedpreds_drop03_aligninner_trial2.json",
        #
        # "./configs/paper_finals/shhs/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_aligninner_v2_trial1_pt.json",
        # "./configs/paper_finals/shhs/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_aligninner_v2_trial2_pt.json",
        # "./configs/paper_finals/shhs/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_aligninner_v2_trial2.json",


    # "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_aligninner_trial1_incomplete_100p_0e_100o.json",
    # "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_aligninner_trial1_pt_incomplete_100p_0e_100o.json",
    # "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_aligninner_trial2_pt_incomplete_100p_0e_100o.json",
    #
    # "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_aligninner_trial1_incomplete_100p_0e_1500o.json",
    # "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_aligninner_trial1_pt_incomplete_100p_0e_1500o.json",
    # "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_aligninner_trial2_pt_incomplete_100p_0e_1500o.json",
    # "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_aligninner_trial2_pt_incomplete_nonsharedpreds_100p_50e_50o.json",
    # "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_aligninner_trial1_pt_incomplete_unsharedpreds_100p_50e_50o.json",
    # "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_aligninner_trial1_incomplete_nonsharedpreds_100p_50e_50o.json",

    # "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_aligninner_trial2_pt_incomplete_nonsharedpreds_100p_100e_100o.json",
    # "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_aligninner_trial1_pt_incomplete_nonsharedpreds_100p_100e_100o.json",
    # "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_aligninner_trial1_incomplete_nonsharedpreds_100p_100e_100o.json",


    # "./configs/paper_finals/shhs/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_aligninner_al100_trial2_pt.json",

    # "./configs/paper_finals/shhs/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_aligninner_al40_trial2.json",
    # "./configs/paper_finals/shhs/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_aligninner_al80_trial2.json",
    # "./configs/paper_finals/shhs/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_aligninner_al100_trial1_pt.json",
    # "./configs/paper_finals/shhs/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_aligninner_al100_trial2.json",

    "./configs/paper_finals/shhs/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_aligninner_al5_trial2.json",
    "./configs/paper_finals/shhs/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_aligninner_al10_trial2.json",

    ]
    finals = []
    for i in config_list:
        config = process_config(i)
        agent_class = globals()[config.agent]
        agent = agent_class(config)
        # agent.check_energies_per_class()
        agent.run()
        agent.finalize()
        del agent

main()