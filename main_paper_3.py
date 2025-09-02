import argparse
from utils.config import process_config

from agents.sleep_test import *

import matplotlib.pyplot as plt
import numpy as np


def main():
    config_list = [
        # "./configs/paper_finals/shhs/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_aligninner_trial1_pt.json",
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
        # "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_aligninner_trial2_pt_incomplete_50p_0e_0o.json",
        # "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_aligninner_trial2_pt_incomplete_50p_50e_0o.json",
        # "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_aligninner_trial2_pt_incomplete_50p_100e_0o.json",
        # "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_aligninner_trial2_pt_incomplete_50p_200e_0o.json",
        #
        # "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_aligninner_trial1_pt_incomplete_50p_0e_0o.json",
        # "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_aligninner_trial1_pt_incomplete_50p_50e_0o.json",
        # "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_aligninner_trial1_pt_incomplete_50p_100e_0o.json",
        # "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_aligninner_trial1_pt_incomplete_50p_200e_0o.json",

        # "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_aligninner_trial2_pt_incomplete_100p_0e_0o.json",
        # "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_aligninner_trial2_pt_incomplete_100p_50e_0o.json",
        # "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_aligninner_trial2_pt_incomplete_100p_100e_0o.json",
        # "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_aligninner_trial2_pt_incomplete_100p_200e_0o.json",
        # "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_aligninner_trial2_pt_incomplete_100p_400e_0o.json",
        # "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_aligninner_trial2_pt_incomplete_100p_600e_0o.json",
        #
        # "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_aligninner_trial1_pt_incomplete_100p_0e_0o.json",
        # "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_aligninner_trial1_pt_incomplete_100p_50e_0o.json",
        # "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_aligninner_trial1_pt_incomplete_100p_100e_0o.json",
        # "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_aligninner_trial1_pt_incomplete_100p_200e_0o.json",
        # "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_aligninner_trial1_pt_incomplete_100p_400e_0o.json",
        # "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_aligninner_trial1_pt_incomplete_100p_600e_0o.json",
        # "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_aligninner_trial1_incomplete_100p_0e_0o.json",
        # "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_aligninner_trial1_incomplete_100p_50e_0o.json",
        # "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_aligninner_trial1_incomplete_100p_100e_0o.json",
        # "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_aligninner_trial1_incomplete_100p_200e_0o.json",
        # "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_aligninner_trial1_incomplete_100p_400e_0o.json",
        # "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_aligninner_trial1_incomplete_100p_600e_0o.json",

        # "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_aligninner_trial1_incomplete_100p_1000e_0o.json",
        # "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_aligninner_trial1_pt_incomplete_100p_1000e_0o.json",
        # "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_aligninner_trial2_pt_incomplete_100p_1000e_0o.json",

        # "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_aligninner_trial1_incomplete_100p_1500e_0o.json",
        # "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_aligninner_trial1_pt_incomplete_100p_1500e_0o.json",
        # "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_aligninner_trial2_pt_incomplete_100p_1500e_0o.json",
        #
        # "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_aligninner_trial1_incomplete_100p_2000e_0o.json",
        # "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_aligninner_trial1_pt_incomplete_100p_2000e_0o.json",
        # "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_aligninner_trial2_pt_incomplete_100p_2000e_0o.json",

    # "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_aligninner_trial1_incomplete_100p_0e_400o.json",
    # "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_aligninner_trial1_pt_incomplete_100p_0e_400o.json",
    # "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_aligninner_trial2_pt_incomplete_100p_0e_400o.json",
    #
    # "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_aligninner_trial1_incomplete_100p_0e_600o.json",
    # "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_aligninner_trial1_pt_incomplete_100p_0e_600o.json",
    # "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_aligninner_trial2_pt_incomplete_100p_0e_600o.json",

    # "./configs/paper_finals/shhs/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_aligninner_al5_trial1_pt.json",
    # "./configs/paper_finals/shhs/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_aligninner_al5_trial2_pt.json",
    # "./configs/paper_finals/shhs/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_aligninner_al10_trial2_pt.json",
    # "./configs/paper_finals/shhs/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_aligninner_al40_trial2_pt.json",
    # "./configs/paper_finals/shhs/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_aligninner_al60_trial2_pt.json",
    # "./configs/paper_finals/shhs/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_aligninner_al80_trial2_pt.json",
    # "./configs/paper_finals/shhs/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_aligninner_al100_trial1_pt.json",
    # "./configs/paper_finals/shhs/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_aligninner_al100_trial1_pt.json",

    # "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_aligninner_trial2_pt_incomplete_nonsharedpreds_100p_50e_50o.json",
    # "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_aligninner_trial1_pt_incomplete_nonsharedpreds_100p_50e_50o.json",
    # "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_aligninner_trial1_incomplete_nonsharedpreds_100p_50e_50o.json",

    # "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_aligninner_trial2_pt_incomplete_unsharedpreds_100p_100e_100o.json",
    # "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_aligninner_trial1_pt_incomplete_nonsharedpreds_100p_100e_100o.json",
    # "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_aligninner_trial1_incomplete_nonsharedpreds_100p_100e_100o.json",

    "./configs/paper_finals/shhs/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_aligninner_al40_trial2.json",
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