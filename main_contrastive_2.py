#!/usr/bin/env python
import argparse
from utils.config import process_config

from agents.sleep_test import *

import matplotlib.pyplot as plt
import numpy as np


def main():
    config_list = [
        # "./configs/shhs/single_channel/fourier_transformer_eeg_connepoch.json",
        # "./configs/nch/single_channel/fourier_transformer_eeg_mat.json",
        # "./configs/nch/multi_modal/fourier_transformer_eeg_eog_merged_mat.json",
        # "./configs/shhs/multi_modal/eeg_eog/established_models/fourier_transformer_eeg_eog_mat_merged_rpos_adv.json"
        # "./configs/shhs/multi_modal/eeg_eog/established_models/fourier_transformer_eeg_eog_mat_late_rpos_adv_temp.json"
        # "./configs/shhs/multi_modal/eeg_eog/fourier_transformer_eeg_eog_mat_BIOBLIP_outer.json"
        # "./configs/nch/multi_channel/fourier_transformer_multichannel_eeg.json",
        # "./configs/nch/multi_modal/established_models/fourier_transformer_eeg_eog_mat_late_rpos_adv_temp.json",
        # "./configs/shhs/multi_modal/eeg_eog/established_models/fourier_transformer_eeg_eog_mat_early_concat_onlyi.json",
        # "./configs/shhs/multi_modal/eeg_eog/established_models/fourier_transformer_eeg_eog_mat_BIOBLIP_twomode_caouter_al0_shared_b16_t025.json"

        # "./configs/shhs/multi_modal/eeg_eog/established_models/fourier_transformer_eeg_eog_mat_BIOBLIP_twomode_caouter_al01_shared_b16_t025.json",
        # "./configs/shhs/multi_modal/eeg_eog/established_models/fourier_transformer_eeg_eog_mat_BIOBLIP_twomode_caouter_al01_shared_b16_t033.json",
        # "./configs/shhs/multi_modal/eeg_eog/established_models/fourier_transformer_eeg_eog_mat_BIOBLIP_twomode_caouter_al1_shared_b16.json",
        # "./configs/shhs/multi_modal/eeg_eog/established_models/fourier_transformer_eeg_eog_mat_BIOBLIP_twomode_caouter_al1_shared_b16_t025.json",
        # "./configs/shhs/multi_modal/eeg_eog/established_models/fourier_transformer_eeg_eog_mat_BIOBLIP_twomode_caouter_al1_shared_b16_t045.json",
        # "./configs/shhs/multi_modal/eeg_eog/established_models/fourier_transformer_eeg_eog_mat_BIOBLIP_twomode_caouter_al1_shared_b16_t045.json",
        # "./configs/shhs/multi_modal/eeg_eog/established_models/fourier_transformer_eeg_eog_mat_early_concat_onlyi.json",
        # "./configs/shhs/multi_modal/eeg_eog/established_models/fourier_transformer_eeg_eog_mat_early_concat_onlyi_t25.json",
        # "./configs/shhs/multi_modal/eeg_eog/established_models/fourier_transformer_eeg_eog_mat_early_concat_onlyi_t25.json",

        # "./users/sista/kkontras/Documents/Sleep_Project/configs/shhs/multi_modal/eeg_eog/established_models/fourier_transformer_eeg_eog_mat_BIOBLIP_twomode_caouter_al1_shared_b16_t01.json",
        # "./users/sista/kkontras/Documents/Sleep_Project/configs/shhs/multi_modal/eeg_eog/established_models/fourier_transformer_eeg_eog_mat_BIOBLIP_twomode_caouter_al1_shared_b16_t025.json",
        # "./configs/shhs/multi_modal/eeg_eog/established_models/fourier_transformer_eeg_eog_mat_BIOBLIP_twomode_caouter_al1_shared_b16_t045.json",
        # "./configs/shhs/multi_modal/eeg_eog/established_models/fourier_transformer_eeg_eog_mat_BIOBLIP_twomode_caouter_al1_shared_b16_t033.json",

        # "./configs/shhs/multi_modal/eeg_eog/established_models/fourier_transformer_eeg_eog_mat_BIOBLIP_twomode_caouter_al1_shared_b16_mult_freetrain_multilr.json"
        # "./configs/shhs/multi_modal/eeg_eog/established_models/fourier_transformer_eeg_eog_mat_BIOBLIP_twomode_caouter_al1_shared_b16_mult_freetrain_lr3_w.json"
        # "./configs/shhs/multi_modal/eeg_eog/established_models/fourier_transformer_eeg_eog_mat_BIOBLIP_twomode_caouter_al1_shared_b16_mult_freetrain_lr3_clean.json"
        # "./configs/shhs/multi_modal/eeg_eog/established_models/fourier_transformer_eeg_eog_mat_BIOBLIP_twomode_caouter_al1_shared_fullca.json"

        # "configs/shhs/multi_modal/eeg_eog/established_models/fourier_transformer_eeg_eog_mat_BIOBLIP_twomode_caouter_onlyal.json",
        # "configs/shhs/multi_modal/eeg_eog/established_models/fourier_transformer_eeg_eog_mat_BIOBLIP_twomode_caouter_onlyal_same.json",
        # "./configs/shhs/multi_modal/eeg_eog/established_models/fourier_transformer_eeg_eog_mat_BIOBLIP_twomode_caouter_al1_shared_b16_mult_freetrain_lr3_cliplike.json",
        "./configs/shhs/multi_modal/eeg_eog/established_models/fourier_transformer_eeg_eog_mat_BIOBLIP_twomode_caouter_al1_shared_b16_mult_freetrain_lr3_cliplike_prevv.json",
        # "./configs/shhs/multi_modal/eeg_eog/established_models/fourier_transformer_eeg_eog_mat_BIOBLIP_twomode_caouter_al01_shared_b16_mult_freetrain_lr3_cliplike.json",
        # "./configs/shhs/multi_modal/eeg_eog/established_models/fourier_transformer_eeg_eog_mat_BIOBLIP_twomode_caouter_al0_shared_b16_mult_freetrain_lr3_cliplike.json"
    ]
    finals = []
    for i in config_list:
        config = process_config(i)
        for fold in range(1):
            print("We are in fold {}".format(fold))
            config.fold = fold
            agent_class = globals()[config.agent] 
            agent = agent_class(config)
            acc, f1, k, per_class_f1 = agent.run()
            agent.finalize()
            if (config.res):
                finals.append([acc, f1, k, per_class_f1])
            print(finals)
            del agent

    # for i, f in enumerate(finals):
    #     print("Fold {}: {}".format(i,f))
    # acc, f1, k, m = [], [], [], []
    # for f in finals:
    #     acc.append(f[0])
    #     f1.append(f[1])
    #     k.append(f[2])
    #     m.append(f[3])
    # print("Acc: {0:.4f}".format(np.array(acc).mean()))
    # print("F1: {0:.4f}".format(np.array(f1).mean()))
    # print("K: {0:.4f}".format(np.array(k).mean()))
    # print(np.array(m).mean(axis=0))


main()