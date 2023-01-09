import argparse
from utils.config import process_config

from agents.sleep_test import *

import matplotlib.pyplot as plt
import numpy as np


def main():
    config_list = [
        # "./configs/shhs/multi_modal/eeg_eog/contrastive_pre/fourier_transformer_eeg_eog_mat_late_gbiasedm.json",
        # "./configs/shhs/multi_modal/eeg_eog/contrastive_pre/fourier_transformer_eeg_eog_mat_late_gbiasedm_plus.json",
        # "./configs/shhs/multi_modal/eeg_eog/contrastive_pre/fourier_transformer_eeg_eog_mat_late_neighbiasedm5.json",
        # "./configs/shhs/multi_modal/eeg_eog/contrastive_pre/fourier_transformer_eeg_eog_mat_late_neighbiasedm3.json",
        # "./configs/shhs/multi_modal/eeg_eog/contrastive_pre/fourier_transformer_eeg_eog_mat_late_neighbiasedm3_plus.json",
        # "./configs/shhs/multi_modal/eeg_eog/contrastive_pre/fourier_transformer_eeg_eog_mat_late_glearnedbiasedm.json",
        # "./configs/shhs/multi_modal/eeg_eog/contrastive_pre/fourier_transformer_eeg_eog_mat_late_glearnedbiasedm_plus.json",
        # "./configs/shhs/multi_modal/eeg_eog/established_models/fourier_transformer_eeg_eog_mat_BIOBLIP_twomode_caouter_al01_shared_b16_t025_discards.json",
        # "./configs/shhs/multi_modal/eeg_eog/established_models/fourier_transformer_eeg_eog_mat_BIOBLIP_twomode_caouter_al1_shared_b16_t045.json",
        # "./configs/shhs/multi_modal/eeg_eog/established_models/fourier_transformer_eeg_eog_mat_early_concat_onlyi_t25.json",

    # "./configs/shhs/multi_modal/eeg_eog/established_models/fourier_transformer_eeg_eog_mat_BIOBLIP_twomode_caouter_al01_shared_b16_t025.json",
    # "./configs/shhs/multi_modal/eeg_eog/established_models/fourier_transformer_eeg_eog_mat_BIOBLIP_twomode_caouter_al01_shared_b16_t033.json",
    # "./configs/shhs/multi_modal/eeg_eog/established_models/fourier_transformer_eeg_eog_mat_BIOBLIP_twomode_caouter_al1_shared_b16.json",
    # "./configs/shhs/multi_modal/eeg_eog/established_models/fourier_transformer_eeg_eog_mat_BIOBLIP_twomode_caouter_al1_shared_b16_t025.json",
    # "./configs/shhs/multi_modal/eeg_eog/established_models/fourier_transformer_eeg_eog_mat_BIOBLIP_twomode_caouter_al1_shared_b16_t045.json",
    "./configs/shhs/multi_modal/eeg_eog/established_models/fourier_transformer_eeg_eog_mat_BIOBLIP_twomode_caouter_al1_shared_b16_t045.json",
    # "./configs/shhs/multi_modal/eeg_eog/established_models/fourier_transformer_eeg_eog_mat_early_concat_onlyi_t25.json",

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

    for i, f in enumerate(finals):
        print("Fold {}: {}".format(i,f))
    acc, f1, k, m = [], [], [], []
    for f in finals:
        acc.append(f[0])
        f1.append(f[1])
        k.append(f[2])
        m.append(f[3])
    print("Acc: {0:.4f}".format(np.array(acc).mean()))
    print("F1: {0:.4f}".format(np.array(f1).mean()))
    print("K: {0:.4f}".format(np.array(k).mean()))
    print(np.array(m).mean(axis=0))


main()