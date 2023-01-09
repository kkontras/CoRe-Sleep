#!/esat/smcdata/users/kkontras/Image_Dataset/no_backup/envs/gl_env/bin/python
import argparse
from utils.config import process_config

from agents.sleep_test import *

import matplotlib.pyplot as plt
import numpy as npvenir



def main():
    config_list = [
        # "./configs/shhs/reconstruction/vae_eοg.json",
        # "./configs/shhs/single_channel/fourier_transformer_eeg_connepoch.json",
        # "./configs/shhs/single_channel/fourier_transformer_cls_eeg_mat_adv.json"
        # "./configs/shhs/multi_modal/eeg_eog/established_models/fourier_transformer_eeg_eog_mat_BIOBLIP_rpos_adv.json",
        # "./configs/shhs/multi_modal/eeg_eog/established_models/fourier_transformer_eeg_eog_mat_BIOBLIP_rpos_adv_neigh.json"
        # "./configs/shhs/multi_modal/eeg_eog/fourier_transformer_eeg_eog_mat_merged_HPFC.json",
        # "./configs/nch/multi_modal/established_models/fourier_transformer_eeg_eog_mat_BIOBLIP_rpos_adv_shared_temp.json"
        # "./configs/nch/multi_modal/established_models/fourier_transformer_eeg_eog_mat_BIOBLIP_rpos_adv_temp.json",
        # "./configs/shhs/multi_modal/eeg_eog/established_models/fourier_transformer_eeg_eog_mat_BIOBLIP_twote.json"
        # "./configs/shhs/multi_modal/eeg_eog/established_models/fourier_transformer_eeg_eog_mat_BIOBLIP_lossw_pretrainedNCH_onlyalign_LE.json",
        # "./configs/nch/multi_modal/established_models/fourier_transformer_eeg_eog_mat_merged_rpos_adv_temp.json"

    # "./configs/shhs/single_channel/fourier_transformer_eeg_long.json",
    # "./configs/shhs/multi_modal/eeg_eog/established_models/fourier_transformer_eeg_eog_mat_early_concat.json",
    # "./configs/shhs/multi_modal/eeg_eog/established_models/fourier_transformer_eeg_eog_mat_BIOBLIP_twomode_caouter_al01_shared_b16_t025.json",
    # "./configs/shhs/multi_modal/eeg_eog/established_models/fourier_transformer_eeg_eog_mat_BIOBLIP_twomode_caouter_al01_shared_b16_t033.json",
    # "./configs/shhs/multi_modal/eeg_eog/established_models/fourier_transformer_eeg_eog_mat_BIOBLIP_twomode_caouter_al1_shared_b16.json",
    # "./configs/shhs/multi_modal/eeg_eog/established_models/fourier_transformer_eeg_eog_mat_BIOBLIP_twomode_caouter_al1_shared_b16_mult.json",
    # "./configs/shhs/multi_modal/eeg_eog/established_models/fourier_transformer_eeg_eog_mat_BIOBLIP_twomode_caouter_al1_shared_b16_t025.json",
    # "./configs/shhs/multi_modal/eeg_eog/established_models/fourier_transformer_eeg_eog_mat_BIOBLIP_twomode_caouter_al1_shared_b16_t045.json",
    # "./configs/shhs/multi_modal/eeg_eog/established_models/fourier_transformer_eeg_eog_mat_BIOBLIP_twomode_caouter_al1_shared_b16_t045.json",
    # "./configs/shhs/multi_modal/eeg_eog/established_models/fourier_transformer_eeg_eog_mat_early_concat_onlyi_t25.json",
    # "./configs/shhs/multi_modal/eeg_eog/established_models/fourier_transformer_eeg_eog_mat_BIOBLIP_twomode_caouter_al1_shared_VAE.json",

    # "./users/sista/kkontras/Documents/Sleep_Project/configs/shhs/multi_modal/eeg_eog/established_models/fourier_transformer_eeg_eog_mat_BIOBLIP_twomode_caouter_al1_shared_b16_t01.json",
    # "./configs/shhs/multi_modal/eeg_eog/established_models/fourier_transformer_eeg_eog_mat_BIOBLIP_twomode_caouter_al1_shared_b16_t025.json",
    # "./users/sista/kkontras/Documents/Sleep_Project/configs/shhs/multi_modal/eeg_eog/established_models/fourier_transformer_eeg_eog_mat_BIOBLIP_twomode_caouter_al1_shared_b16_t045.json",
    # "./con/figs/shhs/multi_modal/eeg_eog/established_models/fourier_transformer_eeg_eog_mat_BIOBLIP_twomode_caouter_al1_shared_b16_mult_freetrain_lr3.json",
    "./configs/shhs/multi_modal/eeg_eog/established_models/fourier_transformer_eeg_eog_mat_BIOBLIP_twomode_caouter_al1_shared_b16_mult_freetrain_multilr.json",

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