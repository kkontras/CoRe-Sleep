import argparse
from utils.config import process_config

from agents.sleep_test import *

import matplotlib.pyplot as plt
import numpy as np

def main():
    config_list = [
        # "./configs/sleep_edf/multi_modal/eeg_eog/fourier_transformer_eeg_eog_mat_BIOBLIP.json",
        # "./configs/sleep_edf/multi_modal/eeg_eog/fourier_transformer_eeg_eog_mat_BIOBLIP_shared.json",
        # "./configs/sleep_edf/multi_modal/eeg_eog/fourier_transformer_eeg_eog_mat_merged.json",

        # "./configs/sleep_edf/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos.json"
        # "./configs/sleep_edf/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nowei.json"
        # "./configs/sleep_edf/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_dropout03_nowei.json"
        # "./configs/sleep_edf/single_channel/fourier_transformer_eeg_mat_drop01.json"
        # "./configs/sleep_edf/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_dropout01_nowei_possin.json"
        # "./configs/sleep_edf/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_dropout01_nowei_possin_trial2.json"
        # "./configs/sleep_edf/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_dropout01_nowei_possin_trial5.json"
        # "./configs/sleep_edf/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_dropout01_nowei_possin_pretrainedal.json"

        "./configs/paper_finals/sleep-edf78/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_nonsharedpreds_drop01_aligninner_possin_trial2.json",
        "./configs/paper_finals/sleep-edf78/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_sharedpreds_drop03_aligninner_possin_trial2.json",
        "./configs/paper_finals/sleep-edf78/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_sharedpreds_drop03_aligninner_nopos_trial2.json",
        "./configs/paper_finals/sleep-edf78/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_nonsharedpreds_drop03_aligninner_possin_trial2.json",
        "./configs/paper_finals/sleep-edf78/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_sharedpreds_drop03_aligninner_possin_small_trial2.json",
    ]
    finals = []
    for i in config_list:
        for fold in range(8,9):
            config = process_config(i)
            print("We are in fold {}".format(fold))
            config.dataset.data_split.fold = fold
            config.model.save_dir = config.model.save_dir.format(fold)
            agent_class = globals()[config.agent]
            agent = agent_class(config)
            agent.run()
            agent.finalize()
            del agent

main()