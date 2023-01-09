import argparse
from utils.config import process_config

from agents.sleep_test import *

import matplotlib.pyplot as plt
import numpy as np


def main():
    config_list = [
    # "./configs/sleep_edf/multi_modal/eeg_eog/established_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos.json",
    "./configs/sleep_edf/single_channel/fourier_transformer_eog_mat.json"
    ]
    for i in config_list:
        for fold in range(5):

            config = process_config(i)
            print("We are in fold {}".format(fold))
            config.dataset.data_split.fold = fold
            config.model.save_dir = config.model.save_dir.format(fold)
            agent_class = globals()[config.agent]
            agent = agent_class(config)
            # agent.check_energies_per_class()
            agent.run()
            agent.finalize()

            del agent

main()