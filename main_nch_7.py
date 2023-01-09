import argparse
from utils.config import process_config

from agents.sleep_test import *

import matplotlib.pyplot as plt
import numpy as np


def main():
    config_list = [
        # "./configs/nch/single_channel/fourier_transformer_eeg_mat_emphasisonN1.json",
        # "./configs/nch/single_channel/fourier_transformer_eog_mat_emphasisonN1.json",
        # "./configs/nch/single_channel/fourier_transformer_emg_mat_emphasisonN1.json",
        # "./configs/nch/multi_modal/fourier_transformer_eeg_eog_merged_mat.json",
        # "./configs/nch/multi_modal/fourier_transformer_eeg_eog_late_mat.json",
        # "./configs/nch/multi_modal/fourier_transformer_eeg_eog_bottleneck_mat.json",
        "./configs/nch/multi_modal/fourier_transformer_eeg_eog_merged_channels_mat.json",
        # "./configs/nch/multi_channel/fourier_transformer_multichannel_eeg.json",
        # "./configs/nch/multi_channel/fourier_transformer_multichannel_eog_w.json"
    ]
    finals = []
    for i in config_list:
        config = process_config(i)
        agent_class = globals()[config.agent]
        agent = agent_class(config)
        agent.run()
        res = agent.finalize()
        if (config.res):
            finals.append(res)
        del agent
    print(finals)


main()