import argparse
from utils.config import process_config

from agents.sleep_test import *

import matplotlib.pyplot as plt
import numpy as np


def main():
    config_list = [
        "./configs/shhs/single_channel/fourier_transformer_cls_eeg_conv.json",
        # "./configs/shhs/multi_modal/eeg_eog/fourier_transformer_eeg_eog_mat_merged_v2.json",
        # "./configs/shhs/multi_modal/eeg_eog/fourier_transformer_eeg_eog_mat_merged_rpos.json",
        # "./configs/shhs/multi_modal/eeg_eog/fourier_transformer_eeg_eog_mat_merged_glearnedbias_rpos.json",

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