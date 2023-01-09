import argparse
from utils.config import process_config

from agents.sleep_test import *

import matplotlib.pyplot as plt
import numpy as np


def main():
    config_list = [
        "./configs/nch/multi_modal/established_models/fourier_transformer_eeg_eog_mat_BIOBLIP_rpos_adv.json"
        # "./configs/nch/multi_modal/established_models/fourier_transformer_eeg_eog_mat_bottleneck_lim0_rpos_adv.json"
        # "./configs/nch/multi_modal/established_models/fourier_transformer_eeg_eog_mat_late_glearnedbiasedm_outerplus_rpos_adv.json"
        # "./configs/nch/multi_modal/established_models/fourier_transformer_eeg_eog_mat_merged_rpos_adv.json"
        # "./configs/nch/multi_modal/established_models/fourier_transformer_eeg_eog_mat_BIOBLIP_rpos_adv_temp.json"
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