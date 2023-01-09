import argparse
from utils.config import process_config

from agents.sleep_test import *

import matplotlib.pyplot as plt
import numpy as np
import random


def main():
    config_list = [
        # "./configs/sleepnet/eeg.json",
        # "./configs/sleepnet/stft.json",
        # "./configs/sleepnet/fusion.json",
        # "./configs/sleepnet/fusion_att.json",
        # "./configs/sleepnet/fusion_seq_big.json",
        # "./configs/sleepnet/fusion_seq_big_freezed.json",
        "./configs/sleepnet/fusion_seq_big_unpre.json",
    ]
    finals = []
    for i in config_list:
        for j in range(10):
            a = random.randrange(500)
            config = process_config(i)
            config.seed = a
            print(a)
            config.model_class = "B_Concat_FC"
            config.save_dir="/users/sista/kkontras/Documents/Sleep_Project/data/2021_data/fusion_results/V1_{}_unpre_try0{}.pth.tar".format(config.model_class,f'{j:02}')
            agent_class = globals()[config.agent]
            agent = agent_class(config)
            agent.run()
            res = agent.finalize()
            if (config.res):
                finals.append(res)
            del agent
    print(finals)


main()