import argparse
from utils.config import process_config

from agents.sleep_test import *

import matplotlib.pyplot as plt
import numpy as np
import random


def main():
    config_list = [
        # "./configs/sleepnet/eeg_pretrained.json",
        # "./configs/sleepnet/stft.json",
        "./configs/sleepnet/fusion_pretrained.json",
    ]
    num_models = 13
    finals = []
    a = random.randrange(500)
    print(a)
    for i in config_list:
        for j in range(num_models):
            config = process_config(i)
            config.seed = a
            print(j)
            config.rand_split = j
            config.rand_splits = num_models
            config.model_class = "STFT_EEG_CNN_3"
            config.encoder_models =  [["EEG_Encoder",3],["STFT_Encoder",3]]
            config.save_dir_encoder = ["/users/sista/kkontras/Documents/Sleep_Project/data/encoders/eeg_type3_c_encoder.pth.tar",
                                       "/users/sista/kkontras/Documents/Sleep_Project/data/encoders/stft_type3_c_encoder.pth.tar"]

            # config.model_class = "STFT_CNN_3"
            # config.model_class = "STFT_EEG_CNN_3"
            config.data_roots = "/esat/stadiustempdatasets/sleep_data/kkontras/Image_Dataset/Version_1"
            config.save_dir="/users/sista/kkontras/Documents/Sleep_Project/data/encoders/fusion_train/eeg_stft_fusion_2/exp4_try0{}.pth.tar".format(f'{j:02}')
            # config.save_dir="/users/sista/kkontras/Documents/Sleep_Project/data/stft_results_1/exp4_try0{}.pth.tar".format(f'{j:02}')
            # config.save_dir="/users/sista/kkontras/Documents/Sleep_Project/data/eeg_stft_fusion_2/exp4_try0{}.pth.tar".format(f'{j:02}')
            agent_class = globals()[config.agent]
            agent = agent_class(config)
            agent.run()
            res = agent.finalize()
            if res < 0.68:
                j-=1
            if (config.res):
                finals.append(res)
            del agent
    print(finals)


main()