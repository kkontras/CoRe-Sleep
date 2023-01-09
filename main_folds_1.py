import argparse
from utils.config import process_config

from agents.sleep_test import *

import matplotlib.pyplot as plt
import numpy as np


def main():
    config_list = [
        # "./configs/sleep_edf/cnn/cnn.json",
        # "./configs/sleep_edf/cnn/cnn_fusion_only.json",
        "./configs/sleep_edf/cnn/cnn.json",
        # "./configs/sleep_edf/cnn/cnn_tf.json",
        # "./configs/sleep_edf/cnn/cnn_fusion_3.json",
        # "./configs/sleep_edf/cnn/cnn_fusion_22.json",
        # "./configs/sleep_edf/cnn/cnn_fusion_conv1_1.json",
        # "./configs/sleep_edf/cnn/cnn_fusion_conv1_2.json",
        # "./configs/sleep_edf/cnn/cnn_fusion_conv1_3.json",
        # "./configs/sleep_edf/transformers/tf_big_small.json",
        # "./configs/sleep_edf/transformers/tf_ch_viproj_small_small.json",
    ]
    finals = []

    for fold in range(10):
        print("We are in fold {}".format(fold))
        for i in config_list:
            config = process_config(i)
            config.fold = fold
            agent_class = globals()[config.agent]
            agent = agent_class(config)
            acc, f1, k, per_class_f1 = agent.run()
            agent.finalize()
            if (config.res):
                finals.append([acc, f1, k, per_class_f1])
            print(finals)
            del agent
    print(finals)
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