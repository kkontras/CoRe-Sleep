import argparse
from utils.config import process_config

from agents.sleep_test import *

import matplotlib.pyplot as plt
import numpy as np


def main():
    config_list = [
        # "./configs/sleepnet/eeg.json",
        # "./configs/sleepnet/eeg_channels.json",
        # "./configs/sleepnet/eeg_1ch.json",
        # "./configs/sleep_edf/cnn/cnn.json",
        # "./configs/sleepnet/eeg_seq.json",
        # "./configs/sleepnet/eeg_att.json",
        # "./configs/sleepnet/stft.json",
        # "./configs/sleepnet/stft_att.json",
        # "./configs/sleepnet/fusion.json",
        # "./configs/sleepnet/fusion_att.json",
        # "./configs/sleepnet/fusion_MulT.json",
        # "./configs/sleepnet/fusion_seq_small.json",
        # "./configs/sleepnet/fusion_seq_big_test.json",
        # "./configs/sleepnet/fusion_xbig.json",
        # "./configs/sleepnet/fusion_xbig_duo.json",
        # "./configs/sleepnet/fusion_xbig_duo_matt.json",
        # "./configs/sleepnet/fusion_xbig_duo_satt.json",
        # "./configs/sleepnet/fusion_xbig_duo_eatt.json",
        # "./configs/sleepnet/fusion_xbig_quad.json",
        # "./configs/sleepnet/config_eeg_unet.json",
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