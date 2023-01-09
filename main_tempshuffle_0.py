import argparse
from utils.config import process_config

from agents.sleep_test import *

import matplotlib.pyplot as plt
import numpy as np


def main():
    config_list = [
        # "./configs/shhs/tempshuffle/tempshuffle_pretraining.json",
        # "./configs/shhs/tempshuffle/tempshuffle_order_pretraining.json",
        # "./configs/shhs/tempshuffle/fourier_transformer_cls_eeg_mat_no_pretrained_benchmark_500.json",
        "./configs/shhs/border/border_pretraining.json"
        # "./configs/shhs/tempshuffle/fourier_transformer_cls_eeg_mat_tempshuffle_pretrained.json",
        # "./configs/shhs/tempshuffle/fourier_transformer_cls_eeg_mat_tempshuffle_p5.json",
        "./configs/shhs/tempshuffle/fourier_transformer_cls_eeg_mat_no_pretrained_benchmark_5.json"
        "./configs/shhs/tempshuffle/fourier_transformer_cls_eeg_mat_tempshuffle_pfull.json",
        "./configs/shhs/tempshuffle/fourier_transformer_cls_eeg_mat_no_pretrained_benchmark_500.json",

    ]
    for i in config_list:
        config = process_config(i)
        agent_class = globals()[config.agent]
        agent = agent_class(config)
        agent.run()
        agent.finalize()
        del agent

main()