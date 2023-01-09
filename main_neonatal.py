import argparse
from utils.config import process_config

from agents.sleep_test import *

import matplotlib.pyplot as plt
import numpy as np


def main():
    config_list = [
        "./configs/neonatal/fourier_transformer_eeg_mat.json"
    ]

    for i in config_list:
        config = process_config(i)
        agent_class = globals()[config.agent]
        agent = agent_class(config)
        agent.run()
        agent.finalize()

main()