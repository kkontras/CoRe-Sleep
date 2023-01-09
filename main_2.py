import argparse
from utils.config import process_config

from agents.sleep_test import *

import matplotlib.pyplot as plt
import numpy as np




def main():
    config_list = [
        "./configs/sleepnet/transformers/transformer2_viproj_big_smallff.json",
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