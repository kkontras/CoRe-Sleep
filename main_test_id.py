import argparse
from utils.config import process_config

from agents.sleep_test import *

import matplotlib.pyplot as plt
import numpy as np

def main():
    config_list = [
        "./configs/sleepnet/eeg_test.json",
        # "./configs/sleepnet/eeg_test.json",
        "./configs/sleepnet/stft_test-t.json",
        # "./configs/sleepnet/fusion_test-t.json",
    ]
    sr=["/users/sista/kkontras/Documents/Sleep_Project/data/eeg_results_0/", "/users/sista/kkontras/Documents/Sleep_Project/data/stft_results_0/"]
    md = ["EEG_CNN_8","STFT_CNN_3"]
    save_dirs, model_class = [],[]
    test_res, finals, ids = [], [], []
    m = []
    for i in range(13):
        sv = ["exp5_try0{}.pth.tar".format(f'{i:02}'),"exp5_try0{}.pth.tar".format(f'{i:02}')]
        test_res, finals, ids = [], [], []

        for j in range(2):
            save_root = sr[j]
            save_dirs = [sv[j]]
            model_class = [md[j]]

        # save_root="/users/sista/kkontras/Documents/Sleep_Project/data//eeg_stft_fusion_{}/".format(z)

        # save_dirs.append("exp0_try0{}.pth.tar".format(f'{j:02}'))  # seed=52
        # save_dirs.append("exp1_try0{}.pth.tar".format(f'{j:02}')) #seed=52
        # model_class.append("EEG_CNN_6")
        # model_class.append("STFT_CNN")
        # model_class.append("STFT_EEG_CNN_1")

        # save_dirs.append("exp2_try0{}.pth.tar".format(f'{j:02}')) #seed=52
        # save_dirs.append("exp3_try0{}.pth.tar".format(f'{j:02}')) #seed=52
        # model_class.append("EEG_CNN_7")
        # model_class.append("EEG_CNN_7")
        # model_class.append("STFT_CNN_2")
        # model_class.append("STFT_EEG_CNN_2")

        # save_dirs.append("exp4_try0{}.pth.tar".format(f'{j:02}')) #seed=52
        # save_dirs.append("exp5_try0{}.pth.tar".format(f'{j:02}')) #seed=52
        # model_class.append("EEG_CNN_8")
        # model_class.append("STFT_CNN_3")
        # model_class.append("STFT_EEG_CNN_3")
            config = process_config(config_list[j])
            config.save_dirs = save_dirs
            config.save_root = save_root
            config.model_class = model_class
            config.post_proc_step = 15
            agent_class = globals()[config.agent]
            agent = agent_class(config)
            agent.run()
            res, k, id = agent.finalize()
            test_res.append(res)
            finals.append(k)
            ids.append(id["found"])
            del agent
        s = 0
        d = 0
        for i in range(len(ids[0])):
            if ids[0][i]==0 or ids[1][i]==0:
                d += 1
                if ids[0][i] == ids[1][i]:
                    s += 1
        print("Out of the {0:.0f} that models made wrong they overlap in {1:.2f}%".format(d, 100*s/d))
        m.append(100*s/d)
    print("Mean overlap {}".format(np.array(m).mean()))
    print("For single net Kappa values are mean: {0:.4f} and std: {1:.4f}".format(np.array(test_res).flatten().mean(),np.array(test_res).flatten().std()))
    print("Ensemble Kappa values are mean: {0:.4f} and std: {1:.4f}".format(np.array(finals).mean(),np.array(finals).std()))
main()