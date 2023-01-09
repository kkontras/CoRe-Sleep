import argparse
from utils.config import process_config

from agents.sleep_test import *

import matplotlib.pyplot as plt
import numpy as np

def main():

    config_list = [
        # "./configs/sleepnet/eeg_test.json",
        # "./configs/sleepnet/stft_test-t.json",
        "./configs/sleepnet/fusion_test.json",
    ]
    num_models = 13
    pp_list = [15]
    pp_results, pp_results_ens = [],[]
    for pp in pp_list:
        test_res, finals = [], []
        for z in range(0,1):
            save_dirs, model_class = [], []
            save_root="/users/sista/kkontras/Documents/Sleep_Project/data/"
            # save_root="/users/sista/kkontras/Documents/Sleep_Project/data/stft_results_{}/".format(z)
            # save_root="/users/sista/kkontras/Documents/Sleep_Project/data//eeg_stft_fusion_{}/".format(z)
            for j in range(num_models):
                # save_dirs.append("exp0_try0{}.pth.tar".format(f'{j:02}'))  # seed=52
                # save_dirs.append("exp1_try0{}.pth.tar".format(f'{j:02}')) #seed=52
                # model_class.append("EEG_CNN_6")
                # model_class.append("STFT_CNN")
                # model_class.append("STFT_EEG_CNN_1")

                # save_dirs.append("exp2_try0{}.pth.tar".format(f'{j:02}')) #seed=52
                # save_dirs.append("eeg_results_{}/exp2_try0{}.pth.tar".format(z,f'{j:02}')) #seed=52
                # save_dirs.append("eeg_results_{}/exp3_try0{}.pth.tar".format(z,f'{j:02}')) #seed=52
                # save_dirs.append("stft_results_{}/exp2_try0{}.pth.tar".format(z,f'{j:02}')) #seed=52
                # save_dirs.append("stft_results_{}/exp3_try0{}.pth.tar".format(z,f'{j:02}')) #seed=52
                save_dirs.append("att_fusion/fusion_results_{}/exp2_try0{}.pth.tar".format(z,f'{j:02}')) #seed=52
                model_class.append("STFT_EEG_COATT")
                # model_class.append("EEG_CNN_7")
                # model_class.append("STFT_CNN_2")
                # model_class.append("STFT_CNN_2")
                # model_class.append("STFT_EEG_CNN_2")

                # save_dirs.append("exp4_try0{}.pth.tar".format(f'{j:02}')) #seed=52
                # save_dirs.append("exp5_try0{}.pth.tar".format(f'{j:02}')) #seed=52
                # model_class.append("EEG_CNN_8")
                # model_class.append("STFT_CNN_3")
                # model_class.append("STFT_EEG_CNN_3")

            for i, c in enumerate(config_list):
                config = process_config(c)
                config.save_dirs = save_dirs
                config.save_root = save_root
                config.model_class = model_class
                config.post_proc_step = pp
                config.num_modalities = 2
                agent_class = globals()[config.agent]
                agent = agent_class(config)
                agent.run()
                res, k, _ = agent.finalize()
                test_res.append(res)
                finals.append(k)
                del agent
        print("For single net Kappa values are mean: {0:.4f} and std: {1:.4f}".format(np.array(test_res).flatten().mean(),np.array(test_res).flatten().std()))
        print("Ensemble Kappa values are mean: {0:.4f} and std: {1:.4f}".format(np.array(finals).mean(),np.array(finals).std()))
        pp_results.append(np.array(test_res).flatten().mean())
        pp_results_ens.append(np.array(finals).mean())
    print(pp_results)
    print(pp_results_ens)
main()