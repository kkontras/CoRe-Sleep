import argparse
from utils.config import process_config

from agents.sleep_test import *

import matplotlib.pyplot as plt
import numpy as np
import optuna

def objective(trial):
    config_i = "./configs/shhs/multi_modal/eeg_eog/established_models/fourier_transformer_eeg_eog_mat_BIOBLIP.json"

    config = process_config(config_i, printing=False)

    config.model.args.multi_loss.multi_loss_weights.alignment_loss = trial.suggest_float("alignment_penalty_parameter", 1e-5, 10, log=True)
    # config.optimizer.weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)
    # config.optimizer.type = trial.suggest_categorical("optimizer", [ "Adam", "[MomentumSGD"])
    # config.training_params.batch_size = trial.suggest_int("batch_size", 32, 128, step=32)
    config.model.save_dir = config.model.save_dir.format(config.model.args.multi_loss.multi_loss_weights.alignment_loss)

    print("Alignment rate is {}".format(config.model.args.multi_loss.multi_loss_weights.alignment_loss))
    # print("weight_decay is {}".format(config.optimizer.weight_decay))
    # print("Save_dir is {}".format(config.model.save_dir))
    # print("momentum is {}".format(config.optimizer.momentum))
    # print("optimizer is {}".format(config.optimizer.type))
    # print("batch_size is {}".format(config.training_params.batch_size))

    agent_class = globals()[config.agent]
    agent = agent_class(config)
    agent.run()
    loss = agent.finalize()
    return loss

if __name__ == "__main__":
    # study = optuna.create_study()
    study = optuna.create_study(pruner=optuna.pruners.MedianPruner(), storage='sqlite:///fourier_optuna_transformer_eeg_eog_mat_BIOBLIP_1.db', load_if_exists=True)
    study.optimize(objective, n_trials=10)