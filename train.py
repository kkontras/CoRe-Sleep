from utils.config import process_config_default, setup_logger
from agents.sleep_agent import *

# xrandr --output DP-4 --scale 0.8x0.8

import argparse
import logging

def main(config_path, default_config_path, args):
    setup_logger()

    config = process_config_default(config_path, default_config_path)

    m = ""
    enc_m = ""

    if "fold" in args and args.fold is not None:
        config.dataset.data_split.fold = int(args.fold)
        config.dataset.fold = int(args.fold)
        m += "fold{}".format(args.fold)
        if args.fold=="0":
            config.dataset.data_split.split_method= "patients_sleeptransformer"
        else:
            config.dataset.data_split.split_method= "patients_test"
        seeds = [35, 28, 35]
        config.training_params.seed = int(seeds[int(args.fold)])
        if hasattr(config.model, "encoders"):
            for i in range(len(config.model.encoders)):
                config.model.encoders[i].pretrainedEncoder.dir = config.model.encoders[i].pretrainedEncoder.dir.format(args.fold)
    if "al" in args and args.al is not None:
        config.model.args.multi_loss.alignment_loss = float(args.al)
        m += "_al{}".format(args.al)
    if "ms" in args and args.ms is not None:
        config.model.args.multi_loss.supervised_losses["eeg"] = float(args.ms)
        config.model.args.multi_loss.supervised_losses["eog"] = float(args.ms)
        m += "_ms{}".format(args.ms)
    if "incboth" in args and args.incboth is not None:
        config.dataset.filter_patients.train.subsets.combined = int(args.incboth)
        m += "_incboth{}".format(args.incboth)
    if "inceeg" in args and args.inceeg is not None:
        config.dataset.filter_patients.train.subsets.eeg = int(args.inceeg)
        m += "_inceeg{}".format(args.inceeg)
    if "inceog" in args and args.inceog is not None:
        config.dataset.filter_patients.train.subsets.eog = int(args.inceog)
        m += "_inceog{}".format(args.inceog)

    config.model.save_dir = config.model.save_dir.format(m)

    if hasattr(config.model, "encoders"):
        for i in range(len(config.model.encoders)):
            config.model.encoders[i].pretrainedEncoder.dir = config.model.encoders[i].pretrainedEncoder.dir.format(enc_m)

    logging.info("save_dir: {}".format(config.model.save_dir))
    agent_class = globals()[config.agent]
    agent = agent_class(config)
    agent.run()
    agent.finalize()


parser = argparse.ArgumentParser(description="My Command Line Program")
parser.add_argument('--config', help="config file path")
parser.add_argument('--default_config', help="default config file path")
parser.add_argument('--fold', help="fold number")
parser.add_argument('--al', help="Alignment loss hyperparameter")
parser.add_argument('--ms', help="Multisupervised loss hyperparameter")
parser.add_argument('--incboth', help="Include patients with both EEG and EOG")
parser.add_argument('--inceeg', help="Include patients with only EEG")
parser.add_argument('--inceog', help="Include patients with only EOG")
args = parser.parse_args()

for var_name in vars(args):
    var_value = getattr(args, var_name)
    if var_value == "None":
        setattr(args, var_name, None)

print(args)

main(config_path=args.config, default_config_path=args.default_config, args=args)