
import numpy as np
import argparse
from colorama import Fore
from posthoc.Helpers.Helper_Importer import Importer
from posthoc.Helpers.Helper_Validator import Validator
from utils.config import process_config, setup_logger
from collections import defaultdict
from utils.deterministic_pytorch import deterministic

def test_each(config_path, default_config_path, args):
    setup_logger()

    importer = Importer(config_name=config_path, default_files=default_config_path, device="cuda:0")

    m = ""
    enc_m = ""
    if "fold" in args and args.fold is not None:
        importer.config.dataset.data_split.fold = int(args.fold)
        importer.config.dataset.fold = int(args.fold)
        if args.fold=="0":
            importer.config.dataset.data_split.split_method= "patients_huy"
        else:
            importer.config.dataset.data_split.split_method= "patients_test"
        m += "fold{}".format(args.fold)
        enc_m += "fold{}".format(args.fold)
        seeds = [35, 28, 35]
        importer.config.training_params.seed = int(seeds[int(args.fold)])
    if "al" in args and args.al is not None:
        m += "_al{}".format(args.al)
    if "ms" in args and args.ms is not None:
        m += "_ms{}".format(args.ms)
    if "incboth" in args and args.incboth is not None:
        m += "_incboth{}".format(args.incboth)
    if "inceeg" in args and args.inceeg is not None:
        m += "_inceeg{}".format(args.inceeg)
    if "inceog" in args and args.inceog is not None:
        m += "_inceog{}".format(args.inceog)

    if args.noisy:
        importer.config.dataset.filter_patients = {
            "total": {"use_type": False, "include_skipped": False},
            "train": {"use_type": False, "include_skipped": False},
            "val": {"use_type": False, "include_skipped": False},
            "test":{"use_type": "include_only_skipped", "skip_skips": True, "whole_patient": True, "std_threshold": 41, "perc_threshold": 0.4}}


    importer.config.model.save_dir = importer.config.model.save_dir.format(m)

    if hasattr(importer.config.model, "encoders"):
        for i in range(len(importer.config.model.encoders)):
            importer.config.model.encoders[i].pretrainedEncoder.dir = importer.config.model.encoders[i].pretrainedEncoder.dir.format(enc_m)


    # importer.config.training_params.test_batch_size = 6


    importer.load_checkpoint()
    deterministic(importer.config.training_params.seed)

    best_model = importer.get_model(return_model="best_model")
    # best_model = importer.get_model(return_model="running_model")

    data_loader = importer.get_dataloaders()

    validator = Validator(model=best_model, data_loader=data_loader, config=importer.config, device="cuda:0")
    # test_results = validator.get_results(set="Validation", print_results=True)

    test_results = validator.get_results(set="Test", print_results=True)

    # validator.save_test_results(checkpoint=importer.checkpoint,
    #                             save_dir=importer.config.model.save_dir, test_results=test_results)


parser = argparse.ArgumentParser(description="My Command Line Program")
parser.add_argument('--config', help="Number of config file")
parser.add_argument('--default_config', help="Number of config file")
parser.add_argument('--fold', help="Fold")
parser.add_argument('--al', help="Alignment loss hyperparameter")
parser.add_argument('--ms', help="Multisupervised loss hyperparameter")
parser.add_argument('--incboth', help="Include patients with both EEG and EOG")
parser.add_argument('--inceeg', help="Include patients with only EEG")
parser.add_argument('--inceog', help="Include patients with only EOG")
parser.add_argument('--noisy', action='store_true')
parser.set_defaults(noisy=False)

args = parser.parse_args()
for var_name in vars(args):
    var_value = getattr(args, var_name)
    if var_value == "None":
        setattr(args, var_name, None)

for i in range(3):
    args.fold = i
    test_each(config_path=args.config, default_config_path=args.default_config, args=args)
