
import numpy as np
import argparse
from colorama import Fore
from posthoc.Helpers.Helper_Importer import Importer
from utils.config import process_config, setup_logger
from collections import defaultdict

def show_each(config_path, default_config_path, args):
    setup_logger()

    importer = Importer(config_name=config_path, default_files=default_config_path, device="cuda:0")

    m = ""

    if "fold" in args and args.fold is not None:
        importer.config.dataset.data_split.fold = int(args.fold)
        importer.config.dataset.fold = int(args.fold)
        m += "fold{}".format(args.fold)
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

    importer.config.model.save_dir = importer.config.model.save_dir.format(m)

    try:
        importer.load_checkpoint()
    except:
        print("We could not load {}".format(importer.config.model.save_dir))
        return

    val_metrics, test_metric = importer.print_progress(multi_fold_results={},
                                                 verbose=False,
                                                 latex_version=False)
    print_metrics(importer, val_metrics, test_metric)
    return val_metrics, test_metric

def print_metrics(importer, val_metrics, test_metric):

    message = Fore.WHITE + "{}  ".format(importer.config.model.save_dir.split("/")[-1])
    # val_metrics = multi_fold_results[0]
    # if "step" in val_metrics:
    #     message += Fore.GREEN + "Step: {}  ".format(val_metrics["step"])
    # if test_flag:
    #     message += Fore.RED + "Test  "

    if "current_epoch" in val_metrics:
        message += Fore.GREEN + "Epoch: {}  ".format(val_metrics["current_epoch"])
    if "steps_no_improve" in val_metrics:
        message += Fore.GREEN + "Steps no improve: {}  ".format(val_metrics["steps_no_improve"])
    if "loss" in val_metrics:
        for i, v in val_metrics["loss"].items(): message += Fore.RED + "{} : {:.6f} ".format(i, v)
    if "acc" in val_metrics:
        for i, v in val_metrics["acc"].items():
            if i == "combined":
                message += Fore.LIGHTBLUE_EX + "Acc_{}: {:.1f} ".format(i, v * 100)
    if "ceu" in val_metrics:
        for i, v in val_metrics["ceu"]["combined"].items(): message += Fore.LIGHTGREEN_EX + "CEU_{}: {:.2f} ".format(i, v)

    # print(test_metric)
    if test_metric and "acc" in test_metric:
        for i, v in test_metric["acc"].items():
            # if i == "combined":
                message += Fore.MAGENTA + "Test_Acc_{}: {:.1f} ".format(i, v * 100)

    # if "top5_acc" in val_metrics:
    #     for i, v in val_metrics["top5_acc"].items(): message += Fore.LIGHTBLUE_EX + "Top5_Acc_{}: {:.2f} ".format(i,
    #                                                                                                               v * 100)
    # if "acc_exzero" in val_metrics:
    #     for i, v in val_metrics["acc_exzero"].items(): message += Fore.LIGHTBLUE_EX + "Acc_ExZ_{}: {:.2f} ".format(i,
    #                                                                                                                v * 100)
    # if "f1" in val_metrics:
    #     for i, v in val_metrics["f1"].items(): message += Fore.LIGHTGREEN_EX + "F1_{}: {:.2f} ".format(i, v * 100)
    # if "k" in val_metrics:
    #     for i, v in val_metrics["k"].items(): message += Fore.LIGHTGREEN_EX + "K_{}: {:.4f} ".format(i, v)
    # if "acc_7" in val_metrics:
    #     for i, v in val_metrics["acc_7"].items(): message += Fore.MAGENTA + "Acc7_{}: {:.4f} ".format(i, v * 100)
    # if "acc_5" in val_metrics:
    #     for i, v in val_metrics["acc_5"].items(): message += Fore.LIGHTMAGENTA_EX + "Acc5_{}: {:.4f} ".format(i, v * 100)
    # if "mae" in val_metrics:
    #     for i, v in val_metrics["mae"].items(): message += Fore.LIGHTBLUE_EX + "MAE_{}: {:.4f} ".format(i, v)
    # if "corr" in val_metrics:
    #     for i, v in val_metrics["corr"].items(): message += Fore.LIGHTWHITE_EX + "Corr_{}: {:.4f} ".format(i, v)
    # if "ece" in test_metric:
    #     for i, v in test_metric["ece"].items(): message += Fore.LIGHTWHITE_EX + "ECE_{}: {:.4f} ".format(i, v)
    print(message + Fore.RESET)

def print_mean(m: dict, val=True):
    agg = {}
    counts = defaultdict(int)  # Keep track of counts for non-dict metrics

    # Step 1: Collect values
    for fold in m:
        for metric in m[fold]:
            if isinstance(m[fold][metric], dict):
                if metric not in agg:
                    agg[metric] = defaultdict(list)
                if metric == "f1_perclass":
                    continue
                for pred in m[fold][metric]:
                    # if pred == "combined":
                        agg[metric][pred].append(m[fold][metric][pred])
            else:
                if metric not in agg:
                    agg[metric] = []
                agg[metric].append(m[fold][metric])
                counts[metric] += 1

    # Step 2: Compute mean and std, and prepare the message
    message = ""
    if val:
        message += Fore.RED + "Val  "
    else:
        message += Fore.GREEN + "Test  "

    for metric in agg:
        if "acc" == metric or "val_acc" == metric or "test_acc" == metric:
            if isinstance(agg[metric], defaultdict):
                for pred in agg[metric]:
                    mean_value = np.mean(agg[metric][pred])
                    std_value = np.std(agg[metric][pred])
                    message += Fore.WHITE + "{}_{}: ".format(metric, pred)
                    message += Fore.LIGHTGREEN_EX + "{:.1f} + {:.1f} ".format(100 * mean_value, 100 * std_value)
            else:
                mean_value = np.mean(agg[metric])
                std_value = np.std(agg[metric])
                message += Fore.WHITE + "{}: ".format(metric)
                message += Fore.LIGHTGREEN_EX + "{:.1f} + {:.1f} ".format(100 * mean_value, 100 * std_value)
        if "acc_7" == metric:
            if isinstance(agg[metric], defaultdict):
                for pred in agg[metric]:
                    mean_value = np.mean(agg[metric][pred])
                    std_value = np.std(agg[metric][pred])
                    message += Fore.GREEN + "{}_{}: ".format(metric, pred)
                    message += Fore.LIGHTGREEN_EX + "{:.4f} + {:.4f} ".format(mean_value, std_value)
        elif "f1" == metric or "val_f1" == metric or "test_f1" == metric:
            if isinstance(agg[metric], defaultdict):
                for pred in agg[metric]:
                    mean_value = np.mean(agg[metric][pred])
                    std_value = np.std(agg[metric][pred])
                    message += Fore.GREEN + "{}_{}: ".format(metric, pred)
                    message += Fore.LIGHTGREEN_EX + "{:.4f} + {:.4f} ".format(mean_value, std_value)
        elif "ece" == metric:
            if isinstance(agg[metric], defaultdict):
                for pred in agg[metric]:
                    mean_value = np.mean(agg[metric][pred])
                    std_value = np.std(agg[metric][pred])
                    message += Fore.GREEN + "{}_{}: ".format(metric, pred)
                    message += Fore.LIGHTGREEN_EX + "{:.4f} + {:.4f} ".format(mean_value, std_value)
        elif "ceu" == metric:
            pred = "combined"
            for each_ceu in agg[metric][pred][0]:
                mean_value = np.concatenate([np.array([i[each_ceu]]) for i in agg[metric][pred]]).mean()
                message += Fore.LIGHTBLUE_EX + "{}_{}: {:.2f} ".format(metric, each_ceu, mean_value)

    print(message)

    # mean_acc_combined = np.mean(agg["acc"]["combined"])
    # std_acc_combined = np.std(agg["acc"]["combined"])
    # return mean_acc_combined, std_acc_combined


parser = argparse.ArgumentParser(description="My Command Line Program")
parser.add_argument('--config', help="Number of config file")
parser.add_argument('--default_config', help="Number of config file")
parser.add_argument('--fold', help="Fold")
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

val = {}
test = {}

if args.fold is None:
    val_metric, test_metric = show_each(config_path=args.config, default_config_path=args.default_config, args=args)
    val[0] = val_metric
    test[0] = test_metric
else:
    for i in range(3):
        args.fold = i
        val_metric, test_metric = show_each(config_path=args.config, default_config_path=args.default_config, args=args)
        val[i] = val_metric
        test[i] = test_metric

print_mean(val, val=True)
print_mean(test, val=False)

