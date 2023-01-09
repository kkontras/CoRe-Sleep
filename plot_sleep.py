import copy
import sys

import einops
import numpy
import numpy as np
import torch

sys.exc_info()
from utils.config import process_config
from datasets.sleepset import *
from graphs.models.attention_models.windowFeature_base import *
from sklearn.metrics import f1_score, cohen_kappa_score, roc_auc_score, confusion_matrix
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from utils.deterministic_pytorch import deterministic
# import umap
from scipy.stats import entropy
import os
from torchdistill.core.forward_hook import ForwardHookManager
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn as sns
import pandas as pd


def get_predictions_time_series(model, views, inits, extract_norm=False):
    """
    This is a function to exploit the fact that time series are not always continuous. We dont want to correlate signals from different patients/recordings just because the batch is not fully dividing the number of recording imgs.
    :param views: List of tensors, data views/modalities
    :param inits: Tensor indicating with value one, when there incontinuities.
    :return: predictions of the model on the batch
    """
    inits_sum = (inits.sum(dim=1) > 1).nonzero(as_tuple=True)[0]
    if len(inits_sum) > 0:
        batch = views[0].shape[0]
        outer = views[0].shape[1]
        batch_idx_checked = torch.ones(batch, dtype=torch.bool)
        pred = torch.zeros(batch * outer, 5).cuda()
        for idx in inits_sum:
            if inits[idx].sum() > 1:
                ones_idx = (inits[idx] > 0).nonzero(as_tuple=True)[0]
                if (ones_idx[0] + 1 == ones_idx[1]):  # and ones_idx[0]!=0 and ones_idx[1]!= len(inits[idx])
                    if ones_idx[0] == 0:
                        pred_split_0 = model([view[idx, ones_idx[0]].unsqueeze(dim=0).unsqueeze(dim=1) for view in views], extract_norm=extract_norm)
                    else:
                        pred_split_0 = model([view[idx, :ones_idx[0] + 1].unsqueeze(dim=0) for view in views], extract_norm=extract_norm)
                    if ones_idx[1] == len(inits[idx]):
                        pred_split_1 = model([view[idx, -1].unsqueeze(dim=0).unsqueeze(dim=1) for view in views], extract_norm=extract_norm)
                    else:
                        pred_split_1 = model([view[idx, ones_idx[1]:].unsqueeze(dim=0) for view in views], extract_norm=extract_norm)

                    pred[idx * outer:(idx + 1) * outer] = torch.cat([pred_split_0, pred_split_1], dim=0)
                    batch_idx_checked[idx] = False
                else:
                    pred[idx * outer:(idx + 1) * outer] = model([view[idx].unsqueeze(dim=0) for view in views], extract_norm=extract_norm)
        pred[batch_idx_checked.repeat_interleave(outer)] = model([view[batch_idx_checked] for view in views], extract_norm=extract_norm)
    else:
        pred = model(views, extract_norm=extract_norm)
    return pred
def perf_measure(y_actual, y_hat):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_hat)):
        if y_actual[i] == y_hat[i] == 1:
            TP += 1
        if y_hat[i] == 1 and y_actual[i] != y_hat[i]:
            FP += 1
        if y_actual[i] == y_hat[i] == 0:
            TN += 1
        if y_hat[i] == 0 and y_actual[i] != y_hat[i]:
            FN += 1

    return (TP, FP, TN, FN)

def get_attention_weights(model, device, batch, seq_l, data_loader, description):

    # device = "cuda:{}".format(config.gpu_device[0])
    # device = "cpu"
    forward_hook_manager = ForwardHookManager(device)
    forward_hook_manager.add_hook(model, 'inner_tf_mod0_l3_RA.inner_tf.layers.0.self_attn_my.scaled_dotproduct_attention', requires_input=True, requires_output=True)
    forward_hook_manager.add_hook(model, 'outer_tf_mod0_l3_RA.outer_tf.layers.0.self_attn_my.scaled_dotproduct_attention', requires_input=True, requires_output=True)
    # forward_hook_manager.add_hook(model, 'layer1.0.bn2', requires_input=True, requires_output=True)
    # forward_hook_manager.add_hook(model, 'fc', requires_input=False, requires_output=True)

    model.eval()
    pbar = tqdm(enumerate(data_loader), desc=description, leave=False)
    for batch_idx, (data, target, init, ids) in pbar:
        views = [data[i].float().to(device) for i in range(len(data))]
        preds = get_predictions_time_series(model, views, init)
        break

    io_dict = forward_hook_manager.pop_io_dict()
    print(io_dict.keys())

    inner_weights = io_dict['inner_tf_mod0_l3_RA.inner_tf.layers.0.self_attn_my.scaled_dotproduct_attention']['output'][1].detach().cpu().numpy()
    outer_weights = io_dict['outer_tf_mod0_l3_RA.outer_tf.layers.0.self_attn_my.scaled_dotproduct_attention']['output'][1].detach().cpu().numpy()
    target = target.detach().cpu().numpy()

    preds = preds.argmax(dim=1)

    print(preds.shape)
    preds = einops.rearrange(preds,"(a b) -> a b ", a=batch, b=seq_l).detach().cpu().numpy()

    t = np.arange(0, 29)
    f = (np.arange(0, 128) / 128) * 50
    this_data = data[0][0, 0, 0, 1:, :].detach().numpy()

    # plt.figure()
    # plt.title("The datapoint we examine")
    # plt.xlabel("Time bins (sec)")
    # plt.ylabel("Freq bins (Hz)")
    # plt.pcolormesh(t, f, this_data, vmin=this_data.min(), vmax=this_data.max(), shading='gouraud')
    # plt.colorbar()
    # plt.show()
    #

    token_of_interest = 0
    idx = 0

    print(inner_weights.shape)
    print(outer_weights.shape)

    print(target)

    # c = {'Token of Interest': 'lightgreen', 'Wrongly classified': 'red', "Correct classified": "lightblue"}
    c = { 'Wrongly classified': 'red', "Correct classified": "lightblue"}
    total_heads = 8
    inner_weights = einops.rearrange(inner_weights,"(outer inner h) d m-> outer inner h d m", outer=batch, inner=seq_l, h=total_heads)
    outer_weights = einops.rearrange(outer_weights,"(outer h) d m-> outer h d m", outer=batch, h=total_heads)
    print(preds[idx]==target[idx])
    for token_of_interest in [0]:

        plt.figure(figsize=(20, 30))
        x = np.linspace(0, 29, 30).astype(int)
        colors = ["lightblue" for i in x]
        # colors[0] = "lightgreen"
        for h in range(total_heads):
            if total_heads>1:
                plt.subplot(int("{}{}{}".format(int(total_heads/2),2,h+1)))
            plt.ylabel("Attention Weight")
            plt.xlabel("Inner Seq Steps")
            plt.xticks(x)
            plt.yticks(x)
            plt.title("CLS Token head {}".format(h))
            plt.imshow(inner_weights[idx][token_of_interest][h], cmap='hot', interpolation='nearest')
        im_ratio = inner_weights[idx][token_of_interest][h].shape[0] / inner_weights[idx][token_of_interest][h].shape[1]
        plt.colorbar( fraction=0.046 * im_ratio, pad=0.04)

            # plt.bar(x, inner_weights[idx][token_of_interest][h][0], color=colors)
        plt.show()
        plt.figure(figsize=(25, 20))

        for h in range(total_heads):
            if total_heads>1:
                plt.subplot(int("{}{}{}".format(int(total_heads/2),2,h+1)))
            plt.ylabel("Attention Weight")
            plt.xlabel("Outer Seq Steps")
            plt.title("Outer single modality EEG head {}".format(h))
            x = np.linspace(0,len(outer_weights[idx][h][token_of_interest])-1,len(outer_weights[idx][h][token_of_interest])).astype(int)
            l = {0:"W",1:"N1",2:"N2",3:"N3",4:"R"}
            # labels = [l[preds[idx][i]]+"/"+l[target[idx][i]] for i in range(len(target[idx]))]
            labels = [l[target[idx][i]] for i in range(len(target[idx]))]
            colors = preds[idx]== target[idx]
            colors = ["lightblue" if i else "red" for i in colors]
            # colors[token_of_interest] = "lightgreen"
            plt.xticks(x,labels, rotation=0)
            plt.yticks(x,labels, rotation=0)
            for ticklabel, tickcolor in zip(plt.gca().get_xticklabels(), colors):
                ticklabel.set_color(tickcolor)
            for ticklabel, tickcolor in zip(plt.gca().get_yticklabels(), colors):
                ticklabel.set_color(tickcolor)

            plt.xlabel("Prediction / Correct Label")
            handles = [plt.Rectangle((0, 0), 1, 1, color=c[l]) for l in list(c.keys())]
            plt.legend(handles, list(c.keys()))
            plt.imshow(outer_weights[idx][h], cmap='hot', interpolation='nearest')
            im_ratio = outer_weights[idx][h].shape[0] / outer_weights[idx][h].shape[1]
        plt.colorbar( fraction=0.046 * im_ratio, pad=0.04)
            # plt.bar(x, outer_weights[idx][h][token_of_interest], color=colors, width=0.7)

        plt.show()

    # figure_list = []
    # fig, ax = plt.subplots()
    # x = np.linspace(0,29,30).astype(int)
    # for i in range(1,30):
    #     ax.set_ylabel("Attention Weight")
    #     ax.set_xlabel("Inner Seq Steps")
    #     ax.set_title("Inner Tokens in an untrained model")
    #     figure_list.append(ax.bar(x, inner_weights[0][i].detach().numpy(), color='lightblue'))
    # ani = animation.ArtistAnimation(fig, figure_list, blit=False)
    # writer = animation.FFMpegWriter(fps=3, extra_args=['-vcodec', 'libx264'])
    # ani.save('inner.mp4', writer=writer)
    # ani.save('inner.gif', 'imagemagick')


    # figure_list = []
    # fig, ax = plt.subplots()
    # x = np.linspace(0,20,21).astype(int)
    # for i in range(21):
    #     ax.set_ylabel("Attention Weight")
    #     ax.set_xlabel("Outer Seq Steps")
    #     ax.set_title("Tokens in an untrained model")
    #     figure_list.append(ax.bar(x, outer_weights[0][i].detach().numpy(), color='lightblue'))
    # ani = animation.ArtistAnimation(fig, figure_list, blit=False)
    # writer = animation.FFMpegWriter(fps=3, extra_args=['-vcodec', 'libx264'])
    # ani.save('outer.mp4', writer=writer)
    # ani.save('outer.gif', 'imagemagick')
    preds = preds[idx]
    target = target[idx]

    non_matches = (preds != target).astype(int)
    non_matches_idx = non_matches.nonzero()[0]
    print("Non matching indices are:")
    hours = len(target)

    non_matches_idx = non_matches_idx

    pred_plus = copy.deepcopy(preds)
    pred_plus[pred_plus == 4] = 5
    pred_plus[pred_plus == 3] = 4
    pred_plus[pred_plus == 2] = 3
    pred_plus[pred_plus == 5] = 2

    target_plus = copy.deepcopy(target)
    target_plus[target_plus == 4] = 5
    target_plus[target_plus == 3] = 4
    target_plus[target_plus == 2] = 3
    target_plus[target_plus == 5] = 2

    # target = target + 0.02
    target_plus = target_plus + 0.02

    plt.figure()
    plt.plot(pred_plus,label="Prediction")
    plt.plot(target_plus,label="True Label")
    plt.scatter(non_matches_idx, pred_plus[non_matches_idx], marker='*', edgecolors="r", label="Mistakes")
    # plt.plot(non_matches_idx,"*")
    plt.yticks([0, 1, 2, 3, 4], labels=["Wake", "N1", "REM", "N2", "N3"])
    plt.xticks([i * 120 for i in range((hours // 120) + 1)],
               labels=["{}".format(i) for i in range((hours // 120) + 1)])
    plt.legend()
    plt.ylabel("Labels")
    plt.xlabel("Hours")
    plt.show()

    return 0
def count_parameters(model):
    count = 0
    counts = {}
    for name, parameter in model.named_parameters():
        # print(name, end="  ")
        if name.split(".")[1] not in counts.keys():
            counts[name.split(".")[1]] = parameter.numel()
        else:
            counts[name.split(".")[1]] += parameter.numel()
        # print(parameter.numel())
        count += parameter.numel()

    print("Total of parameter: {}".format(count))
    for key in counts.keys():
        print("{}  {}".format(key, counts[key]))
def get_attention_weights_merged(model, batch, seq_l, device, data_loader, description):

    # device = "cuda:{}".format(config.gpu_device[0])
    # device = "cpu"
    forward_hook_manager = ForwardHookManager(device)
    forward_hook_manager.add_hook(model, 'inner_tf_mod0_l3.inner_tf.layers.0.self_attn_my.scaled_dotproduct_attention', requires_input=True, requires_output=True)
    forward_hook_manager.add_hook(model, 'outer_tf_mod0_l3.outer_tf.layers.0.self_attn_my.scaled_dotproduct_attention', requires_input=True, requires_output=True)
    # forward_hook_manager.add_hook(model, 'layer1.0.bn2', requires_input=True, requires_output=True)
    # forward_hook_manager.add_hook(model, 'fc', requires_input=False, requires_output=True)

    model.eval()
    pbar = tqdm(enumerate(data_loader), desc=description, leave=False)
    for batch_idx, (data, target, init, ids) in pbar:
        views = [data[i].float().to(device) for i in range(len(data))]
        preds = get_predictions_time_series(model, views, init)
        break

    io_dict = forward_hook_manager.pop_io_dict()
    print(target)

    inner_weights = io_dict['inner_tf_mod0_l3.inner_tf.layers.0.self_attn_my.scaled_dotproduct_attention']['output'][1].detach().cpu().numpy()
    outer_weights = io_dict['outer_tf_mod0_l3.outer_tf.layers.0.self_attn_my.scaled_dotproduct_attention']['output'][1].detach().cpu().numpy()
    target = target.detach().cpu().numpy()

    preds = preds.argmax(dim=1)

    preds = einops.rearrange(preds,"(a b) -> a b ", a=batch, b=seq_l).detach().cpu().numpy()

    t = np.arange(0, 29)
    f = (np.arange(0, 128) / 128) * 50
    this_data = data[0][0, 0, 0, 1:, :].detach().numpy()

    # plt.figure()
    # plt.title("The datapoint we examine")
    # plt.xlabel("Time bins (sec)")
    # plt.ylabel("Freq bins (Hz)")
    # plt.pcolormesh(t, f, this_data, vmin=this_data.min(), vmax=this_data.max(), shading='gouraud')
    # plt.colorbar()
    # plt.show()

    # plt.figure()
    # plt.subplot(211)
    # plt.ylabel("Attention Weight EEG")
    # plt.title("CLS Token")
    # x = np.linspace(0,59,60).astype(int)
    # colors = ["lightblue" for i in range(len(x))]
    # colors[0] = "lightgreen"
    # # print(inner_weights[0][0].shape)
    # # print(x.shape)
    # plt.bar(x, inner_weights[0][0], color=colors)
    #
    # plt.subplot(212)
    # plt.ylabel("Attention Weight EOG")
    # plt.xlabel("Inner Seq Steps")
    # colors = ["lightblue" for i in range(len(x))]
    # colors[-1] = "lightgreen"
    # plt.bar(x, inner_weights[0][-1], color=colors)
    # plt.show()
    token_of_interest = 0
    idx = -1

    c = {'Token of Interest': 'lightgreen', 'Wrongly classified': 'red', "Correct classified": "lightblue"}
    l = {0: "W", 1: "N1", 2: "N2", 3: "N3", 4: "R"}

    print(ids[idx])
    token_of_interests = [0]
    for token_of_interest in token_of_interests:
        plt.figure(figsize=(25, 20))
        plt.subplot(211)
        plt.ylabel("Attention Weight")
        plt.xlabel("Outer Seq Steps")
        plt.title("EEG Tokens")
        x = np.linspace(0,int(len(outer_weights[idx][token_of_interest])/2)-1, int(len(outer_weights[idx][token_of_interest])/2) ).astype(int)

        # labels = [l[preds[idx][i]]+"/"+l[target[idx][i]] for i in range(len(target[idx]))]
        labels = [l[target[idx][i]] for i in range(len(target[idx]))]
        print(len(labels))
        print(x.shape)
        colors = preds[idx]== target[idx]
        colors = ["lightblue" if i else "red" for i in colors]
        colors_eeg = copy.deepcopy(colors)
        if token_of_interest < len(outer_weights[idx][token_of_interest])/2:
            colors_eeg[token_of_interest] = "lightgreen"
        else:
            colors[token_of_interest - int(len(colors_eeg))] = "lightgreen"
        plt.xticks(x,labels, rotation=0)
        plt.ylim([0,0.1])
        plt.xlabel("Correct Label")
        handles = [plt.Rectangle((0, 0), 1, 1, color=c[l]) for l in list(c.keys())]
        plt.legend(handles, list(c.keys()))
        plt.bar(x, outer_weights[idx][token_of_interest][:int(len(outer_weights[idx][token_of_interest])/2)], color=colors_eeg, width=0.7)

        plt.subplot(212)
        plt.ylabel("Attention Weight")
        plt.xlabel("Outer Seq Steps")
        plt.title("EOG Tokens")
        # labels = [l[preds[idx][i]]+"/"+l[target[idx][i]] for i in range(len(target[idx]))]
        plt.ylim([0,0.1])
        plt.xticks(x, labels, rotation=0)
        plt.xlabel("Correct Label")
        plt.bar(x, outer_weights[idx][token_of_interest][int(len(outer_weights[idx][token_of_interest]) / 2):],
                color=colors, width=0.7)

        plt.show()

    c = { 'Wrongly classified': 'red', "Correct classified": "lightblue"}
    total_heads = 1
    inner_weights = einops.rearrange(inner_weights,"(outer inner h) d m-> outer inner h d m", outer=batch, inner=seq_l*2, h=total_heads)
    outer_weights = einops.rearrange(outer_weights,"(outer h) d m-> outer h d m", outer=batch, h=total_heads)
    print(preds[idx]==target[idx])
    for token_of_interest in [5]:

        plt.figure(figsize=(20, 30))
        x = np.linspace(0, inner_weights.shape[1]-1,  inner_weights.shape[1]).astype(int)
        colors = ["lightblue" for i in x]
        # colors[0] = "lightgreen"
        for h in range(total_heads):
            if total_heads>1:
                plt.subplot(int("{}{}{}".format(int(total_heads/2),2,h+1)))
            plt.ylabel("Attention Weight")
            plt.xlabel("Inner Seq Steps")
            plt.xticks(x)
            plt.yticks(x)
            plt.title("CLS Token head {}".format(h))
            plt.imshow(inner_weights[idx][token_of_interest][h], cmap='hot', interpolation='nearest')
        im_ratio = inner_weights[idx][token_of_interest][h].shape[0] / inner_weights[idx][token_of_interest][h].shape[1]
        plt.colorbar( fraction=0.046 * im_ratio, pad=0.04)

            # plt.bar(x, inner_weights[idx][token_of_interest][h][0], color=colors)
        plt.show()
        plt.figure(figsize=(25, 20))

        for h in range(total_heads):
            if total_heads>1:
                plt.subplot(int("{}{}{}".format(int(total_heads/2),2,h+1)))
            plt.ylabel("Attention Weight")
            plt.xlabel("Outer Seq Steps")
            plt.title("Outer single modality EEG head {}".format(h))
            x = np.linspace(0,len(outer_weights[idx][h][token_of_interest])-1,len(outer_weights[idx][h][token_of_interest])).astype(int)
            l = {0:"W",1:"N1",2:"N2",3:"N3",4:"R"}
            # labels = [l[preds[idx][i]]+"/"+l[target[idx][i]] for i in range(len(target[idx]))]
            labels = ["EEG "+ l[target[idx][i%21]] if i<21 else "EOG "+ l[target[idx][i%21]] for i in range(2*len(target[idx]))]
            colors = preds[idx]== target[idx]
            colors_eeg = ["lightblue" if i else "red" for i in colors]
            colors_eog = ["lightblue" if i else "red" for i in colors]
            colors = colors_eeg + colors_eog
            colors[token_of_interest] = "lightgreen"
            plt.xticks(x,labels, rotation=90)
            plt.yticks(x,labels, rotation=0)
            for ticklabel, tickcolor in zip(plt.gca().get_xticklabels(), colors):
                ticklabel.set_color(tickcolor)
            for ticklabel, tickcolor in zip(plt.gca().get_yticklabels(), colors):
                ticklabel.set_color(tickcolor)

            plt.xlabel("Label")
            handles = [plt.Rectangle((0, 0), 1, 1, color=c[l]) for l in list(c.keys())]
            plt.legend(handles, list(c.keys()))
            plt.imshow(outer_weights[idx][h], cmap='hot', interpolation='nearest')
            im_ratio = outer_weights[idx][h].shape[0] / outer_weights[idx][h].shape[1]
        plt.colorbar( fraction=0.046 * im_ratio, pad=0.04)
            # plt.bar(x, outer_weights[idx][h][token_of_interest], color=colors, width=0.7)

        plt.show()
    # figure_list = []
    # fig, ax = plt.subplots()
    # x = np.linspace(0,29,30).astype(int)
    # for i in range(1,30):
    #     ax.set_ylabel("Attention Weight")
    #     ax.set_xlabel("Inner Seq Steps")
    #     ax.set_title("Inner Tokens in an untrained model")
    #     figure_list.append(ax.bar(x, inner_weights[0][i].detach().numpy(), color='lightblue'))
    # ani = animation.ArtistAnimation(fig, figure_list, blit=False)
    # writer = animation.FFMpegWriter(fps=3, extra_args=['-vcodec', 'libx264'])
    # ani.save('inner.mp4', writer=writer)
    # ani.save('inner.gif', 'imagemagick')


    # figure_list = []
    # fig, ax = plt.subplots()
    # x = np.linspace(0,20,21).astype(int)
    # for i in range(21):
    #     ax.set_ylabel("Attention Weight")
    #     ax.set_xlabel("Outer Seq Steps")
    #     ax.set_title("Tokens in an untrained model")
    #     figure_list.append(ax.bar(x, outer_weights[0][i].detach().numpy(), color='lightblue'))
    # ani = animation.ArtistAnimation(fig, figure_list, blit=False)
    # writer = animation.FFMpegWriter(fps=3, extra_args=['-vcodec', 'libx264'])
    # ani.save('outer.mp4', writer=writer)
    # ani.save('outer.gif', 'imagemagick')
    preds = preds[idx]
    target = target[idx]
    print(target)
    non_matches = (preds != target).astype(int)
    non_matches_idx = non_matches.nonzero()[0]
    print("Non matching indices are:")
    hours = len(target)

    non_matches_idx = non_matches_idx

    pred_plus = copy.deepcopy(preds)
    pred_plus[pred_plus == 4] = 5
    pred_plus[pred_plus == 3] = 4
    pred_plus[pred_plus == 2] = 3
    pred_plus[pred_plus == 5] = 2

    target_plus = copy.deepcopy(target)
    target_plus[target_plus == 4] = 5
    target_plus[target_plus == 3] = 4
    target_plus[target_plus == 2] = 3
    target_plus[target_plus == 5] = 2

    # target = target + 0.02
    target_plus = target_plus + 0.02

    plt.figure()
    plt.plot(pred_plus,label="Prediction")
    plt.plot(target_plus,label="True Label")
    plt.scatter(non_matches_idx, pred_plus[non_matches_idx], marker='*', edgecolors="r", label="Mistakes")
    # plt.plot(non_matches_idx,"*")
    plt.yticks([0, 1, 2, 3, 4], labels=["Wake", "N1", "REM", "N2", "N3"])
    plt.xticks([i * 120 for i in range((hours // 120) + 1)],
               labels=["{}".format(i) for i in range((hours // 120) + 1)])
    plt.legend()
    plt.ylabel("Labels")
    plt.xlabel("Hours")
    plt.show()

    return 0
def get_attention_weights_concat(model, device, data_loader, description):

    # device = "cuda:{}".format(config.gpu_device[0])
    # device = "cpu"
    forward_hook_manager = ForwardHookManager(device)
    forward_hook_manager.add_hook(model, 'inner_tf_mod0_l3.inner_tf.layers.0.self_attn_my.scaled_dotproduct_attention', requires_input=True, requires_output=True)
    forward_hook_manager.add_hook(model, 'outer_tf_mod0_l3.outer_tf.layers.0.self_attn_my.scaled_dotproduct_attention', requires_input=True, requires_output=True)
    # forward_hook_manager.add_hook(model, 'layer1.0.bn2', requires_input=True, requires_output=True)
    # forward_hook_manager.add_hook(model, 'fc', requires_input=False, requires_output=True)

    model.eval()
    pbar = tqdm(enumerate(data_loader), desc=description, leave=False)
    for batch_idx, (data, target, init, ids) in pbar:
        views = [data[i].float().to(device) for i in range(len(data))]
        preds = get_predictions_time_series(model, views, init)
        break

    io_dict = forward_hook_manager.pop_io_dict()
    print(io_dict.keys())

    inner_weights = io_dict['inner_tf_mod0_l3.inner_tf.layers.0.self_attn_my.scaled_dotproduct_attention']['output'][1].detach().cpu().numpy()
    outer_weights = io_dict['outer_tf_mod0_l3.outer_tf.layers.0.self_attn_my.scaled_dotproduct_attention']['output'][1].detach().cpu().numpy()
    target = target.detach().cpu().numpy()

    preds = preds.argmax(dim=1)

    preds = einops.rearrange(preds,"(a b) -> a b ", a=32, b=41).detach().cpu().numpy()

    t = np.arange(0, 29)
    f = (np.arange(0, 128) / 128) * 50
    this_data = data[0][0, 0, 0, 1:, :].detach().numpy()

    # plt.figure()
    # plt.title("The datapoint we examine")
    # plt.xlabel("Time bins (sec)")
    # plt.ylabel("Freq bins (Hz)")
    # plt.pcolormesh(t, f, this_data, vmin=this_data.min(), vmax=this_data.max(), shading='gouraud')
    # plt.colorbar()
    # plt.show()

    print(target.shape)
    print(outer_weights.shape)

    # plt.figure()
    # plt.subplot(211)
    # plt.ylabel("Attention Weight EEG")
    # plt.title("CLS Token")
    # x = np.linspace(0,59,60).astype(int)
    # colors = ["lightblue" for i in range(len(x))]
    # colors[0] = "lightgreen"
    # plt.bar(x, inner_weights[0][0], color=colors)
    #
    # plt.subplot(212)
    # plt.ylabel("Attention Weight EOG")
    # plt.xlabel("Inner Seq Steps")
    # colors = ["lightblue" for i in range(len(x))]
    # colors[-1] = "lightgreen"
    # plt.bar(x, inner_weights[0][-1], color=colors)
    # plt.show()
    token_of_interest = 0
    idx = 15

    c = {'Token of Interest': 'lightgreen', 'Wrongly classified': 'red', "Correct classified": "lightblue"}
    l = {0: "W", 1: "N1", 2: "N2", 3: "N3", 4: "R"}

    print(ids[idx])
    token_of_interests = [0 ]
    for token_of_interest in token_of_interests:
        plt.figure(figsize=(20, 10))
        plt.ylabel("Attention Weight")
        plt.xlabel("Outer Seq Steps")
        plt.title("EEG-EOG Concat Tokens")
        x = np.linspace(0,int(len(outer_weights[idx][token_of_interest]))-1, int(len(outer_weights[idx][token_of_interest])) ).astype(int)

        # labels = [l[preds[idx][i]]+"/"+l[target[idx][i]] for i in range(len(target[idx]))]
        labels = [l[target[idx][i]] for i in range(len(target[idx]))]
        print(len(labels))
        print(x.shape)
        colors = preds[idx]== target[idx]
        colors = ["lightblue" if i else "red" for i in colors]
        colors_eeg = copy.deepcopy(colors)
        if token_of_interest < len(outer_weights[idx][token_of_interest]):
            colors_eeg[token_of_interest] = "lightgreen"
        else:
            colors[token_of_interest - int(len(colors_eeg))] = "lightgreen"
        plt.xticks(x,labels, rotation=0)
        plt.ylim([0,0.1])
        plt.xlabel("Correct Label")
        handles = [plt.Rectangle((0, 0), 1, 1, color=c[l]) for l in list(c.keys())]
        plt.legend(handles, list(c.keys()))

        plt.bar(x, outer_weights[idx][token_of_interest], color=colors_eeg, width=0.7)
        plt.show()

    # figure_list = []
    # fig, ax = plt.subplots()
    # x = np.linspace(0,29,30).astype(int)
    # for i in range(1,30):
    #     ax.set_ylabel("Attention Weight")
    #     ax.set_xlabel("Inner Seq Steps")
    #     ax.set_title("Inner Tokens in an untrained model")
    #     figure_list.append(ax.bar(x, inner_weights[0][i].detach().numpy(), color='lightblue'))
    # ani = animation.ArtistAnimation(fig, figure_list, blit=False)
    # writer = animation.FFMpegWriter(fps=3, extra_args=['-vcodec', 'libx264'])
    # ani.save('inner.mp4', writer=writer)
    # ani.save('inner.gif', 'imagemagick')


    # figure_list = []
    # fig, ax = plt.subplots()
    # x = np.linspace(0,20,21).astype(int)
    # for i in range(21):
    #     ax.set_ylabel("Attention Weight")
    #     ax.set_xlabel("Outer Seq Steps")
    #     ax.set_title("Tokens in an untrained model")
    #     figure_list.append(ax.bar(x, outer_weights[0][i].detach().numpy(), color='lightblue'))
    # ani = animation.ArtistAnimation(fig, figure_list, blit=False)
    # writer = animation.FFMpegWriter(fps=3, extra_args=['-vcodec', 'libx264'])
    # ani.save('outer.mp4', writer=writer)
    # ani.save('outer.gif', 'imagemagick')
    preds = preds[idx]
    target = target[idx]

    non_matches = (preds != target).astype(int)
    non_matches_idx = non_matches.nonzero()[0]
    print("Non matching indices are:")
    hours = len(target)

    non_matches_idx = non_matches_idx

    pred_plus = copy.deepcopy(preds)
    pred_plus[pred_plus == 4] = 5
    pred_plus[pred_plus == 3] = 4
    pred_plus[pred_plus == 2] = 3
    pred_plus[pred_plus == 5] = 2

    target_plus = copy.deepcopy(target)
    target_plus[target_plus == 4] = 5
    target_plus[target_plus == 3] = 4
    target_plus[target_plus == 2] = 3
    target_plus[target_plus == 5] = 2

    # target = target + 0.02
    target_plus = target_plus + 0.02

    plt.figure()
    plt.plot(pred_plus,label="Prediction")
    plt.plot(target_plus,label="True Label")
    plt.scatter(non_matches_idx, pred_plus[non_matches_idx], marker='*', edgecolors="r", label="Mistakes")
    # plt.plot(non_matches_idx,"*")
    plt.yticks([0, 1, 2, 3, 4], labels=["Wake", "N1", "REM", "N2", "N3"])
    plt.xticks([i * 120 for i in range((hours // 120) + 1)],
               labels=["{}".format(i) for i in range((hours // 120) + 1)])
    plt.legend()
    plt.ylabel("Labels")
    plt.xlabel("Hours")
    plt.show()

    return 0
def get_attention_weights_bottleneck(model, device, data_loader, description, context_points=1):

    # device = "cuda:{}".format(config.gpu_device[0])
    # device = "cpu"
    forward_hook_manager = ForwardHookManager(device)
    forward_hook_manager.add_hook(model, 'inner_tf_mod0_l3.inner_tf.layers.0.self_attn_my.scaled_dotproduct_attention', requires_input=True, requires_output=True)
    forward_hook_manager.add_hook(model, 'inner_tf_mod1_l3.inner_tf.layers.0.self_attn_my.scaled_dotproduct_attention', requires_input=True, requires_output=True)
    forward_hook_manager.add_hook(model, 'outer_tf_mod0_l3.outer_tf.layers.0.self_attn_my.scaled_dotproduct_attention', requires_input=True, requires_output=True)
    forward_hook_manager.add_hook(model, 'outer_tf_mod1_l3.outer_tf.layers.0.self_attn_my.scaled_dotproduct_attention', requires_input=True, requires_output=True)
    # forward_hook_manager.add_hook(model, 'layer1.0.bn2', requires_input=True, requires_output=True)
    # forward_hook_manager.add_hook(model, 'fc', requires_input=False, requires_output=True)

    model.eval()
    pbar = tqdm(enumerate(data_loader), desc=description, leave=False)
    for batch_idx, (data, target, init, ids) in pbar:
        views = [data[i].float().to(device) for i in range(len(data))]
        preds = get_predictions_time_series(model, views, init)
        break

    io_dict = forward_hook_manager.pop_io_dict()
    print(io_dict.keys())

    inner_weights_eeg = io_dict['inner_tf_mod0_l3.inner_tf.layers.0.self_attn_my.scaled_dotproduct_attention']['output'][1].detach().cpu().numpy()
    inner_weights_eog = io_dict['inner_tf_mod1_l3.inner_tf.layers.0.self_attn_my.scaled_dotproduct_attention']['output'][1].detach().cpu().numpy()
    outer_weights_eeg = io_dict['outer_tf_mod0_l3.outer_tf.layers.0.self_attn_my.scaled_dotproduct_attention']['output'][1].detach().cpu().numpy()
    outer_weights_eog = io_dict['outer_tf_mod1_l3.outer_tf.layers.0.self_attn_my.scaled_dotproduct_attention']['output'][1].detach().cpu().numpy()

    target = target.detach().cpu().numpy()

    preds = preds.argmax(dim=1)
    inner_shape = int(len(preds)/32)
    preds = einops.rearrange(preds,"(a b) -> a b ", a=32, b=inner_shape).detach().cpu().numpy()

    t = np.arange(0, 29)
    f = (np.arange(0, 128) / 128) * 50
    this_data = data[0][0, 0, 0, 1:, :].detach().numpy()

    # plt.figure()
    # plt.title("The datapoint we examine")
    # plt.xlabel("Time bins (sec)")
    # plt.ylabel("Freq bins (Hz)")
    # plt.pcolormesh(t, f, this_data, vmin=this_data.min(), vmax=this_data.max(), shading='gouraud')
    # plt.colorbar()
    # plt.show()
    #
    colors = ["lightblue" if i else "red" for i in range(len(inner_weights_eeg[0][0]))]
    colors[0] = "lightgreen"
    for i in range(len(colors)-context_points, len(colors)):
        colors[i] = "orange"
    c = {'Token of Interest': 'lightgreen', "Intermediate Steps": "lightblue", "Context":"orange"}

    plt.figure()
    plt.subplot(211)
    plt.ylabel("Attention Weight")
    plt.title("CLS Token in EEG")
    x = np.linspace(0,len(inner_weights_eeg[0][0])-1,len(inner_weights_eeg[0][0])).astype(int)
    plt.xticks([],[])
    plt.ylim([0, 0.13])
    handles = [plt.Rectangle((0, 0), 1, 1, color=c[l]) for l in list(c.keys())]
    plt.legend(handles, list(c.keys()))
    plt.bar(x, inner_weights_eeg[0][0], color=colors)
    plt.subplot(212)
    plt.ylim([0, 0.13])
    plt.ylabel("Attention Weight")
    plt.xlabel("Inner Seq Steps")
    plt.title("EOG")
    plt.bar(x, inner_weights_eog[0][0], color=colors)
    plt.show()
    token_of_interest = 0
    idx = 15


    c = {'Token of Interest': 'lightgreen', 'Wrongly classified': 'red', "Correct classified": "lightblue", "Context": "orange"}
    l = {0: "W", 1: "N1", 2: "N2", 3: "N3", 4: "R"}

    print(target)

    print(outer_weights_eeg.shape)
    print(preds[idx]==target[idx])
    token_of_interests = [0]
    for token_of_interest in token_of_interests:
        plt.figure(figsize=(25, 20))
        plt.subplot(211)
        plt.ylabel("Attention Weight")
        plt.xlabel("Outer Seq Steps")
        plt.title("EEG Tokens")
        x = np.linspace(0,len(outer_weights_eeg[idx][token_of_interest])-1, len(outer_weights_eeg[idx][token_of_interest]) ).astype(int)

        # labels = [l[preds[idx][i]]+"/"+l[target[idx][i]] for i in range(len(target[idx]))]
        labels = [l[target[idx][i]] for i in range(len(target[idx]))]

        colors = preds[idx]== target[idx]
        colors = ["lightblue" if i else "red" for i in colors]
        colors[token_of_interest] = "lightgreen"
        for i in range(context_points):
            labels.append("C")
            colors.append("orange")
        plt.xticks(x, labels, rotation=0)
        plt.ylim([0,0.1])
        plt.xlabel("Correct Label")
        handles = [plt.Rectangle((0, 0), 1, 1, color=c[l]) for l in list(c.keys())]
        plt.legend(handles, list(c.keys()))
        print(outer_weights_eeg[idx][token_of_interest][:len(outer_weights_eeg[idx][token_of_interest])])

        plt.bar(x, outer_weights_eeg[idx][token_of_interest][:len(outer_weights_eeg[idx][token_of_interest])], color=colors, width=0.7)

        plt.subplot(212)
        plt.ylabel("Attention Weight")
        plt.xlabel("Outer Seq Steps")
        plt.title("EOG Tokens")
        # labels = [l[preds[idx][i]]+"/"+l[target[idx][i]] for i in range(len(target[idx]))]
        plt.ylim([0,0.1])
        plt.xticks(x, labels, rotation=0)
        plt.xlabel("Correct Label")
        print(outer_weights_eog[idx][token_of_interest][:len(outer_weights_eog[idx][token_of_interest])])
        plt.bar(x, outer_weights_eog[idx][token_of_interest][:len(outer_weights_eog[idx][token_of_interest])],
                color=colors, width=0.7)

        plt.show()

    # figure_list = []
    # fig, ax = plt.subplots()
    # x = np.linspace(0,29,30).astype(int)
    # for i in range(1,30):
    #     ax.set_ylabel("Attention Weight")
    #     ax.set_xlabel("Inner Seq Steps")
    #     ax.set_title("Inner Tokens in an untrained model")
    #     figure_list.append(ax.bar(x, inner_weights[0][i].detach().numpy(), color='lightblue'))
    # ani = animation.ArtistAnimation(fig, figure_list, blit=False)
    # writer = animation.FFMpegWriter(fps=3, extra_args=['-vcodec', 'libx264'])
    # ani.save('inner.mp4', writer=writer)
    # ani.save('inner.gif', 'imagemagick')


    # figure_list = []
    # fig, ax = plt.subplots()
    # x = np.linspace(0,20,21).astype(int)
    # for i in range(21):
    #     ax.set_ylabel("Attention Weight")
    #     ax.set_xlabel("Outer Seq Steps")
    #     ax.set_title("Tokens in an untrained model")
    #     figure_list.append(ax.bar(x, outer_weights[0][i].detach().numpy(), color='lightblue'))
    # ani = animation.ArtistAnimation(fig, figure_list, blit=False)
    # writer = animation.FFMpegWriter(fps=3, extra_args=['-vcodec', 'libx264'])
    # ani.save('outer.mp4', writer=writer)
    # ani.save('outer.gif', 'imagemagick')
    preds = preds[idx]
    target = target[idx]

    non_matches = (preds != target).astype(int)
    non_matches_idx = non_matches.nonzero()[0]
    print("Non matching indices are:")
    hours = len(target)

    non_matches_idx = non_matches_idx

    pred_plus = copy.deepcopy(preds)
    pred_plus[pred_plus == 4] = 5
    pred_plus[pred_plus == 3] = 4
    pred_plus[pred_plus == 2] = 3
    pred_plus[pred_plus == 5] = 2

    target_plus = copy.deepcopy(target)
    target_plus[target_plus == 4] = 5
    target_plus[target_plus == 3] = 4
    target_plus[target_plus == 2] = 3
    target_plus[target_plus == 5] = 2

    # target = target + 0.02
    target_plus = target_plus + 0.02

    plt.figure()
    plt.plot(pred_plus,label="Prediction")
    plt.plot(target_plus,label="True Label")
    plt.scatter(non_matches_idx, pred_plus[non_matches_idx], marker='*', edgecolors="r", label="Mistakes")
    # plt.plot(non_matches_idx,"*")
    plt.yticks([0, 1, 2, 3, 4], labels=["Wake", "N1", "REM", "N2", "N3"])
    plt.xticks([i * 120 for i in range((hours // 120) + 1)],
               labels=["{}".format(i) for i in range((hours // 120) + 1)])
    plt.legend()
    plt.ylabel("Labels")
    plt.xlabel("Hours")
    plt.show()

    return 0
def get_attention_weights_late(model, device, data_loader, description):

    # device = "cuda:{}".format(config.gpu_device[0])
    # device = "cpu"
    forward_hook_manager = ForwardHookManager(device)
    forward_hook_manager.add_hook(model, 'inner_tf_mod0_l3.inner_tf.layers.0.self_attn_my.scaled_dotproduct_attention', requires_input=True, requires_output=True)
    forward_hook_manager.add_hook(model, 'inner_tf_mod1_l3.inner_tf.layers.0.self_attn_my.scaled_dotproduct_attention', requires_input=True, requires_output=True)
    forward_hook_manager.add_hook(model, 'outer_tf_mod0_l3.outer_tf.layers.0.self_attn_my.scaled_dotproduct_attention', requires_input=True, requires_output=True)
    forward_hook_manager.add_hook(model, 'outer_tf_mod1_l3.outer_tf.layers.0.self_attn_my.scaled_dotproduct_attention', requires_input=True, requires_output=True)
    # forward_hook_manager.add_hook(model, 'layer1.0.bn2', requires_input=True, requires_output=True)
    # forward_hook_manager.add_hook(model, 'fc', requires_input=False, requires_output=True)

    model.eval()
    pbar = tqdm(enumerate(data_loader), desc=description, leave=False)
    for batch_idx, (data, target, init, ids) in pbar:
        views = [data[i].float().to(device) for i in range(len(data))]
        preds = get_predictions_time_series(model, views, init)
        break

    io_dict = forward_hook_manager.pop_io_dict()
    print(io_dict.keys())

    inner_weights_eeg = io_dict['inner_tf_mod0_l3.inner_tf.layers.0.self_attn_my.scaled_dotproduct_attention']['output'][1].detach().cpu().numpy()
    inner_weights_eog = io_dict['inner_tf_mod1_l3.inner_tf.layers.0.self_attn_my.scaled_dotproduct_attention']['output'][1].detach().cpu().numpy()
    outer_weights_eeg = io_dict['outer_tf_mod0_l3.outer_tf.layers.0.self_attn_my.scaled_dotproduct_attention']['output'][1].detach().cpu().numpy()
    outer_weights_eog = io_dict['outer_tf_mod1_l3.outer_tf.layers.0.self_attn_my.scaled_dotproduct_attention']['output'][1].detach().cpu().numpy()

    target = target.detach().cpu().numpy()

    preds = preds.argmax(dim=1)
    inner_shape = int(len(preds)/32)
    preds = einops.rearrange(preds,"(a b) -> a b ", a=32, b=inner_shape).detach().cpu().numpy()

    t = np.arange(0, 29)
    f = (np.arange(0, 128) / 128) * 50
    this_data = data[0][0, 0, 0, 1:, :].detach().numpy()

    # plt.figure()
    # plt.title("The datapoint we examine")
    # plt.xlabel("Time bins (sec)")
    # plt.ylabel("Freq bins (Hz)")
    # plt.pcolormesh(t, f, this_data, vmin=this_data.min(), vmax=this_data.max(), shading='gouraud')
    # plt.colorbar()
    # plt.show()
    #
    colors = ["lightblue" if i else "red" for i in range(len(inner_weights_eeg[0][0]))]
    colors[0] = "lightgreen"

    c = {'Token of Interest': 'lightgreen', "Intermediate Steps": "lightblue"}

    plt.figure()
    plt.subplot(211)
    plt.ylabel("Attention Weight")
    plt.title("CLS Token in EEG")
    x = np.linspace(0,len(inner_weights_eeg[0][0])-1,len(inner_weights_eeg[0][0])).astype(int)
    plt.xticks([],[])
    plt.ylim([0, 0.13])
    handles = [plt.Rectangle((0, 0), 1, 1, color=c[l]) for l in list(c.keys())]
    plt.legend(handles, list(c.keys()))
    plt.bar(x, inner_weights_eeg[0][0], color=colors)
    plt.subplot(212)
    plt.ylim([0, 0.13])
    plt.ylabel("Attention Weight")
    plt.xlabel("Inner Seq Steps")
    plt.title("EOG")
    plt.bar(x, inner_weights_eog[0][0], color=colors)
    plt.show()
    token_of_interest = 0
    idx = -5


    c = {'Token of Interest': 'lightgreen', 'Wrongly classified': 'red', "Correct classified": "lightblue"}
    l = {0: "W", 1: "N1", 2: "N2", 3: "N3", 4: "R"}

    print(target)

    print(outer_weights_eeg.shape)
    print(preds[idx]==target[idx])
    token_of_interests = [0]
    for token_of_interest in token_of_interests:
        plt.figure(figsize=(25, 20))
        plt.subplot(211)
        plt.ylabel("Attention Weight")
        plt.xlabel("Outer Seq Steps")
        plt.title("EEG Tokens")
        x = np.linspace(0,len(outer_weights_eeg[idx][token_of_interest])-1, len(outer_weights_eeg[idx][token_of_interest]) ).astype(int)

        # labels = [l[preds[idx][i]]+"/"+l[target[idx][i]] for i in range(len(target[idx]))]
        labels = [l[target[idx][i]] for i in range(len(target[idx]))]

        colors = preds[idx]== target[idx]
        colors = ["lightblue" if i else "red" for i in colors]
        colors[token_of_interest] = "lightgreen"
        plt.xticks(x, labels, rotation=0)
        plt.ylim([0,0.1])
        plt.xlabel("Correct Label")
        handles = [plt.Rectangle((0, 0), 1, 1, color=c[l]) for l in list(c.keys())]
        plt.legend(handles, list(c.keys()))
        print(outer_weights_eeg[idx][token_of_interest][:len(outer_weights_eeg[idx][token_of_interest])])

        plt.bar(x, outer_weights_eeg[idx][token_of_interest][:len(outer_weights_eeg[idx][token_of_interest])], color=colors, width=0.7)

        plt.subplot(212)
        plt.ylabel("Attention Weight")
        plt.xlabel("Outer Seq Steps")
        plt.title("EOG Tokens")
        # labels = [l[preds[idx][i]]+"/"+l[target[idx][i]] for i in range(len(target[idx]))]
        plt.ylim([0,0.1])
        plt.xticks(x, labels, rotation=0)
        plt.xlabel("Correct Label")
        print(outer_weights_eog[idx][token_of_interest][:len(outer_weights_eog[idx][token_of_interest])])
        plt.bar(x, outer_weights_eog[idx][token_of_interest][:len(outer_weights_eog[idx][token_of_interest])],
                color=colors, width=0.7)

        plt.show()

    # figure_list = []
    # fig, ax = plt.subplots()
    # x = np.linspace(0,29,30).astype(int)
    # for i in range(1,30):
    #     ax.set_ylabel("Attention Weight")
    #     ax.set_xlabel("Inner Seq Steps")
    #     ax.set_title("Inner Tokens in an untrained model")
    #     figure_list.append(ax.bar(x, inner_weights[0][i].detach().numpy(), color='lightblue'))
    # ani = animation.ArtistAnimation(fig, figure_list, blit=False)
    # writer = animation.FFMpegWriter(fps=3, extra_args=['-vcodec', 'libx264'])
    # ani.save('inner.mp4', writer=writer)
    # ani.save('inner.gif', 'imagemagick')


    # figure_list = []
    # fig, ax = plt.subplots()
    # x = np.linspace(0,20,21).astype(int)
    # for i in range(21):
    #     ax.set_ylabel("Attention Weight")
    #     ax.set_xlabel("Outer Seq Steps")
    #     ax.set_title("Tokens in an untrained model")
    #     figure_list.append(ax.bar(x, outer_weights[0][i].detach().numpy(), color='lightblue'))
    # ani = animation.ArtistAnimation(fig, figure_list, blit=False)
    # writer = animation.FFMpegWriter(fps=3, extra_args=['-vcodec', 'libx264'])
    # ani.save('outer.mp4', writer=writer)
    # ani.save('outer.gif', 'imagemagick')
    preds = preds[idx]
    target = target[idx]

    non_matches = (preds != target).astype(int)
    non_matches_idx = non_matches.nonzero()[0]
    print("Non matching indices are:")
    hours = len(target)

    non_matches_idx = non_matches_idx

    pred_plus = copy.deepcopy(preds)
    pred_plus[pred_plus == 4] = 5
    pred_plus[pred_plus == 3] = 4
    pred_plus[pred_plus == 2] = 3
    pred_plus[pred_plus == 5] = 2

    target_plus = copy.deepcopy(target)
    target_plus[target_plus == 4] = 5
    target_plus[target_plus == 3] = 4
    target_plus[target_plus == 2] = 3
    target_plus[target_plus == 5] = 2

    # target = target + 0.02
    target_plus = target_plus + 0.02

    plt.figure()
    plt.plot(pred_plus,label="Prediction")
    plt.plot(target_plus,label="True Label")
    plt.scatter(non_matches_idx, pred_plus[non_matches_idx], marker='*', edgecolors="r", label="Mistakes")
    # plt.plot(non_matches_idx,"*")
    plt.yticks([0, 1, 2, 3, 4], labels=["Wake", "N1", "REM", "N2", "N3"])
    plt.xticks([i * 120 for i in range((hours // 120) + 1)],
               labels=["{}".format(i) for i in range((hours // 120) + 1)])
    plt.legend()
    plt.ylabel("Labels")
    plt.xlabel("Hours")
    plt.show()

    return 0
def get_attention_weights_late_contrastive(model, device, batch, seq_l, data_loader, description):

    # device = "cuda:{}".format(config.gpu_device[0])
    # device = "cpu"

    forward_hook_manager = ForwardHookManager(device)
    num_layers = 4
    for i in range(num_layers):
        forward_hook_manager.add_hook(model, 'enc_0.outer_tf_mod0.outer_tf.layers.{}.self_attn_my.scaled_dotproduct_attention'.format(i), requires_input=False, requires_output=True)
        forward_hook_manager.add_hook(model, 'enc_0.inner_tf_mod0.inner_tf.layers.{}.self_attn_my.scaled_dotproduct_attention'.format(i), requires_input=False, requires_output=True)
        forward_hook_manager.add_hook(model, 'enc_0.outer_tf_mod1.outer_tf.layers.{}.self_attn_my.scaled_dotproduct_attention'.format(i), requires_input=False, requires_output=True)
        forward_hook_manager.add_hook(model, 'enc_0.inner_tf_mod1.inner_tf.layers.{}.self_attn_my.scaled_dotproduct_attention'.format(i), requires_input=False, requires_output=True)

        forward_hook_manager.add_hook(model, 'enc_0.outer_tf_mod0.outer_tf.layers.{}.norm_calc'.format(i), requires_input=False, requires_output=True)
        forward_hook_manager.add_hook(model, 'enc_0.inner_tf_mod0.inner_tf.layers.{}.norm_calc'.format(i), requires_input=False, requires_output=True)
        forward_hook_manager.add_hook(model, 'enc_0.outer_tf_mod1.outer_tf.layers.{}.norm_calc'.format(i), requires_input=False, requires_output=True)
        forward_hook_manager.add_hook(model, 'enc_0.inner_tf_mod1.inner_tf.layers.{}.norm_calc'.format(i), requires_input=False, requires_output=True)

    # forward_hook_manager.add_hook(model, 'enc_0.inner_tf_mod0_l3.inner_tf.layers.0.self_attn_my.scaled_dotproduct_attention', requires_input=True, requires_output=True)
    # forward_hook_manager.add_hook(model, 'enc_0.inner_tf_mod1_l3.inner_tf.layers.0.self_attn_my.scaled_dotproduct_attention', requires_input=True, requires_output=True)
    # forward_hook_manager.add_hook(model, 'enc_0.outer_tf_mod0_l3.outer_tf.layers.0.self_attn_my.scaled_dotproduct_attention', requires_input=True, requires_output=True)
    # forward_hook_manager.add_hook(model, 'enc_0.outer_tf_mod1_l3.outer_tf.layers.0.self_attn_my.scaled_dotproduct_attention', requires_input=True, requires_output=True)
    # forward_hook_manager.add_hook(model, 'enc_0.inner_tf_mod0_l0.inner_tf.layers.0.self_attn_my.scaled_dotproduct_attention', requires_input=True, requires_output=True)
    # forward_hook_manager.add_hook(model, 'enc_0.inner_tf_mod1_l0.inner_tf.layers.0.self_attn_my.scaled_dotproduct_attention', requires_input=True, requires_output=True)
    # forward_hook_manager.add_hook(model, 'enc_0.outer_tf_mod0_l0.outer_tf.layers.0.self_attn_my.scaled_dotproduct_attention', requires_input=True, requires_output=True)
    # forward_hook_manager.add_hook(model, 'enc_0.outer_tf_mod1_l0.outer_tf.layers.0.self_attn_my.scaled_dotproduct_attention', requires_input=True, requires_output=True)
    # forward_hook_manager.add_hook(model, 'enc_0.inner_tf_mod0_l1.inner_tf.layers.0.self_attn_my.scaled_dotproduct_attention', requires_input=True, requires_output=True)
    # forward_hook_manager.add_hook(model, 'enc_0.inner_tf_mod1_l1.inner_tf.layers.0.self_attn_my.scaled_dotproduct_attention', requires_input=True, requires_output=True)
    # forward_hook_manager.add_hook(model, 'enc_0.outer_tf_mod0_l1.outer_tf.layers.0.self_attn_my.scaled_dotproduct_attention', requires_input=True, requires_output=True)
    # forward_hook_manager.add_hook(model, 'enc_0.outer_tf_mod1_l1.outer_tf.layers.0.self_attn_my.scaled_dotproduct_attention', requires_input=True, requires_output=True)
    # forward_hook_manager.add_hook(model, 'enc_0.inner_tf_mod0_l2.inner_tf.layers.0.self_attn_my.scaled_dotproduct_attention', requires_input=True, requires_output=True)
    # forward_hook_manager.add_hook(model, 'enc_0.inner_tf_mod1_l2.inner_tf.layers.0.self_attn_my.scaled_dotproduct_attention', requires_input=True, requires_output=True)
    # forward_hook_manager.add_hook(model, 'enc_0.outer_tf_mod0_l2.outer_tf.layers.0.self_attn_my.scaled_dotproduct_attention', requires_input=True, requires_output=True)
    # forward_hook_manager.add_hook(model, 'enc_0.outer_tf_mod1_l2.outer_tf.layers.0.self_attn_my.scaled_dotproduct_attention', requires_input=True, requires_output=True)

    # forward_hook_manager.add_hook(model, 'enc_0.outer_tf_mod0_l0.outer_tf.layers.0.norm_calc', requires_input=False, requires_output=True)
    # forward_hook_manager.add_hook(model, 'enc_0.outer_tf_mod0_l1.outer_tf.layers.0.norm_calc', requires_input=False, requires_output=True)
    # forward_hook_manager.add_hook(model, 'enc_0.outer_tf_mod0_l2.outer_tf.layers.0.norm_calc', requires_input=False, requires_output=True)
    # forward_hook_manager.add_hook(model, 'enc_0.outer_tf_mod0_l3.outer_tf.layers.0.norm_calc', requires_input=False, requires_output=True)
    #
    # forward_hook_manager.add_hook(model, 'enc_0.outer_tf_mod1_l0.outer_tf.layers.0.norm_calc', requires_input=False, requires_output=True)
    # forward_hook_manager.add_hook(model, 'enc_0.outer_tf_mod1_l1.outer_tf.layers.0.norm_calc', requires_input=False, requires_output=True)
    # forward_hook_manager.add_hook(model, 'enc_0.outer_tf_mod1_l2.outer_tf.layers.0.norm_calc', requires_input=False, requires_output=True)
    # forward_hook_manager.add_hook(model, 'enc_0.outer_tf_mod1_l3.outer_tf.layers.0.norm_calc', requires_input=False, requires_output=True)

    # forward_hook_manager.add_hook(model, 'layer1.0.bn2', requires_input=True, requires_output=True)
    # forward_hook_manager.add_hook(model, 'fc', requires_input=False, requires_output=True)

    model.eval()
    pbar = tqdm(enumerate(data_loader), desc=description, leave=False)
    for batch_idx, (data, target, init, ids) in pbar:

        views = [data[i].float().to(device) for i in range(len(data))]
        # views[0] = torch.cat([views[0][:16],views[0][16:]],dim=1)
        # views[1] = torch.cat([views[1][:16],views[1][16:]],dim=1)
        preds = get_predictions_time_series(model, views, init, extract_norm=True)
        break

    # batch, seq_l = 16, 42

    io_dict = forward_hook_manager.pop_io_dict()

    outer_norms_eeg = {"norm":{}}
    outer_norms_eog = {"norm":{}}
    inner_norms_eeg = {"norm":{}}
    inner_norms_eog = {"norm":{}}
    for i in range(num_layers):
        outer_norms_eeg["norm"]["layer_{}".format(i)] = io_dict['enc_0.outer_tf_mod0.outer_tf.layers.{}.norm_calc'.format(i)]['output']
        inner_norms_eeg["norm"]["layer_{}".format(i)] = io_dict['enc_0.inner_tf_mod0.inner_tf.layers.{}.norm_calc'.format(i)]['output']
        outer_norms_eog["norm"]["layer_{}".format(i)] = io_dict['enc_0.outer_tf_mod1.outer_tf.layers.{}.norm_calc'.format(i)]['output']
        inner_norms_eog["norm"]["layer_{}".format(i)] = io_dict['enc_0.inner_tf_mod1.inner_tf.layers.{}.norm_calc'.format(i)]['output']

    inner_weights_eeg = {"att":{}}
    inner_weights_eog = {"att":{}}
    outer_weights_eeg = {"att":{}}
    outer_weights_eog = {"att":{}}

    for i in range(num_layers):
        inner_weights_eeg["att"]["layer_{}".format(i)] = io_dict['enc_0.inner_tf_mod0.inner_tf.layers.{}.self_attn_my.scaled_dotproduct_attention'.format(i)]['output']
        inner_weights_eog["att"]["layer_{}".format(i)] = io_dict['enc_0.inner_tf_mod0.inner_tf.layers.{}.self_attn_my.scaled_dotproduct_attention'.format(i)]['output']
        outer_weights_eeg["att"]["layer_{}".format(i)] = io_dict['enc_0.outer_tf_mod1.outer_tf.layers.{}.self_attn_my.scaled_dotproduct_attention'.format(i)]['output']
        outer_weights_eog["att"]["layer_{}".format(i)] = io_dict['enc_0.outer_tf_mod1.outer_tf.layers.{}.self_attn_my.scaled_dotproduct_attention'.format(i)]['output']

    # print(io_dict.keys())
    # inner_weights_eeg = {}
    # inner_weights_eeg["layer_0"] = io_dict['enc_0.inner_tf_mod0_l0.inner_tf.layers.0.self_attn_my.scaled_dotproduct_attention']['output'][1].detach().cpu().numpy()
    # inner_weights_eeg["layer_1"] = io_dict['enc_0.inner_tf_mod0_l1.inner_tf.layers.0.self_attn_my.scaled_dotproduct_attention']['output'][1].detach().cpu().numpy()
    # inner_weights_eeg["layer_2"] = io_dict['enc_0.inner_tf_mod0_l2.inner_tf.layers.0.self_attn_my.scaled_dotproduct_attention']['output'][1].detach().cpu().numpy()
    # inner_weights_eeg["layer_3"] = io_dict['enc_0.inner_tf_mod0_l3.inner_tf.layers.0.self_attn_my.scaled_dotproduct_attention']['output'][1].detach().cpu().numpy()
    #
    # inner_weights_eog = {}
    # inner_weights_eog["layer_0"] = io_dict['enc_0.inner_tf_mod1_l0.inner_tf.layers.0.self_attn_my.scaled_dotproduct_attention']['output'][1].detach().cpu().numpy()
    # inner_weights_eog["layer_1"] = io_dict['enc_0.inner_tf_mod1_l1.inner_tf.layers.0.self_attn_my.scaled_dotproduct_attention']['output'][1].detach().cpu().numpy()
    # inner_weights_eog["layer_2"] = io_dict['enc_0.inner_tf_mod1_l2.inner_tf.layers.0.self_attn_my.scaled_dotproduct_attention']['output'][1].detach().cpu().numpy()
    # inner_weights_eog["layer_3"] = io_dict['enc_0.inner_tf_mod1_l3.inner_tf.layers.0.self_attn_my.scaled_dotproduct_attention']['output'][1].detach().cpu().numpy()
    #
    # outer_weights_eeg = {}
    # outer_weights_eeg["layer_0"] = io_dict['enc_0.outer_tf_mod0_l0.outer_tf.layers.0.self_attn_my.scaled_dotproduct_attention']['output'][1].detach().cpu().numpy()
    # outer_weights_eeg["layer_1"] = io_dict['enc_0.outer_tf_mod0_l1.outer_tf.layers.0.self_attn_my.scaled_dotproduct_attention']['output'][1].detach().cpu().numpy()
    # outer_weights_eeg["layer_2"] = io_dict['enc_0.outer_tf_mod0_l2.outer_tf.layers.0.self_attn_my.scaled_dotproduct_attention']['output'][1].detach().cpu().numpy()
    # outer_weights_eeg["layer_3"] = io_dict['enc_0.outer_tf_mod0_l3.outer_tf.layers.0.self_attn_my.scaled_dotproduct_attention']['output'][1].detach().cpu().numpy()
    #
    # outer_weights_eog = {}
    # outer_weights_eog["layer_0"] = io_dict['enc_0.outer_tf_mod1_l0.outer_tf.layers.0.self_attn_my.scaled_dotproduct_attention']['output'][1].detach().cpu().numpy()
    # outer_weights_eog["layer_1"] = io_dict['enc_0.outer_tf_mod1_l1.outer_tf.layers.0.self_attn_my.scaled_dotproduct_attention']['output'][1].detach().cpu().numpy()
    # outer_weights_eog["layer_2"] = io_dict['enc_0.outer_tf_mod1_l2.outer_tf.layers.0.self_attn_my.scaled_dotproduct_attention']['output'][1].detach().cpu().numpy()
    # outer_weights_eog["layer_3"] = io_dict['enc_0.outer_tf_mod1_l3.outer_tf.layers.0.self_attn_my.scaled_dotproduct_attention']['output'][1].detach().cpu().numpy()
    #
    # outer_norms_eeg = {}
    # outer_norms_eeg["layer_0"] = io_dict['enc_0.outer_tf_mod0_l0.outer_tf.layers.0.norm_calc']['output']
    # outer_norms_eeg["layer_1"] = io_dict['enc_0.outer_tf_mod0_l1.outer_tf.layers.0.norm_calc']['output']
    # outer_norms_eeg["layer_2"] = io_dict['enc_0.outer_tf_mod0_l2.outer_tf.layers.0.norm_calc']['output']
    # outer_norms_eeg["layer_3"] = io_dict['enc_0.outer_tf_mod0_l3.outer_tf.layers.0.norm_calc']['output']
    #
    # outer_norms_eog = {}
    # outer_norms_eog["layer_0"] = io_dict['enc_0.outer_tf_mod1_l0.outer_tf.layers.0.norm_calc']['output']
    # outer_norms_eog["layer_1"] = io_dict['enc_0.outer_tf_mod1_l1.outer_tf.layers.0.norm_calc']['output']
    # outer_norms_eog["layer_2"] = io_dict['enc_0.outer_tf_mod1_l2.outer_tf.layers.0.norm_calc']['output']
    # outer_norms_eog["layer_3"] = io_dict['enc_0.outer_tf_mod1_l3.outer_tf.layers.0.norm_calc']['output']

    target = target.detach().cpu().numpy()
    preds = preds.argmax(dim=1)
    preds = einops.rearrange(preds,"(a b) -> a b ", a=batch, b=seq_l).detach().cpu().numpy()

    print(target)

    total_layers = 4
    total_heads = 8
    batch_idx = -1
    token_of_interest = -4

    l = {0: "W", 1: "N1", 2: "N2", 3: "N3", 4: "R"}
    labels = [l[target[batch_idx][i % 21]] for i in range(len(target[batch_idx]))]

    neigh_rest_ratio_per_layer = []
    diag_rest_ratio_per_layer = []
    for layer in range(4):
        head_attn_n, attn_n, attnres_n, attnresln_n, attn_n_ratio, attnres_n_ratio, attnresln_n_ratio = outer_norms_eeg["norm"]["layer_{}".format(layer)]

        neigh_diag = torch.ones(attnresln_n.shape[2] - 1, dtype=torch.long)
        neigh_diag_mask = torch.diagflat(neigh_diag, offset=1) + torch.diagflat(neigh_diag, offset=-1)
        neigh_diag_2_mask = torch.diagflat(torch.ones(attnresln_n.shape[2] - 2, dtype=torch.long), offset=2) + torch.diagflat(torch.ones(attnresln_n.shape[2] - 2, dtype=torch.long), offset=-2)
        neigh_diag_mask += neigh_diag_2_mask

        diag_mask = torch.eye(attnresln_n.shape[2])
        rest_mask = (torch.ones([attnresln_n.shape[2], attnresln_n.shape[2]], dtype=torch.long) - diag_mask - neigh_diag_mask) > 0
        neigh_diag_mask = neigh_diag_mask > 0

        attnresln_n_diag = attnresln_n[:,diag_mask>0]
        attnresln_n_rest = attnresln_n[:,diag_mask<1]

        attnresln_n_neigh_rest = attnresln_n[:, rest_mask]
        attnresln_n_neigh = attnresln_n[:, neigh_diag_mask]

        neigh_rest_ratio = attnresln_n_neigh.mean() / (attnresln_n_neigh_rest.mean() + attnresln_n_neigh.mean())
        diag_rest_ratio = attnresln_n_rest.mean() / (attnresln_n_rest.mean() + attnresln_n_diag.mean())

        neigh_rest_ratio_per_layer.append(neigh_rest_ratio.detach().cpu().numpy())
        diag_rest_ratio_per_layer.append(diag_rest_ratio.detach().cpu().numpy())

        print("Our neighboring ratio is {}".format(neigh_rest_ratio))
        print("Our diag ratio is {}".format(diag_rest_ratio))
        plt.figure()
        df = pd.DataFrame(attnresln_n[batch_idx].detach().cpu().numpy(), columns=labels , index=labels)
        sns.heatmap(df, cmap="Blues", square=True, annot=True, fmt='.1g', annot_kws={"fontsize": 4})
        plt.title("Context Ratio layer {}".format(layer))
        plt.show()

    t = np.concatenate(
        [np.expand_dims(np.array(neigh_rest_ratio_per_layer),axis=0),
        np.expand_dims(np.array(diag_rest_ratio_per_layer),axis=0)],
        axis=0)
    plt.figure()
    df = pd.DataFrame(t , columns=["Layer 0","Layer 1","Layer 2","Layer 3"] , index=["Neighbor R","Context R"])
    sns.heatmap(df, cmap="Blues", square=True, annot=True, fmt='.3g', annot_kws={"fontsize": 8})
    plt.title("Ratios Per Layer")
    plt.show()

    #   hidden_states: Representations from previous layer and inputs to self-attention. (batch, seq_length, all_head_size)
    #   attention_probs: Attention weights calculated in self-attention. (batch, num_heads, seq_length, seq_length)
    #   value_layer: Value vectors calculated in self-attention. (batch, num_heads, seq_length, head_size)
    #   dense: Dense layer in self-attention. nn.Linear(all_head_size, all_head_size)
    #   LayerNorm: nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
    #   pre_ln_states: Vectors just before LayerNorm (batch, seq_length, all_head_size)

    weights = [inner_weights_eeg["att"], inner_weights_eog["att"], outer_weights_eeg["att"], outer_weights_eog["att"]]
    labels = ["Inner Weights EEG", "Inner Weights EOG", "Outer Weights EEG", "Outer Weights EOG"]

    for plot_num in range(len(weights)):
        fig, ax = plt.subplots()
        plt.title(labels[plot_num])
        plt.ylabel("Layers")
        plt.xlabel("Heads")
        plt.box(on=None)
        plt.xticks([])
        plt.yticks([])
        for l in range(total_layers):
            w = weights[plot_num]["layer_{}".format(l)][1]
            if plot_num<2:
                w = einops.rearrange(w, "(outer inner h) d m-> outer inner h d m", outer=batch, inner=seq_l, h=total_heads)
                w = w[batch_idx][token_of_interest].detach().cpu().numpy()
            else:
                w = einops.rearrange(w, "(outer h) d m-> outer h d m", outer=batch, h=total_heads)
                w = w[batch_idx].detach().cpu().numpy()
            for h in range(total_heads):
                current_subplot = (l * total_heads) + h + 1
                ax = fig.add_subplot(total_layers, total_heads, current_subplot)
                ax.axis('off')
                ax.imshow(w[h], cmap='OrRd_r', interpolation='nearest')
        plt.subplots_adjust(wspace=0.05, hspace=0)
        plt.show()

    preds = preds[batch_idx]
    target = target[batch_idx]

    non_matches = (preds != target).astype(int)
    non_matches_idx = non_matches.nonzero()[0]
    print("Non matching indices are:")
    hours = len(target)

    non_matches_idx = non_matches_idx

    pred_plus = copy.deepcopy(preds)
    pred_plus[pred_plus == 4] = 5
    pred_plus[pred_plus == 3] = 4
    pred_plus[pred_plus == 2] = 3
    pred_plus[pred_plus == 5] = 2

    target_plus = copy.deepcopy(target)
    target_plus[target_plus == 4] = 5
    target_plus[target_plus == 3] = 4
    target_plus[target_plus == 2] = 3
    target_plus[target_plus == 5] = 2

    # target = target + 0.02
    target_plus = target_plus + 0.02

    plt.figure()
    plt.plot(pred_plus,label="Prediction")
    plt.plot(target_plus,label="True Label")
    plt.scatter(non_matches_idx, pred_plus[non_matches_idx], marker='*', edgecolors="r", label="Mistakes")
    # plt.plot(non_matches_idx,"*")
    plt.yticks([0, 1, 2, 3, 4], labels=["Wake", "N1", "REM", "N2", "N3"])
    plt.xticks([i * 120 for i in range((hours // 120) + 1)],
               labels=["{}".format(i) for i in range((hours // 120) + 1)])
    plt.legend()
    plt.ylabel("Labels")
    plt.xlabel("Hours")
    plt.show()

    return 0

def get_attention_weights_late_norm(model, device, batch, seq_l, data_loader, description):

    # device = "cuda:{}".format(config.gpu_device[0])
    # device = "cpu"
    forward_hook_manager = ForwardHookManager(device)
    num_layers = 4
    for i in range(num_layers):
        forward_hook_manager.add_hook(model, 'enc_0.outer_tf_mod0.outer_tf.layers.{}.norm_calc'.format(i), requires_input=False, requires_output=True)
        forward_hook_manager.add_hook(model, 'enc_0.inner_tf_mod0.inner_tf.layers.{}.norm_calc'.format(i), requires_input=False, requires_output=True)

    # forward_hook_manager.add_hook(model, 'enc_0.inner_tf_mod1', requires_output=True)
    # forward_hook_manager.add_hook(model, 'enc_0.outer_tf_mod1', requires_output=True)

    model.eval()
    pbar = tqdm(enumerate(data_loader), desc=description, leave=False)
    for batch_idx, (data, target, init, ids) in pbar:

        views = [data[i].float().to(device) for i in range(len(data))]
        # views[0] = torch.cat([views[0][:16],views[0][16:]],dim=1)
        # views[1] = torch.cat([views[1][:16],views[1][16:]],dim=1)
        preds = get_predictions_time_series(model, views, init, extract_norm=True)
        break

    # batch, seq_l = 16, 42

    io_dict = forward_hook_manager.pop_io_dict()
    # print(io_dict.keys())
    # inner_weights_eeg = {}
    # inner_weights_eeg["layer_0"] = io_dict['enc_0.inner_tf_mod0_l0.inner_tf.layers.0.self_attn_my.scaled_dotproduct_attention']['output'][1].detach().cpu().numpy()
    # inner_weights_eeg["layer_1"] = io_dict['enc_0.inner_tf_mod0_l1.inner_tf.layers.0.self_attn_my.scaled_dotproduct_attention']['output'][1].detach().cpu().numpy()
    # inner_weights_eeg["layer_2"] = io_dict['enc_0.inner_tf_mod0_l2.inner_tf.layers.0.self_attn_my.scaled_dotproduct_attention']['output'][1].detach().cpu().numpy()
    # inner_weights_eeg["layer_3"] = io_dict['enc_0.inner_tf_mod0_l3.inner_tf.layers.0.self_attn_my.scaled_dotproduct_attention']['output'][1].detach().cpu().numpy()
    #
    # inner_weights_eog = {}
    # inner_weights_eog["layer_0"] = io_dict['enc_0.inner_tf_mod1_l0.inner_tf.layers.0.self_attn_my.scaled_dotproduct_attention']['output'][1].detach().cpu().numpy()
    # inner_weights_eog["layer_1"] = io_dict['enc_0.inner_tf_mod1_l1.inner_tf.layers.0.self_attn_my.scaled_dotproduct_attention']['output'][1].detach().cpu().numpy()
    # inner_weights_eog["layer_2"] = io_dict['enc_0.inner_tf_mod1_l2.inner_tf.layers.0.self_attn_my.scaled_dotproduct_attention']['output'][1].detach().cpu().numpy()
    # inner_weights_eog["layer_3"] = io_dict['enc_0.inner_tf_mod1_l3.inner_tf.layers.0.self_attn_my.scaled_dotproduct_attention']['output'][1].detach().cpu().numpy()

    outer_norms = {"norm":{}}
    inner_norms = {"norm":{}}
    for i in range(num_layers):
        outer_norms["norm"]["layer_{}".format(i)] = io_dict['enc_0.outer_tf_mod0.outer_tf.layers.{}.norm_calc'.format(i)]['output']
        inner_norms["norm"]["layer_{}".format(i)] = io_dict['enc_0.inner_tf_mod0.inner_tf.layers.{}.norm_calc'.format(i)]['output']

    target = target.detach().cpu().numpy()
    for weights in [outer_norms]:

        idx = -5
        l = {0: "W", 1: "N1", 2: "N2", 3: "N3", 4: "R"}
        labels = [l[target[idx][i % 21]] for i in range(len(target[idx]))]

        neigh_rest_ratio_per_layer = []
        diag_rest_ratio_per_layer = []
        for layer in range(num_layers):
            head_attn_n, attn_n, attnres_n, attnresln_n, attn_n_ratio, attnres_n_ratio, attnresln_n_ratio = weights["norm"]["layer_{}".format(layer)]

            neigh_diag = torch.ones(attnresln_n.shape[2] - 1, dtype=torch.long)
            neigh_diag_mask = torch.diagflat(neigh_diag, offset=1) + torch.diagflat(neigh_diag, offset=-1)
            neigh_diag_2_mask = torch.diagflat(torch.ones(attnresln_n.shape[2] - 2, dtype=torch.long), offset=2) + torch.diagflat(torch.ones(attnresln_n.shape[2] - 2, dtype=torch.long), offset=-2)
            neigh_diag_3_mask = torch.diagflat(torch.ones(attnresln_n.shape[2] - 3, dtype=torch.long), offset=3) + torch.diagflat(torch.ones(attnresln_n.shape[2] - 3, dtype=torch.long), offset=-3)
            neigh_diag_mask += neigh_diag_2_mask +neigh_diag_3_mask

            diag_mask = torch.eye(attnresln_n.shape[2])
            rest_mask = (torch.ones([attnresln_n.shape[2], attnresln_n.shape[2]], dtype=torch.long) - diag_mask - neigh_diag_mask) > 0
            neigh_diag_mask = neigh_diag_mask > 0

            norms_matrix = attnresln_n

            attnresln_n_diag = norms_matrix[:,diag_mask>0]
            attnresln_n_rest = norms_matrix[:,diag_mask<1]

            attnresln_n_neigh_rest = norms_matrix[:, rest_mask]
            attnresln_n_neigh = norms_matrix[:, neigh_diag_mask]

            neigh_rest_ratio = attnresln_n_neigh.mean() / (attnresln_n_neigh_rest.mean() + attnresln_n_neigh.mean())
            diag_rest_ratio = attnresln_n_rest.mean() / (attnresln_n_rest.mean() + attnresln_n_diag.mean())

            neigh_rest_ratio_per_layer.append(neigh_rest_ratio.detach().cpu().numpy())
            diag_rest_ratio_per_layer.append(diag_rest_ratio.detach().cpu().numpy())

            print("Our neighboring ratio is {}".format(neigh_rest_ratio))
            print("Our diag ratio is {}".format(diag_rest_ratio))
            plt.figure()
            # df = pd.DataFrame(attnresln_n[idx].detach().cpu().numpy(), columns=labels , index=labels)
            df = pd.DataFrame(norms_matrix[idx].detach().cpu().numpy())
            sns.heatmap(df, cmap="Blues", square=True, annot=True, fmt='.1g', annot_kws={"fontsize": 4})
            plt.title("Context Ratio layer {}".format(layer))
            plt.show()

        t = np.concatenate(
            [np.expand_dims(np.array(neigh_rest_ratio_per_layer),axis=0),
            np.expand_dims(np.array(diag_rest_ratio_per_layer),axis=0)],
            axis=0)
        plt.figure()
        layer_columns = ["Layer {}".format(i) for i in range(num_layers)]
        df = pd.DataFrame(t , columns=layer_columns , index=["Neighbor R","Context R"])
        sns.heatmap(df, cmap="Blues", square=True, annot=True, fmt='.3g', annot_kws={"fontsize": 8})
        plt.title("Ratios Per Layer")
        plt.show()

    return 0

    for l_i in range(4):
        plt.figure()
        summed_afx_norm = weights["norm"][l_i][3]
        print(summed_afx_norm.shape)
        norm = summed_afx_norm.detach().cpu()
        diag_mask = torch.eye(norm.shape[2])
        # norm[:, diag_mask > 0] = 0
        norm = norm[idx].numpy()
        print(norm.shape)
        # df = pd.DataFrame(norm )
        df = pd.DataFrame(norm, columns=labels, index=labels )
        sns.heatmap(df, cmap="Blues", square=True, annot=True, fmt='.1g', annot_kws={"fontsize":4})
        plt.gcf().subplots_adjust(bottom=0.2)
        plt.title("neighboring ratio layer {}".format(l_i))
        plt.show()

    return 0

    print(attnresln_n_ratio)

    fig = plt.figure()
    grid = plt.GridSpec(2, 6)
    ratio_inner = torch.cat([inner_weights_eeg["norm"][i][-1][0].unsqueeze(dim=0) for i in range(4)],dim=0).cpu().numpy()
    ratio_outer = torch.cat([outer_weights_eeg["norm"][i][-1][0].unsqueeze(dim=0) for i in range(4)],dim=0).cpu().numpy()
    df_inner = pd.DataFrame(ratio_inner)
    df_outer = pd.DataFrame(ratio_outer)
    plt.subplot(grid[0,0:5])
    sns.heatmap(df_inner, cmap="Reds", square=True)
    plt.gcf().subplots_adjust(bottom=0.2)
    plt.xticks([])
    plt.title("Inner Ratio")
    plt.ylabel("Layers")

    fig.axes[1].set_visible(False)
    plt.subplot(grid[1,0:5])
    plt.title("Outer Ratio")
    sns.heatmap(df_outer, cmap="Reds", square=True)
    plt.gcf().subplots_adjust(bottom=0.2)
    plt.xlabel("Sequence")
    plt.ylabel("Layers")
    plt.show()

    fig = plt.figure()
    grid = plt.GridSpec(4, 12)
    for layer in range(4):
        plt.subplot(grid[layer, 0:5])
        summed_afx_norm = inner_weights_eeg["norm"][layer][3]
        norm = summed_afx_norm[0].cpu().numpy()
        df = pd.DataFrame(norm)
        sns.heatmap(df, cmap="Reds", square=True, cbar=False)
        plt.gcf().subplots_adjust(bottom=0.2)
        # fig.axes[layer*2+1].set_visible(False)

        plt.subplot(grid[layer, 6:11])
        summed_afx_norm = inner_weights_eeg["norm"][layer][3]
        norm = summed_afx_norm[0].cpu().numpy()
        df = pd.DataFrame(norm)
        sns.heatmap(df, cmap="Reds", square=True, cbar=False)
        plt.gcf().subplots_adjust(bottom=0.2)
        # fig.axes[(layer+1)*2+1].set_visible(False)


        # plt.title("AttnResLn-N visualization Layer {}".format(layer+1))
    plt.show()


    # Set the layer and head you want to check. (layer: 1~12, head: 1~12)
    layer = 4
    head = 8
    target = target.detach().cpu().numpy()

    plt.figure()
    attention = weights["attention"][layer - 1][0][head - 1].detach().cpu().numpy()
    df = pd.DataFrame(attention)
    sns.heatmap(df, cmap="Reds", square=True)
    plt.gcf().subplots_adjust(bottom=0.2)
    plt.title("Attn-W visualization head")
    plt.show()

    plt.figure()
    afx_norm = weights["norm"][layer - 1][0]
    norm = afx_norm[0][head - 1].cpu().numpy()
    df = pd.DataFrame(norm)
    sns.heatmap(df, cmap="Reds", square=True)
    plt.gcf().subplots_adjust(bottom=0.2)
    plt.title("Attn-N visualization head")
    plt.show()

    plt.figure()
    attention = weights["attention"][layer - 1][0].mean(0).detach().cpu().numpy()
    df = pd.DataFrame(attention)
    sns.heatmap(df, cmap="Reds", square=True)
    plt.gcf().subplots_adjust(bottom=0.2)
    plt.title("Attn-W visualization layer")
    plt.show()

    plt.figure()
    summed_afx_norm = weights["norm"][layer - 1][1]
    norm = summed_afx_norm[0].cpu().numpy()
    df = pd.DataFrame(norm)
    sns.heatmap(df, cmap="Reds", square=True)
    plt.gcf().subplots_adjust(bottom=0.2)
    plt.title("Attn-N visualization layer")
    plt.show()

    plt.figure()
    attention = weights["attention"][layer - 1][0].mean(0).detach().cpu().numpy()
    res = np.zeros((len(attention), len(attention)), int)
    np.fill_diagonal(res, 1)
    attnres_w = 0.5 * attention + 0.5 * res
    df = pd.DataFrame(attnres_w)
    sns.heatmap(df, cmap="Reds", square=True)
    plt.gcf().subplots_adjust(bottom=0.2)
    plt.title("AttnRes-W visualization layer")
    plt.show()

    plt.figure()
    summed_afx_norm = weights["norm"][layer - 1][2]
    norm = summed_afx_norm[0].cpu().numpy()
    df = pd.DataFrame(norm)
    sns.heatmap(df, cmap="Reds", square=True)
    plt.gcf().subplots_adjust(bottom=0.2)
    plt.title("AttnRes-N visualization")
    plt.show()


    return 0

    target = target.detach().cpu().numpy()
    preds = preds.argmax(dim=1)
    preds = einops.rearrange(preds,"(a b) -> a b ", a=batch, b=seq_l).detach().cpu().numpy()

    print(target)

    total_layers = 4
    total_heads = 8
    batch_idx = -4
    token_of_interest = 0

    print(outer_weights_eeg["af"]["layer_0"].shape)
    print(outer_weights_eeg["f"]["layer_0"].shape)
    print(outer_weights_eeg["attention"]["layer_0"].shape)

    # weights = [outer_weights_eeg["af"], outer_weights_eeg["f"], outer_weights_eeg, outer_weights_eog]
    # labels = [ "Outer Weights EEG"]
    fig, ax = plt.subplots()
    plt.title("Function Norm Analysis")
    plt.box(on=None)
    plt.xticks([])
    plt.yticks([])

    total_image = []
    for l in range(total_layers):
        w = outer_weights_eeg["attention"]["layer_{}".format(l)]
        print(w.shape)
        w = einops.rearrange(w, "(batch h) a b-> batch h (a b)", batch=batch,  h=total_heads)
        w = np.linalg.norm(w[batch_idx], axis=-1)
        total_image.append(np.expand_dims(w,axis=0))
    total_image = numpy.concatenate(total_image, axis=0)

    ax = fig.add_subplot(1, 3, 1)
    # ax.axis('off')
    ax.imshow(total_image, cmap='OrRd_r', interpolation='nearest')
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    ax.set_xlabel("Heads")
    ax.set_ylabel("Layers")
    ax.set_title("|Att|")

    total_image = []
    for l in range(total_layers):
        w = outer_weights_eeg["f"]["layer_{}".format(l)]
        w = einops.rearrange(w, "outer (batch h) f-> batch h (outer f)", batch=batch,  h=total_heads)
        w = np.linalg.norm(w[batch_idx], axis=-1)
        total_image.append(np.expand_dims(w,axis=0))
    total_image = numpy.concatenate(total_image, axis=0)

    ax = fig.add_subplot(1, 3, 2)
    # ax.axis('off')
    ax.imshow(total_image, cmap='OrRd_r', interpolation='nearest')
    ax.set_xlabel("Heads")
    ax.set_ylabel("Layers")
    ax.set_title("|F(x)|")
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])

    total_image = []
    for l in range(total_layers):
        w = outer_weights_eeg["af"]["layer_{}".format(l)]
        w = einops.rearrange(w, "outer (batch h) f-> batch h (outer f)", batch=batch,  h=total_heads)
        w = np.linalg.norm(w[batch_idx], axis=-1)
        total_image.append(np.expand_dims(w,axis=0))
    total_image = numpy.concatenate(total_image, axis=0)

    ax = fig.add_subplot(1, 3, 3)
    # ax.axis('off')
    ax.imshow(total_image, cmap='OrRd_r', interpolation='nearest')
    ax.set_xlabel("Heads")
    ax.set_ylabel("Layers")
    ax.set_title("|α F(x)|")
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])

    plt.subplots_adjust(wspace=0.15, hspace=0.15)
    plt.show()


    fig, ax = plt.subplots()
    plt.title("Attention weights")
    plt.ylabel("Layers")
    plt.xlabel("Heads")
    plt.box(on=None)
    plt.xticks([])
    plt.yticks([])
    for l in range(total_layers):
        w = outer_weights_eeg["attention"]["layer_{}".format(l)]
        w = einops.rearrange(w, "(outer h) d m-> outer h d m", outer=batch, h=total_heads)
        w = w[batch_idx]
        for h in range(total_heads):
            current_subplot = (l * total_heads) + h + 1
            ax = fig.add_subplot(total_layers, total_heads, current_subplot)
            ax.axis('off')
            ax.imshow(w[h], cmap='OrRd_r', interpolation='nearest')
    plt.subplots_adjust(wspace=0.05, hspace=0)
    plt.show()

    fig, ax = plt.subplots()
    plt.title("Attention weights normed with F")
    plt.ylabel("Layers")
    plt.xlabel("Heads")
    plt.box(on=None)
    plt.xticks([])
    plt.yticks([])
    for l in range(total_layers):
        w = outer_weights_eeg["attention"]["layer_{}".format(l)]
        f = outer_weights_eeg["f"]["layer_{}".format(l)]
        f = einops.rearrange(f, "outer (batch h) f-> batch outer (h f)", batch=batch,  h=total_heads)
        f = np.linalg.norm(f[batch_idx], axis=-1)
        print(f.shape)
        w = einops.rearrange(w, "(b h) d m-> b h d m", b=batch, h=total_heads)
        w = w[batch_idx]

        for i in range(len(f)):
            w[:,:,i] *= f[i]

        from scipy.special import softmax

        for h in range(total_heads):
            current_subplot = (l * total_heads) + h + 1
            ax = fig.add_subplot(total_layers, total_heads, current_subplot)
            ax.axis('off')
            ax.imshow( softmax(w[h],axis=-1), cmap='OrRd_r', interpolation='nearest')
    plt.subplots_adjust(wspace=0.05, hspace=0)
    plt.show()

    return 0
def get_attention_weights_late_retarded_norm(model, device, batch, seq_l, data_loader, description):

    # device = "cuda:{}".format(config.gpu_device[0])
    # device = "cpu"
    forward_hook_manager = ForwardHookManager(device)
    num_layers = 4
    for i in range(num_layers):
        forward_hook_manager.add_hook(model, 'enc_0.outer_tf_mod0_l{}.outer_tf.layers.0.norm_calc'.format(i), requires_input=False, requires_output=True)
        forward_hook_manager.add_hook(model, 'enc_0.inner_tf_mod0_l{}.inner_tf.layers.0.norm_calc'.format(i), requires_input=False, requires_output=True)

    # forward_hook_manager.add_hook(model, 'enc_0.inner_tf_mod1', requires_output=True)
    # forward_hook_manager.add_hook(model, 'enc_0.outer_tf_mod1', requires_output=True)

    model.eval()
    pbar = tqdm(enumerate(data_loader), desc=description, leave=False)
    for batch_idx, (data, target, init, ids) in pbar:

        views = [data[i].float().to(device) for i in range(len(data))]
        # views[0] = torch.cat([views[0][:16],views[0][16:]],dim=1)
        # views[1] = torch.cat([views[1][:16],views[1][16:]],dim=1)
        preds = get_predictions_time_series(model, views, init, extract_norm=True)
        break

    # batch, seq_l = 16, 42

    io_dict = forward_hook_manager.pop_io_dict()

    outer_norms = {"norm":{}}
    inner_norms = {"norm":{}}
    for i in range(num_layers):
        outer_norms["norm"]["layer_{}".format(i)] = io_dict['enc_0.outer_tf_mod0_l{}.outer_tf.layers.0.norm_calc'.format(i)]['output']
        inner_norms["norm"]["layer_{}".format(i)] = io_dict['enc_0.inner_tf_mod0_l{}.inner_tf.layers.0.norm_calc'.format(i)]['output']

    target = target.detach().cpu().numpy()
    for weights in [outer_norms]:

        idx = -5
        l = {0: "W", 1: "N1", 2: "N2", 3: "N3", 4: "R"}
        labels = [l[target[idx][i % 21]] for i in range(len(target[idx]))]

        neigh_rest_ratio_per_layer = []
        diag_rest_ratio_per_layer = []
        for layer in range(num_layers):
            head_attn_n, attn_n, attnres_n, attnresln_n, attn_n_ratio, attnres_n_ratio, attnresln_n_ratio = weights["norm"]["layer_{}".format(layer)]

            neigh_diag = torch.ones(attnresln_n.shape[2] - 1, dtype=torch.long)
            neigh_diag_mask = torch.diagflat(neigh_diag, offset=1) + torch.diagflat(neigh_diag, offset=-1)
            neigh_diag_2_mask = torch.diagflat(torch.ones(attnresln_n.shape[2] - 2, dtype=torch.long), offset=2) + torch.diagflat(torch.ones(attnresln_n.shape[2] - 2, dtype=torch.long), offset=-2)
            neigh_diag_3_mask = torch.diagflat(torch.ones(attnresln_n.shape[2] - 3, dtype=torch.long), offset=3) + torch.diagflat(torch.ones(attnresln_n.shape[2] - 3, dtype=torch.long), offset=-3)
            neigh_diag_mask += neigh_diag_2_mask +neigh_diag_3_mask

            diag_mask = torch.eye(attnresln_n.shape[2])
            rest_mask = (torch.ones([attnresln_n.shape[2], attnresln_n.shape[2]], dtype=torch.long) - diag_mask - neigh_diag_mask) > 0
            neigh_diag_mask = neigh_diag_mask > 0

            norms_matrix = attnresln_n

            attnresln_n_diag = norms_matrix[:,diag_mask>0]
            attnresln_n_rest = norms_matrix[:,diag_mask<1]

            attnresln_n_neigh_rest = norms_matrix[:, rest_mask]
            attnresln_n_neigh = norms_matrix[:, neigh_diag_mask]

            neigh_rest_ratio = attnresln_n_neigh.mean() / (attnresln_n_neigh_rest.mean() + attnresln_n_neigh.mean())
            diag_rest_ratio = attnresln_n_rest.mean() / (attnresln_n_rest.mean() + attnresln_n_diag.mean())

            neigh_rest_ratio_per_layer.append(neigh_rest_ratio.detach().cpu().numpy())
            diag_rest_ratio_per_layer.append(diag_rest_ratio.detach().cpu().numpy())

            print("Our neighboring ratio is {}".format(neigh_rest_ratio))
            print("Our diag ratio is {}/{}+{} = {}".format(attnresln_n_rest.mean(),attnresln_n_rest.mean(), attnresln_n_diag.mean(), diag_rest_ratio))
            plt.figure()
            # df = pd.DataFrame(attnresln_n[idx].detach().cpu().numpy(), columns=labels , index=labels)
            df = pd.DataFrame(norms_matrix[idx].detach().cpu().numpy())
            sns.heatmap(df, cmap="Blues", square=True, annot=True, fmt='.1g', annot_kws={"fontsize": 4})
            plt.title("Context Ratio layer {}".format(layer))
            plt.show()

        t = np.concatenate(
            [np.expand_dims(np.array(neigh_rest_ratio_per_layer),axis=0),
            np.expand_dims(np.array(diag_rest_ratio_per_layer),axis=0)],
            axis=0)
        plt.figure()
        layer_columns = ["Layer {}".format(i) for i in range(num_layers)]
        df = pd.DataFrame(t , columns=layer_columns , index=["Neighbor R","Context R"])
        sns.heatmap(df, cmap="Blues", square=True, annot=True, fmt='.3g', annot_kws={"fontsize": 8})
        plt.title("Ratios Per Layer")
        plt.show()

    return 0
def get_attention_weights_merged_norm(model, device, batch, seq_l, data_loader, description):

    # device = "cuda:{}".format(config.gpu_device[0])
    # device = "cpu"
    forward_hook_manager = ForwardHookManager(device)
    forward_hook_manager.add_hook(model, 'enc_0.inner_tf', requires_output=True)
    forward_hook_manager.add_hook(model, 'enc_0.outer_tf', requires_output=True)
    num_layers = 4
    for i in range(num_layers):
        forward_hook_manager.add_hook(model, 'enc_0.outer_tf.outer_tf.layers.{}.norm_calc'.format(i), requires_input=False, requires_output=True)
        forward_hook_manager.add_hook(model, 'enc_0.inner_tf.inner_tf.layers.{}.norm_calc'.format(i), requires_input=False, requires_output=True)

    # forward_hook_manager.add_hook(model, 'enc_0.inner_tf_mod1', requires_output=True)
    # forward_hook_manager.add_hook(model, 'enc_0.outer_tf_mod1', requires_output=True)

    model.eval()
    pbar = tqdm(enumerate(data_loader), desc=description, leave=False)
    for batch_idx, (data, target, init, ids) in pbar:
        print(batch_idx)
        views = [data[i].float().to(device) for i in range(len(data))]
        # views[0] = torch.cat([views[0][:16],views[0][16:]],dim=1)
        # views[1] = torch.cat([views[1][:16],views[1][16:]],dim=1)
        preds = get_predictions_time_series(model, views, init, extract_norm=True)
        break

    # batch, seq_l = 16, 42

    io_dict = forward_hook_manager.pop_io_dict()
    # print(io_dict.keys())
    # inner_weights_eeg = {}
    # inner_weights_eeg["layer_0"] = io_dict['enc_0.inner_tf_mod0_l0.inner_tf.layers.0.self_attn_my.scaled_dotproduct_attention']['output'][1].detach().cpu().numpy()
    # inner_weights_eeg["layer_1"] = io_dict['enc_0.inner_tf_mod0_l1.inner_tf.layers.0.self_attn_my.scaled_dotproduct_attention']['output'][1].detach().cpu().numpy()
    # inner_weights_eeg["layer_2"] = io_dict['enc_0.inner_tf_mod0_l2.inner_tf.layers.0.self_attn_my.scaled_dotproduct_attention']['output'][1].detach().cpu().numpy()
    # inner_weights_eeg["layer_3"] = io_dict['enc_0.inner_tf_mod0_l3.inner_tf.layers.0.self_attn_my.scaled_dotproduct_attention']['output'][1].detach().cpu().numpy()
    #
    # inner_weights_eog = {}
    # inner_weights_eog["layer_0"] = io_dict['enc_0.inner_tf_mod1_l0.inner_tf.layers.0.self_attn_my.scaled_dotproduct_attention']['output'][1].detach().cpu().numpy()
    # inner_weights_eog["layer_1"] = io_dict['enc_0.inner_tf_mod1_l1.inner_tf.layers.0.self_attn_my.scaled_dotproduct_attention']['output'][1].detach().cpu().numpy()
    # inner_weights_eog["layer_2"] = io_dict['enc_0.inner_tf_mod1_l2.inner_tf.layers.0.self_attn_my.scaled_dotproduct_attention']['output'][1].detach().cpu().numpy()
    # inner_weights_eog["layer_3"] = io_dict['enc_0.inner_tf_mod1_l3.inner_tf.layers.0.self_attn_my.scaled_dotproduct_attention']['output'][1].detach().cpu().numpy()

    outer_norms = {"norm":{}}
    inner_norms = {"norm":{}}
    for i in range(num_layers):
        outer_norms["norm"]["layer_{}".format(i)] = io_dict['enc_0.outer_tf.outer_tf.layers.{}.norm_calc'.format(i)]["output"]
        inner_norms["norm"]["layer_{}".format(i)] = io_dict['enc_0.inner_tf.inner_tf.layers.{}.norm_calc'.format(i)]["output"]

    # target = target.detach().cpu().numpy()
    target = target.detach().argmax(dim=-1).cpu().numpy()
    print(target)

    idx = 1
    # colors_b = ["c" for i in range(29)]
    # colors_c = ["y" for i in range(29)]
    # colors = ["k"] + colors_b + colors_c
    # from matplotlib.lines import Line2D
    # for weights in [inner_norms]:
    #     plt.figure()
    #     plt.subplot(221)
    #     i=0
    #     plt.bar(np.arange(59),weights["norm"]["layer_{}".format(i)][1].mean(dim=0)[0].detach().cpu().numpy(), color=colors)
    #     plt.title("CLS layer {}".format(i))
    #     plt.axis("off")
    #     plt.subplot(222)
    #     i=1
    #     plt.bar(np.arange(59),weights["norm"]["layer_{}".format(i)][1].mean(dim=0)[0].detach().cpu().numpy(), color=colors)
    #     plt.title("CLS layer {}".format(i))
    #     plt.axis("off")
    #     plt.subplot(223)
    #     i=2
    #     plt.bar(np.arange(59),weights["norm"]["layer_{}".format(i)][1].mean(dim=0)[0].detach().cpu().numpy(), color=colors)
    #     plt.title("CLS layer {}".format(i))
    #     plt.axis("off")
    #
    #     plt.subplot(224)
    #     i=3
    #     plt.bar(np.arange(59),weights["norm"]["layer_{}".format(i)][1].mean(dim=0)[0].detach().cpu().numpy(), color=colors)
    #     plt.title("CLS layer {}".format(i))
    #     plt.axis("off")
    #
    #     legend_elements = [ Line2D([0], [0], marker='o', color='c', label='EEG',
    #                               markerfacecolor='c', markersize=10),
    #                         Line2D([0], [0], marker='o', color='y', label='EOG',
    #                                markerfacecolor='y', markersize=10),
    #                         Line2D([0], [0], marker='o', color='k', label='CLS',
    #                                markerfacecolor='k', markersize=10)
    #                         ]
    #     plt.legend(handles=legend_elements, loc="lower center")
    #
    #     plt.show()
            # df = pd.DataFrame(weights["norm"]["layer_{}".format(i)][3][idx].detach().cpu().numpy())
            # sns.heatmap(df, cmap="Blues", square=True, annot=True, fmt='.1g', annot_kws={"fontsize": 4})
            # plt.title("Context Ratio layer {}".format(i))
            # plt.show()

    # print(target[idx])
    for weights in [inner_norms]:

        # l = {0: "W", 1: "N1", 2: "N2", 3: "N3", 4: "R"}
        # labels = [l[target[idx][i % 21]] for j in range(2) for i in range(len(target[idx]))]

        neigh_rest_ratio_per_layer = []
        diag_rest_ratio_per_layer = []
        crossmodal_rest_ratio_per_layer_mod0to1 = []
        crossmodal_rest_ratio_per_layer_mod1to0 = []
        run_crossmodal = False
        for layer in range(num_layers):
            head_attn_n, attn_n, attnres_n, attnresln_n, attn_n_ratio, attnres_n_ratio, attnresln_n_ratio = weights["norm"]["layer_{}".format(i)]

            #leave out cls
            attnresln_n = attnresln_n[:,1:,1:]

            neigh_diag = torch.ones(attnresln_n.shape[2] - 1, dtype=torch.long)
            neigh_diag_mask = torch.diagflat(neigh_diag, offset=1) + torch.diagflat(neigh_diag, offset=-1)
            neigh_diag_2_mask = torch.diagflat(torch.ones(attnresln_n.shape[2] - 2, dtype=torch.long), offset=2) + torch.diagflat(torch.ones(attnresln_n.shape[2] - 2, dtype=torch.long), offset=-2)
            neigh_diag_mask += neigh_diag_2_mask

            if run_crossmodal:
                cross_modal_mask_mod0to1 = torch.zeros([attnresln_n.shape[2], attnresln_n.shape[2]])
                cross_modal_mask_mod1to0 = torch.zeros([attnresln_n.shape[2], attnresln_n.shape[2]])
                cross_modal_mask_mod1to0[int(cross_modal_mask_mod1to0.shape[0] / 2):, :int(cross_modal_mask_mod1to0.shape[0] / 2)] = torch.ones([int(attnresln_n.shape[2]/2), int(attnresln_n.shape[2]/2)])
                cross_modal_mask_mod0to1[:int(cross_modal_mask_mod0to1.shape[0] / 2), int(cross_modal_mask_mod0to1.shape[0] / 2):] = torch.ones([int(attnresln_n.shape[2]/2), int(attnresln_n.shape[2]/2)])


            diag_mask = torch.eye(attnresln_n.shape[2])
            rest_mask = (torch.ones([attnresln_n.shape[2], attnresln_n.shape[2]], dtype=torch.long) - diag_mask - neigh_diag_mask) > 0
            neigh_diag_mask = neigh_diag_mask > 0

            attnresln_n_diag = attnresln_n[:,diag_mask>0]
            attnresln_n_rest = attnresln_n[:,diag_mask<1]

            attnresln_n_neigh_rest = attnresln_n[:, rest_mask]
            attnresln_n_neigh = attnresln_n[:, neigh_diag_mask]

            if run_crossmodal:
                attnresln_n_crossmodal_rest_mod0to1 = attnresln_n[:, cross_modal_mask_mod0to1>0]
                attnresln_n_crossmodal_mod0to1 = attnresln_n[:, cross_modal_mask_mod0to1<1]

                attnresln_n_crossmodal_rest_mod1to0  = attnresln_n[:, cross_modal_mask_mod1to0>0]
                attnresln_n_crossmodal_mod1to0  = attnresln_n[:, cross_modal_mask_mod1to0<1]

            neigh_rest_ratio = attnresln_n_neigh_rest.mean() / (attnresln_n_neigh_rest.mean() + attnresln_n_neigh.mean())
            diag_rest_ratio = attnresln_n_rest.mean() / (attnresln_n_rest.mean() + attnresln_n_diag.mean())

            if run_crossmodal:
                crossmodal_rest_ratio_mod0to1 = attnresln_n_crossmodal_rest_mod0to1.mean() / (attnresln_n_crossmodal_rest_mod0to1.mean() + attnresln_n_crossmodal_mod0to1.mean())
                crossmodal_rest_ratio_mod1to0 = attnresln_n_crossmodal_rest_mod1to0.mean() / (attnresln_n_crossmodal_rest_mod1to0.mean() + attnresln_n_crossmodal_mod1to0.mean())

            neigh_rest_ratio_per_layer.append(neigh_rest_ratio.detach().cpu().numpy())
            diag_rest_ratio_per_layer.append(diag_rest_ratio.detach().cpu().numpy())
            if run_crossmodal:
                crossmodal_rest_ratio_per_layer_mod0to1.append(crossmodal_rest_ratio_mod0to1.detach().cpu().numpy())
                crossmodal_rest_ratio_per_layer_mod1to0.append(crossmodal_rest_ratio_mod1to0.detach().cpu().numpy())

            print("Our neighboring ratio is {}".format(neigh_rest_ratio))
            print("Our diag ratio is {}".format(diag_rest_ratio))
            if run_crossmodal:
                print("Our cross modal context ratio from eeg to eog is {}".format(crossmodal_rest_ratio_mod0to1))
                print("Our cross modal context ratio from eog to eeg is {}".format(crossmodal_rest_ratio_mod1to0))

            plt.figure()
            df = pd.DataFrame(attnresln_n[idx].detach().cpu().numpy())
            # df = pd.DataFrame(attnresln_n[idx].detach().cpu().numpy(), columns=labels , index=labels)
            sns.heatmap(df, cmap="Blues", square=True, annot=False, fmt='.1g', annot_kws={"fontsize": 4})
            plt.title("Context Ratio layer {}".format(layer))
            plt.show()

        t = np.concatenate(
            [
            # np.expand_dims(np.array(neigh_rest_ratio_per_layer),axis=0),
            # np.expand_dims(np.array(diag_rest_ratio_per_layer),axis=0),
            np.expand_dims(np.array(crossmodal_rest_ratio_per_layer_mod0to1),axis=0),
            np.expand_dims(np.array(crossmodal_rest_ratio_per_layer_mod1to0),axis=0)],
            axis=0)
        plt.figure()
        layer_columns = ["Layer {}".format(i) for i in range(num_layers)]
        df = pd.DataFrame(t , columns=layer_columns , index=["EEG->EOG", "EOG->EEG"])
        sns.heatmap(df, cmap="Blues", square=True, annot=True, fmt='.3g', annot_kws={"fontsize": 8})
        plt.title("Ratios Per Layer")
        plt.show()

def get_learnable_pos(model, device, batch, seq_l, data_loader, description):
    inner_pos_eeg = model.module.enc_0.module.inner_positional_embedding_0.pos
    inner_pos_eog = model.module.enc_0.module.inner_positional_embedding_1.pos

    outer_pos_eeg = model.module.enc_0.module.outer_positional_embedding_0.pos
    outer_pos_eog = model.module.enc_0.module.outer_positional_embedding_1.pos

    plt.figure()
    plt.subplot(221)
    plt.imshow(inner_pos_eeg.squeeze().cpu().detach().numpy())
    plt.subplot(222)
    plt.imshow(inner_pos_eog.squeeze().cpu().detach().numpy())
    plt.subplot(223)
    plt.imshow(outer_pos_eeg.squeeze().cpu().detach().numpy())
    plt.subplot(224)
    plt.imshow(outer_pos_eog.squeeze().cpu().detach().numpy())
    plt.show()

def validate(config, model,data_loader, description, device="cpu"):
    model.eval()
    with torch.no_grad():
        tts, preds, inits = [], [], []
        pbar = tqdm(enumerate(data_loader), desc=description, leave=False)
        for batch_idx, (data, target, init, _) in pbar:

            views = [data[i].float().to(device) for i in range(len(data))]
            if "softlabels" in config and config.softlabels:
                target = target.to(device).flatten(start_dim=0, end_dim=1).float()
            else:
                target = target.to(device).flatten(start_dim=0, end_dim=1).long()
                if len(target.shape) > 1:
                    target = target.argmax(dim=1)

            if "random_shuffle_data" in config and config.random_shuffle_data:
                perms = torch.randperm(views[0].shape[1])
                views = [view[:, perms] for view in views]
                target = einops.rearrange(target, "(batch seq) -> batch seq", batch=views[0].shape[0],
                                          seq=views[0].shape[1])[:, perms].flatten()
                init = init[:, perms]

            if "random_shuffle_data_batch" in config and config.random_shuffle_data_batch:
                perms = torch.randperm(views[0].shape[0] * views[0].shape[1])
                d_shape = views[0].shape
                views = [einops.rearrange(einops.rearrange(view, "batch seq b c d -> (batch seq) b c d")[perms],
                                          "(batch seq) b c d -> batch seq b c d", batch=d_shape[0],
                                          seq=d_shape[1])
                         for view in views]
                target = target.flatten()[perms]
                init = einops.rearrange(einops.rearrange(init, "batch seq -> (batch seq)")[perms],
                                        "(batch seq) -> batch seq", batch=d_shape[0], seq=d_shape[1])

            pred = model(views)
            # pred = get_predictions_time_series(model, views, init)

            tts.append(target)
            preds.append(pred["preds"]["combined"])
            # preds.append(pred)
            inits.append(init.flatten())
            pbar.set_description("{} batch {}/{}".format(description, int(batch_idx), int(len(data_loader))))
            pbar.refresh()

        if "softlabels" in config and config.softlabels:
            tts = torch.cat(tts).argmax(dim=1).cpu().numpy()
        else:
            tts = torch.cat(tts).cpu().numpy()

        preds = torch.cat(preds).cpu().numpy()

    multiclass = False
    if preds.shape[1] > 2:
        multiclass = True

    # entropy_pred = entropy(preds, axis=1)
    # class_pred = preds.argmax(axis=-1)
    # entropy_correct_class = entropy_pred[class_pred==tts].mean()
    # entropy_wrong_class = entropy_pred[class_pred!=tts].mean()
    #
    # print("{} entropy for correct class class prediction is {} and for wrong class predictions {}".format(description, entropy_correct_class, entropy_wrong_class))
    future_preds = copy.deepcopy(preds)
    preds = preds.argmax(axis=1)
    test_acc = np.equal(tts, preds).sum() / len(tts)
    test_f1 = f1_score(preds, tts) if not multiclass else f1_score(preds, tts, average="macro")
    test_perclass_f1 = f1_score(preds, tts) if not multiclass else f1_score(preds, tts, average=None)
    test_k = cohen_kappa_score(tts, preds)
    test_auc = roc_auc_score(tts, preds) if not multiclass else 0
    test_conf = confusion_matrix(tts, preds)
    tp, fp, tn, fn = perf_measure(tts, preds)
    test_spec = tn / (tn + fp) if (tn + fp)!=0 else 0
    test_sens = tp / (tp + fn) if (tp + fn)!=0 else 0
    print("{0} accuracy: {1:.2f}% f1 :{2:.4f}, k :{3:.4f}, sens:{4:.4f}, spec:{5:.4f}, f1_per_class :{6:40}".format(
            description,
            test_acc * 100,
            test_f1,
            test_k, test_spec, test_sens,
            "{}".format(list(test_perclass_f1))))
    norm_n_plot_confusion_matrix(test_conf, description)

    return test_acc, test_f1, test_k, test_auc, test_conf, test_perclass_f1, test_spec, test_sens
def validate_borders(config, model,data_loader, description):
    model.eval()
    with torch.no_grad():
        tts, preds, inits, ids, seq_nums = [], [], [], [], []
        pbar = tqdm(enumerate(data_loader), desc=description, leave=False)
        for batch_idx, (data, target, init, id) in pbar:

            views = [data[i].float().to(device) for i in range(len(data))]
            if "softlabels" in config and config.softlabels:
                target = target.to(device).flatten(start_dim=0, end_dim=1).float()
            else:
                target = target.to(device).flatten(start_dim=0, end_dim=1).long()
                if len(target.shape) > 1:
                    target = target.argmax(dim=1)

            if "random_shuffle_data" in config and config.random_shuffle_data:
                perms = torch.randperm(views[0].shape[1])
                views = [view[:, perms] for view in views]
                target = einops.rearrange(target, "(batch seq) -> batch seq", batch=views[0].shape[0],
                                          seq=views[0].shape[1])[:, perms].flatten()
                init = init[:, perms]

            if "random_shuffle_data_batch" in config and config.random_shuffle_data_batch:
                perms = torch.randperm(views[0].shape[0] * views[0].shape[1])
                d_shape = views[0].shape
                views = [einops.rearrange(einops.rearrange(view, "batch seq b c d -> (batch seq) b c d")[perms],
                                          "(batch seq) b c d -> batch seq b c d", batch=d_shape[0],
                                          seq=d_shape[1])
                         for view in views]
                target = target.flatten()[perms]
                init = einops.rearrange(einops.rearrange(init, "batch seq -> (batch seq)")[perms],
                                        "(batch seq) -> batch seq", batch=d_shape[0], seq=d_shape[1])

            pred = model(views)["preds"]["combined"]
            # pred = get_predictions_time_series(model, views, init)
            tts.append(target)
            seq_nums.append(np.expand_dims(np.arange(init.shape[1]), axis=0).repeat(init.shape[0], axis=0).flatten())
            ids.append(id.flatten())
            preds.append(pred)
            inits.append(init.flatten())
            pbar.set_description("{} batch {}/{}".format(description, int(batch_idx), int(len(data_loader))))
            pbar.refresh()

        if "softlabels" in config and config.softlabels:
            tts = torch.cat(tts).argmax(dim=1).cpu().numpy()
        else:
            tts = torch.cat(tts).cpu().numpy()
        preds = torch.cat(preds).cpu().numpy()

    multiclass = False
    if preds.shape[1] > 2:
        multiclass = True

    border_points = [4]
    for i in range(1, len(tts) - 1):
        if tts[i - 1] != tts[i] or tts[i + 1] != tts[i]:
            border_points.append(0)
        else:
            border_points.append(4)
    border_points.append(4)
    for i in range(len(border_points)):
        if border_points[i] == 0:
            if border_points[i + 1] != 0:
                border_points[i + 1] = 1
            if border_points[i - 1] != 0:
                border_points[i - 1] = 1
            if border_points[i + 2] != 0 and border_points[i + 2] != 1:
                border_points[i + 2] = 2
            if border_points[i - 2] != 0 and border_points[i - 2] != 1:
                border_points[i - 2] = 2
            if border_points[i + 3] != 0 and border_points[i + 3] != 1 and border_points[i + 3] != 2:
                border_points[i + 3] = 3
            if border_points[i - 3] != 0 and border_points[i - 3] != 1 and border_points[i - 3] != 2:
                border_points[i - 3] = 3
    border_points = np.array(border_points)
    class_pred = preds.argmax(axis=-1)
    _, border_counts_p = np.unique(border_points, return_counts=True)
    _, border_counts = np.unique(border_points[~(class_pred==tts)], return_counts=True)
    border_counts = border_counts/border_counts.sum()
    border_colors = [(0,0,0.8), (0,0,0.6), (0,0,0.4), (0,0,0.2), "gold"]
    plt.bar(np.array([0,1,2,3,4]), border_counts, color=border_colors)
    plt.xticks(np.array([0,1,2,3,4]), labels=['Border', '1', '2', '3', 'Rest'])
    plt.ylabel("Percentage of Mistakes")
    plt.title("Border %mistakes per set on {} SHHS".format(description))
    plt.show()

    _, border_counts= np.unique(border_points, return_counts=True)
    print("We have in total {}".format(border_counts.sum()))
    print(border_counts)
    border_counts = border_counts/border_counts.sum()
    border_colors = [(0,0,0.8), (0,0,0.6), (0,0,0.4), (0,0,0.2), "gold"]
    plt.bar(np.array([0,1,2,3,4]), border_counts, color=border_colors)
    plt.xticks(np.array([0,1,2,3,4]), labels=['Border', '1', '2', '3', 'Rest'])
    plt.ylabel("%")
    plt.title("Border distribution on {} SHHS".format(description))
    plt.show()

    seq_nums = np.concatenate(seq_nums).flatten()
    _, seq_counts= np.unique(seq_nums[~(class_pred==tts)], return_counts=True)
    seq_counts = seq_counts/seq_counts.sum()
    x = np.arange(21)
    plt.bar(x, seq_counts, color=border_colors)
    plt.ylabel("Percentage of Mistakes")
    plt.title("Seq Num mistakes on {} SHHS".format(description))
    plt.show()

    entropy_pred = entropy(preds, axis=1)
    entropy_correct_class = entropy_pred[class_pred==tts].mean()
    entropy_wrong_class = entropy_pred[class_pred!=tts].mean()

    print("{} entropy for correct class class prediction is {} and for wrong class predictions {}".format(description, entropy_correct_class, entropy_wrong_class))

    preds = preds.argmax(axis=1)
    test_acc = np.equal(tts, preds).sum() / len(tts)
    test_f1 = f1_score(preds, tts) if not multiclass else f1_score(preds, tts, average="macro")
    test_perclass_f1 = f1_score(preds, tts) if not multiclass else f1_score(preds, tts, average=None)
    test_k = cohen_kappa_score(tts, preds)
    test_auc = roc_auc_score(tts, preds) if not multiclass else 0
    test_conf = confusion_matrix(tts, preds)
    tp, fp, tn, fn = perf_measure(tts, preds)
    test_spec = tn / (tn + fp) if (tn + fp)!=0 else 0
    test_sens = tp / (tp + fn) if (tp + fn)!=0 else 0
    print("{0} accuracy: {1:.2f}% f1 :{2:.4f}, k :{3:.4f}, sens:{4:.4f}, spec:{5:.4f}, f1_per_class :{6:40}".format(
            description,
            test_acc * 100,
            test_f1,
            test_k, test_spec, test_sens,
            "{}".format(list(test_perclass_f1))))
    norm_n_plot_confusion_matrix(test_conf, description)

    preds = preds[entropy_pred<0.03]
    tts = tts[entropy_pred<0.03]
    test_acc = np.equal(tts, preds).sum() / len(tts)
    test_f1 = f1_score(preds, tts) if not multiclass else f1_score(preds, tts, average="macro")
    test_perclass_f1 = f1_score(preds, tts) if not multiclass else f1_score(preds, tts, average=None)
    test_k = cohen_kappa_score(tts, preds)
    test_auc = roc_auc_score(tts, preds) if not multiclass else 0
    test_conf = confusion_matrix(tts, preds)
    tp, fp, tn, fn = perf_measure(tts, preds)
    test_spec = tn / (tn + fp) if (tn + fp)!=0 else 0
    test_sens = tp / (tp + fn) if (tp + fn)!=0 else 0
    print("{0} accuracy: {1:.2f}% f1 :{2:.4f}, k :{3:.4f}, sens:{4:.4f}, spec:{5:.4f}, f1_per_class :{6:40}".format(
            description,
            test_acc * 100,
            test_f1,
            test_k, test_spec, test_sens,
            "{}".format(list(test_perclass_f1))))

    border_points = np.array(border_points)[entropy_pred<0.03]
    class_pred = preds
    _, border_counts_p = np.unique(border_points, return_counts=True)
    _, border_counts = np.unique(border_points[~(class_pred==tts)], return_counts=True)
    border_counts = border_counts/border_counts.sum()
    border_colors = [(0,0,0.8), (0,0,0.6), (0,0,0.4), (0,0,0.2), "gold"]
    plt.bar(np.array([0,1,2,3,4]), border_counts, color=border_colors)
    plt.xticks(np.array([0,1,2,3,4]), labels=['Border', '1', '2', '3', 'Rest'])
    plt.ylabel("Percentage of Mistakes")
    plt.title("Border %mistakes per set on {} SHHS".format(description))
    plt.show()

    norm_n_plot_confusion_matrix(test_conf, description)

    return test_acc, test_f1, test_k, test_auc, test_conf, test_perclass_f1, test_spec, test_sens

def validate_ensembles(models,data_loader, description):
    ensemble_preds = []
    for model in models:
        model.eval()
        with torch.no_grad():
            tts, preds, inits = [], [], []
            pbar = tqdm(enumerate(data_loader), desc=description, leave=False)
            for batch_idx, (data, target, init, _) in pbar:
                views = [data[i].float().to(device) for i in range(len(data))]
                label = target.to(device).flatten()
                pred = get_predictions_time_series(model, views, init)
                tts.append(label)
                preds.append(pred)
                inits.append(init.flatten())
                pbar.set_description("{} batch {}/{}".format(description, int(batch_idx), int(len(data_loader))))
                pbar.refresh()

            tts = torch.cat(tts).cpu().numpy()
            preds = torch.cat(preds).cpu().numpy()
        ensemble_preds.append(preds)

    for i in range(len(models)):
        print("--------------------------------------------")
        print("Model {} has ".format(i))
        preds = np.array(ensemble_preds[i])

        multiclass = False
        if preds.shape[1] > 2:
            multiclass = True

        entropy_pred = entropy(preds, axis=1)
        class_pred = preds.argmax(axis=1)
        entropy_correct_class = entropy_pred[class_pred==tts].mean()
        entropy_wrong_class = entropy_pred[class_pred!=tts].mean()

        print("{} entropy for correct class class prediction is {} and for wrong class predictions {}".format(description, entropy_correct_class, entropy_wrong_class))

        preds = preds.argmax(axis=1)
        test_acc = np.equal(tts, preds).sum() / len(tts)
        test_f1 = f1_score(preds, tts) if not multiclass else f1_score(preds, tts, average="macro")
        test_perclass_f1 = f1_score(preds, tts) if not multiclass else f1_score(preds, tts, average=None)
        test_k = cohen_kappa_score(tts, preds)
        test_auc = roc_auc_score(tts, preds) if not multiclass else 0
        test_conf = confusion_matrix(tts, preds)
        tp, fp, tn, fn = perf_measure(tts, preds)
        test_spec = tn / (tn + fp) if (tn + fp)!=0 else 0
        test_sens = tp / (tp + fn) if (tp + fn)!=0 else 0
        print("{0} accuracy: {1:.2f}% f1 :{2:.4f}, k :{3:.4f}, sens:{4:.4f}, spec:{5:.4f}, f1_per_class :{6:40}".format(
                description,
                test_acc * 100,
                test_f1,
                test_k, test_spec, test_sens,
                "{}".format(list(test_perclass_f1))))
        norm_n_plot_confusion_matrix(test_conf, description)


    print("--------------------------------------------")
    print("Ensemble of Models")

    preds = np.array(ensemble_preds).mean(axis=0)

    multiclass = False
    if preds.shape[1] > 2:
        multiclass = True

    entropy_pred = entropy(preds, axis=1)
    class_pred = preds.argmax(axis=1)
    entropy_correct_class = entropy_pred[class_pred == tts].mean()
    entropy_wrong_class = entropy_pred[class_pred != tts].mean()

    print("{} entropy for correct class class prediction is {} and for wrong class predictions {}".format(description,
                                                                                                          entropy_correct_class,
                                                                                                          entropy_wrong_class))

    preds = preds.argmax(axis=1)
    test_acc = np.equal(tts, preds).sum() / len(tts)
    test_f1 = f1_score(preds, tts) if not multiclass else f1_score(preds, tts, average="macro")
    test_perclass_f1 = f1_score(preds, tts) if not multiclass else f1_score(preds, tts, average=None)
    test_k = cohen_kappa_score(tts, preds)
    test_auc = roc_auc_score(tts, preds) if not multiclass else 0
    test_conf = confusion_matrix(tts, preds)
    tp, fp, tn, fn = perf_measure(tts, preds)
    test_spec = tn / (tn + fp) if (tn + fp) != 0 else 0
    test_sens = tp / (tp + fn) if (tp + fn) != 0 else 0
    print("{0} accuracy: {1:.2f}% f1 :{2:.4f}, k :{3:.4f}, sens:{4:.4f}, spec:{5:.4f}, f1_per_class :{6:40}".format(
        description,
        test_acc * 100,
        test_f1,
        test_k, test_spec, test_sens,
        "{}".format(list(test_perclass_f1))))
    norm_n_plot_confusion_matrix(test_conf, description)

    return test_acc, test_f1, test_k, test_auc, test_conf, test_perclass_f1, test_spec, test_sens

def validate_specific_patient(data_loader, patient_num, model, model_eeg, model_eog, device, plot_hypnogram_flag=True, return_matches=False, plot_matches=False, plot_entropy=True):
    model.eval()
    model_eeg.eval()
    model_eog.eval()
    this_data_loader = copy.deepcopy(data_loader)
    # metrics_mean = this_data_loader.dataset.mean
    # metrics_std = this_data_loader.dataset.std
    this_data_loader.dataset.choose_specific_patient(patient_num)
    # this_data_loader.dataset.config.statistics["print"] = True
    # this_data_loader.dataset.print_statistics_per_patient()

    with torch.no_grad():
        try:
            tts, preds, matches, inits, views_eeg, views_eog, inter_eeg, inter_eog = [], [], [], [], [], [], [], []
            views_eeg_time, views_eog_time = [], []
            preds_eeg, preds_eog = [], []
            for batch_idx, (data, target, init, _) in enumerate(this_data_loader):
                views = [data[i].float().to(device) for i in range(len(data))]
                label = target.to(device)
                if return_matches:
                    output = model(views, return_matches=return_matches, return_inter_reps=True)
                    matches.append(output["matches"])
                else:
                    output = model(views)
                output_eeg = model_eeg([views[0]])
                output_eog = model_eog([views[1]])
                # pred = get_predictions_time_series(model, views, init)
                tts.append(label)
                # preds.append(output["preds"]["combined"])
                preds.append(output)
                preds_eeg.append(output_eeg)
                preds_eog.append(output_eog)
                # preds.append(pred)
                views_eeg.append(views[0])
                views_eog.append(views[1])
                views_eeg_time.append(views[2])
                views_eog_time.append(views[3])
                # if type(output)==dict and "inter_reps" in output:
                #     inter_eeg.append(output["inter_reps"][0])
                #     inter_eog.append(output["inter_reps"][1])
                inits.append(init.flatten())

            tts = torch.cat(tts).cpu().numpy()
            preds = torch.cat(preds).cpu().numpy()
            preds_eeg = torch.cat(preds_eeg).cpu().numpy()
            preds_eog = torch.cat(preds_eog).cpu().numpy()


            views_eeg = torch.cat(views_eeg).cpu().squeeze().flatten(start_dim=0, end_dim=1)
            views_eeg = einops.rearrange(views_eeg, "a b c -> (a c) b").numpy()
            views_eog = torch.cat(views_eog).cpu().squeeze().flatten(start_dim=0, end_dim=1)
            views_eog = einops.rearrange(views_eog, "a b c -> (a c) b").numpy()

            views_eeg_time = torch.cat(views_eeg_time).cpu().squeeze().flatten(start_dim=0, end_dim=1)
            views_eog_time = torch.cat(views_eog_time).cpu().squeeze().flatten(start_dim=0, end_dim=1)

            # inter_eeg = torch.cat(inter_eeg).cpu().squeeze().flatten(start_dim=0, end_dim=1)
            # inter_eeg = einops.rearrange(inter_eeg, "a b c -> (a c) b").numpy()
            # inter_eog = torch.cat(inter_eog).cpu().squeeze().flatten(start_dim=0, end_dim=1)
            # inter_eog = einops.rearrange(inter_eog, "a b c -> (a c) b").numpy()

            tts_unfolded = torch.from_numpy(tts).flatten(start_dim=0, end_dim=1).unfold(0, 40, 40).numpy()
            preds_unfolded = torch.from_numpy(preds).argmax(dim=-1).unfold(0, 40, 40).numpy()
            preds_eeg_unfolded = torch.from_numpy(preds_eeg).argmax(dim=-1).unfold(0, 40, 40).numpy()
            preds_eog_unfolded = torch.from_numpy(preds_eog).argmax(dim=-1).unfold(0, 40, 40).numpy()

            # kappa_window = np.array([cohen_kappa_score(tts_unfolded[i], preds_unfolded[i]) for i in range(tts_unfolded.shape[0])])
            # kappa_window = np.array([f1_score(preds_unfolded[i], tts_unfolded[i],  average="macro") for i in range(tts_unfolded.shape[0])])
            kappa_window = np.array([np.equal(tts_unfolded[i], preds_unfolded[i]).sum() / len(tts_unfolded[i]) for i in range(tts_unfolded.shape[0])])
            kappa_window[kappa_window != kappa_window] = 1
            for i in kappa_window: print("{:.3f}".format(i), end=" ")
            print()
            kappa_window = kappa_window.repeat(40)

            # kappa_window_eeg = np.array([f1_score(preds_eeg_unfolded[i], tts_unfolded[i],  average="macro") for i in range(tts_unfolded.shape[0])])
            kappa_window_eeg = np.array([np.equal(tts_unfolded[i], preds_eeg_unfolded[i]).sum() / len(tts_unfolded[i]) for i in range(tts_unfolded.shape[0])])
            kappa_window_eeg[kappa_window_eeg != kappa_window_eeg] = 1
            for i in kappa_window_eeg: print("{:.3f}".format(i), end=" ")
            print()
            kappa_window_eeg = kappa_window_eeg.repeat(40)

            # kappa_window_eog = np.array([f1_score(preds_eog_unfolded[i], tts_unfolded[i],  average="macro") for i in range(tts_unfolded.shape[0])])
            kappa_window_eog = np.array([np.equal(tts_unfolded[i], preds_eog_unfolded[i]).sum() / len(tts_unfolded[i]) for i in range(tts_unfolded.shape[0])])
            # kappa_window_eog = np.array([f1_score(preds_eog_unfolded[i], tts_unfolded[i],  average="macro") for i in range(tts_unfolded.shape[0])])
            kappa_window_eog[kappa_window_eog != kappa_window_eog] = 1
            for i in kappa_window_eog: print("{:.3f}".format(i), end=" ")
            print()
            kappa_window_eog = kappa_window_eog.repeat(40)

            # inter_eeg_mean_floor = torch.cat([window.mean().unsqueeze(dim=0) for window in inter_eeg], dim=0)
            # inter_eeg_std_floor = torch.cat([window.std().unsqueeze(dim=0) for window in inter_eeg], dim=0)
            # inter_eeg_std_mean = inter_eeg_std_floor.mean()
            # inter_eeg_std_std = inter_eeg_std_floor.std()
            #
            # inter_eeg_mean_mean = inter_eeg_mean_floor.mean()
            # inter_eeg_mean_std = inter_eeg_mean_floor.std()

            # inter_eeg_mean_floor = inter_eeg_mean_floor.unfold(0, 40, 40).mean(dim=1, keepdim=True).repeat(1, 40).flatten().unsqueeze(dim=1).repeat(1, 3000)
            # inter_eeg_std_floor = inter_eeg_std_floor.unfold(0, 40, 40).mean(dim=1, keepdim=True).repeat(1, 40).flatten().unsqueeze(dim=1).repeat(1, 3000)

            # inter_eog_mean_floor = torch.cat([window.mean().unsqueeze(dim=0) for window in inter_eog], dim=0)
            # inter_eog_std_floor = torch.cat([window.std().unsqueeze(dim=0) for window in inter_eog], dim=0)
            #
            # inter_eog_mean_floor = inter_eog_mean_floor.unfold(0, 40, 40).mean(dim=1, keepdim=True).repeat(1, 40).flatten().unsqueeze(dim=1).repeat(1, 3000)
            # inter_eog_std_floor = inter_eog_std_floor.unfold(0, 40, 40).mean(dim=1, keepdim=True).repeat(1, 40).flatten().unsqueeze(dim=1).repeat(1, 3000)

            # print(inter_eeg.shape)
            # inter_distance = torch.einsum("xy, xy -> x",inter_eeg,inter_eog)
            # inter_distance = (inter_distance - inter_distance.mean())/inter_distance.std()

            # inter_eeg = inter_eeg.numpy()
            # inter_eog = inter_eog.numpy()
            if return_matches:
                matches = torch.cat(matches).cpu()

            multiclass = False
            if preds.shape[1] > 2:
                multiclass = True

            entropy_pred = entropy(preds, axis=1)
            class_pred = preds.argmax(axis=1)
            # entropy_correct_class = entropy_pred[class_pred == tts].mean()
            # entropy_wrong_class = entropy_pred[class_pred != tts].mean()
            # total_entropy = entropy_pred.mean()
            total_entropy = 0

            # print("{} entropy for correct class class prediction is {} and for wrong class predictions {}".format(patient_num,
            #                                                                                                       entropy_correct_class,
            #                                                                                                       entropy_wrong_class))

            preds_for_loss = copy.deepcopy(preds)
            preds = preds.argmax(axis=1)
            if len(tts.shape)>2:
                tts = tts.argmax(axis=-1)
            tts = tts.flatten()

            test_acc = np.equal(tts, preds).sum() / len(tts)
            test_f1 = f1_score(preds, tts) if not multiclass else f1_score(preds, tts, average="macro")
            test_perclass_f1 = f1_score(preds, tts) if not multiclass else f1_score(preds, tts, average=None)
            test_k = cohen_kappa_score(tts, preds)
            test_auc = roc_auc_score(tts, preds) if not multiclass else 0
            test_conf = confusion_matrix(tts, preds)
            tp, fp, tn, fn = perf_measure(tts, preds)
            test_spec = tn / (tn + fp) if (tn + fp) != 0 else 0
            test_sens = tp / (tp + fn) if (tp + fn) != 0 else 0
            print("Merged Patient {} has acc: {}, f1: {}, k:{} and f1_per_class: {}".format(patient_num,
                                                                                     round(test_acc * 100, 1),
                                                                                     round(test_f1 * 100, 1),
                                                                                     round(test_k, 3),
                                                                                     np.round(test_perclass_f1 * 100,
                                                                                              1)))

            preds_eeg = preds_eeg.argmax(axis=1)
            if len(tts.shape)>2:
                tts = tts.argmax(axis=-1)
            tts = tts.flatten()

            test_acc = np.equal(tts, preds_eeg).sum() / len(tts)
            test_f1 = f1_score(preds_eeg, tts) if not multiclass else f1_score(preds_eeg, tts, average="macro")
            test_perclass_f1 = f1_score(preds_eeg, tts) if not multiclass else f1_score(preds_eeg, tts, average=None)
            test_k = cohen_kappa_score(tts, preds_eeg)
            test_auc = roc_auc_score(tts, preds_eeg) if not multiclass else 0
            test_conf = confusion_matrix(tts, preds_eeg)
            tp, fp, tn, fn = perf_measure(tts, preds_eeg)
            test_spec = tn / (tn + fp) if (tn + fp) != 0 else 0
            test_sens = tp / (tp + fn) if (tp + fn) != 0 else 0
            print("EEG Patient {} has acc: {}, f1: {}, k:{} and f1_per_class: {}".format(patient_num,
                                                                                     round(test_acc * 100, 1),
                                                                                     round(test_f1 * 100, 1),
                                                                                     round(test_k, 3),
                                                                                     np.round(test_perclass_f1 * 100,
                                                                                              1)))

            preds_eog = preds_eog.argmax(axis=1)
            if len(tts.shape)>2:
                tts = tts.argmax(axis=-1)
            tts = tts.flatten()

            test_acc = np.equal(tts, preds_eog).sum() / len(tts)
            test_f1 = f1_score(preds_eog, tts) if not multiclass else f1_score(preds_eog, tts, average="macro")
            test_perclass_f1 = f1_score(preds, tts) if not multiclass else f1_score(preds_eog, tts, average=None)
            test_k = cohen_kappa_score(tts, preds_eog)
            test_auc = roc_auc_score(tts, preds_eog) if not multiclass else 0
            test_conf = confusion_matrix(tts, preds_eog)
            tp, fp, tn, fn = perf_measure(tts, preds_eog)
            test_spec = tn / (tn + fp) if (tn + fp) != 0 else 0
            test_sens = tp / (tp + fn) if (tp + fn) != 0 else 0
            print("EOG Patient {} has acc: {}, f1: {}, k:{} and f1_per_class: {}".format(patient_num,
                                                                                     round(test_acc * 100, 1),
                                                                                     round(test_f1 * 100, 1),
                                                                                     round(test_k, 3),
                                                                                     np.round(test_perclass_f1 * 100,
                                                                                              1)))
            # norm_n_plot_confusion_matrix(test_conf, patient_num)

            if plot_hypnogram_flag or True:
            # if test_f1>0.85:

                pred_plus = copy.deepcopy(preds)
                pred_plus[pred_plus == 4] = 5
                pred_plus[pred_plus == 3] = 4
                pred_plus[pred_plus == 2] = 3
                pred_plus[pred_plus == 5] = 2

                pred_eeg_plus = copy.deepcopy(preds_eeg)
                pred_eeg_plus[pred_eeg_plus == 4] = 5
                pred_eeg_plus[pred_eeg_plus == 3] = 4
                pred_eeg_plus[pred_eeg_plus == 2] = 3
                pred_eeg_plus[pred_eeg_plus == 5] = 2

                pred_eog_plus = copy.deepcopy(preds_eog)
                pred_eog_plus[pred_eog_plus == 4] = 5
                pred_eog_plus[pred_eog_plus == 3] = 4
                pred_eog_plus[pred_eog_plus == 2] = 3
                pred_eog_plus[pred_eog_plus == 5] = 2


                target_plus = copy.deepcopy(tts)
                target_plus[target_plus == 4] = 5
                target_plus[target_plus == 3] = 4
                target_plus[target_plus == 2] = 3
                target_plus[target_plus == 5] = 2

                from_hours_to_plot = int(120*0)
                hours_to_plot = -1 #int(120*2.5)
                print("TIME",from_hours_to_plot, hours_to_plot)
                pred_plus = pred_plus[from_hours_to_plot:hours_to_plot]
                pred_eeg_plus = pred_eeg_plus[from_hours_to_plot:hours_to_plot]
                pred_eog_plus = pred_eog_plus[from_hours_to_plot:hours_to_plot]
                target_plus = target_plus[from_hours_to_plot:hours_to_plot]
                views_eeg_time = views_eeg_time[from_hours_to_plot:hours_to_plot]
                views_eog_time = views_eog_time[from_hours_to_plot:hours_to_plot]
                views_eeg = views_eeg[from_hours_to_plot:hours_to_plot]
                views_eog = views_eog[from_hours_to_plot:hours_to_plot]
                # match_loss = match_loss[:hours_to_plot]

                # target = target + 0.02

                non_matches = (pred_plus != target_plus).astype(int)
                non_matches_idx = non_matches.nonzero()[0]

                non_matches_eeg = (pred_eeg_plus != target_plus).astype(int)
                non_matches_idx_eeg = non_matches_eeg.nonzero()[0]

                non_matches_eog = (pred_eog_plus != target_plus).astype(int)
                non_matches_idx_eog = non_matches_eog.nonzero()[0]

                target_plus = target_plus + 0.02

                # print("Non matching indices are:")
                # print(non_matches_idx)
                hours = len(target_plus)

                # plt.figure()
                # plt.plot(pred_plus, label="Prediction")
                # plt.plot(target_plus, label="True Label")
                # plt.scatter(non_matches_idx, pred_plus[non_matches_idx], marker='*', edgecolors="r", label="Mistakes")
                # # plt.plot(non_matches_idx,"*")
                # plt.yticks([0, 1, 2, 3, 4], labels=["Wake", "N1", "REM", "N2", "N3"])
                # plt.xticks([i * 120 for i in range((hours // 120) + 1)],
                #            labels=["{}".format(i) for i in range((hours // 120) + 1)])
                # plt.legend()
                # plt.ylabel("Labels")
                # plt.xlabel("Hours")
                # plt.title("Patient {}".format(patient_num))
                # plt.show()

            if plot_matches or True:

                # match_label = torch.eye(n=500).unsqueeze(dim=0).repeat(500, 1, 1)[:matches.shape[0],:matches.shape[1],:matches.shape[2]].argmax(-1).flatten()
                # matches = matches.flatten(start_dim=0, end_dim=1)
                #
                # match_loss = nn.CrossEntropyLoss(reduction='none')(matches, match_label)
                ce_loss = nn.CrossEntropyLoss(reduction='none')(torch.from_numpy(preds_for_loss),torch.from_numpy(tts))
                # combined_loss = match_loss + ce_loss

                # match_loss = (match_loss - match_loss.mean())/match_loss.std()
                # ce_loss = (ce_loss - ce_loss.mean())/ce_loss.std()
                # combined_loss = (combined_loss - combined_loss.mean())/combined_loss.std()

                def hl_envelopes_idx(s, dmin=-1, dmax=1, split=False):
                    """
                    Input :
                    s: 1d-array, data signal from which to extract high and low envelopes
                    dmin, dmax: int, optional, size of chunks, use this if the size of the input signal is too big
                    split: bool, optional, if True, split the signal in half along its mean, might help to generate the envelope in some cases
                    Output :
                    lmin,lmax : high/low envelope idx of input signal s
                    """

                    # locals min
                    lmin = (np.diff(np.sign(np.diff(s))) > 0).nonzero()[0] + 1
                    # locals max
                    lmax = (np.diff(np.sign(np.diff(s))) < 0).nonzero()[0] + 1

                    if split:
                        # s_mid is zero if s centered around x-axis or more generally mean of signal
                        s_mid = np.mean(s)
                        # pre-sorting of locals min based on relative position with respect to s_mid
                        lmin = lmin[s[lmin] < s_mid]
                        # pre-sorting of local max based on relative position with respect to s_mid
                        lmax = lmax[s[lmax] > s_mid]

                    # global max of dmax-chunks of locals max
                    lmin = lmin[[i + np.argmin(s[lmin[i:i + dmin]]) for i in range(0, len(lmin), dmin)]]
                    # global min of dmin-chunks of locals min
                    lmax = lmax[[i + np.argmax(s[lmax[i:i + dmax]]) for i in range(0, len(lmax), dmax)]]

                    return lmin, lmax
                # high_idx_match, low_idx_match = hl_envelopes_idx(match_loss.numpy())
                # high_idx_inter_dist, low_idx_inter_dist = hl_envelopes_idx(inter_distance.numpy())
                high_idx_ce, low_idx_ce = hl_envelopes_idx(ce_loss.numpy())
                # high_idx_combined, low_idx_combined = hl_envelopes_idx(combined_loss.numpy())

                # x = np.linspace(0, len(match_loss) - 1, len(match_loss))

                # plt.figure()
                # plt.subplot(211)
                # plt.plot(pred_plus, label="Prediction")
                # plt.plot(target_plus, label="True Label")
                # plt.scatter(non_matches_idx, pred_plus[non_matches_idx], marker='*', edgecolors="r", label="Mistakes")
                # # plt.plot(non_matches_idx,"*")
                # plt.yticks([0, 1, 2, 3, 4], labels=["Wake", "N1", "REM", "N2", "N3"])
                # plt.xticks([i * 120 for i in range((hours // 120) + 1)],
                #            labels=["{}".format(i) for i in range((hours // 120) + 1)])
                # plt.legend()
                # plt.ylabel("Labels")
                # plt.xlabel("Hours")
                # plt.title("Patient {}".format(patient_num))
                # plt.subplot(212)
                # # plt.plot(match_loss.numpy(), label="Matches")
                # # plt.plot(entropy_pred, label="Entropy")
                # # plt.plot(low_idx, label="Match Loss")
                # plt.plot(x[low_idx], match_loss[low_idx], 'g', label='high')
                # plt.scatter(non_matches_idx, entropy_pred[non_matches_idx], marker='*', edgecolors="r", label="Mistakes")
                # # plt.scatter(non_matches_idx, match_loss[non_matches_idx], marker='*', edgecolors="r", label="Mistakes")
                # plt.xticks([i * 120 for i in range((hours // 120) + 1)],
                #            labels=["{}".format(i) for i in range((hours // 120) + 1)])
                # # plt.title("Patient {}".format(patient_num))
                # # plt.ylabel("CE loss Envelope")
                # plt.ylabel("Entropy Envelope")
                # plt.xlabel("Hours")
                # plt.show()

            if plot_entropy:

                plt.figure()
                plt.subplot(411)
                plt.plot(pred_plus, label="Prediction", linewidth=0.6)
                plt.plot(target_plus, label="True Label", linewidth=0.6)
                plt.scatter(non_matches_idx, pred_plus[non_matches_idx], marker='*', edgecolors="r", label="Mistakes", linewidth=0.6)
                # plt.plot(non_matches_idx,"*")
                plt.yticks([0, 1, 2, 3, 4], labels=["Wake", "N1", "REM", "N2", "N3"])
                # plt.xticks([i * 120 for i in range((hours // 120) + 1)],
                #            labels=["{}".format(i) for i in range((hours // 120) + 1)])
                plt.xticks([])
                plt.xlim(0,hours)
                plt.yticks(fontsize=8)


                # plt.legend()
                plt.ylabel("Labels", fontsize=8)
                # plt.xlabel("Hours")
                plt.title("Patient {}".format(patient_num))

                plt.subplot(412)
                plt.plot(pred_eeg_plus, label="Pred EEG", linewidth=0.6)
                plt.plot(target_plus, label="True Label", linewidth=0.6)
                plt.scatter(non_matches_idx_eeg, pred_eeg_plus[non_matches_idx_eeg], marker='*', edgecolors="r", label="Mistakes", linewidth=0.6)
                plt.yticks([0, 1, 2, 3, 4], labels=["Wake", "N1", "REM", "N2", "N3"])
                plt.xticks([])
                plt.xlim(0,hours)
                plt.yticks(fontsize=8)
                plt.ylabel("Labels EEG", fontsize=8)

                plt.subplot(413)
                plt.plot(pred_eog_plus, label="Pred EOG", linewidth=0.6)
                plt.plot(target_plus, label="True Label", linewidth=0.6)
                plt.scatter(non_matches_idx_eog, pred_eog_plus[non_matches_idx_eog], marker='*', edgecolors="r", label="Mistakes", linewidth=0.6)
                plt.yticks([0, 1, 2, 3, 4], labels=["Wake", "N1", "REM", "N2", "N3"])
                plt.xticks([])
                plt.xlim(0,hours)
                plt.yticks(fontsize=8)
                plt.ylabel("Labels EOG", fontsize=8)

                # plt.subplot(412)
                # plt.plot(entropy_pred, label="Entropy", linewidth=0.6)
                # plt.scatter(non_matches_idx, entropy_pred[non_matches_idx], marker='*', edgecolors="r", label="Mistakes", linewidth=0.4)
                # # plt.xticks([i * 120 for i in range((hours // 120) + 1)],
                # #            labels=["{}".format(i) for i in range((hours // 120) + 1)])
                # plt.ylabel("Entropy", fontsize=8)
                # plt.xlim(0,hours)
                # plt.xticks([])

                plt.subplot(414)
                plt.plot(kappa_window.flatten(), color="k", label="MM", linewidth=0.7)
                plt.plot(kappa_window_eeg.flatten(), color="b", label="EEG", linewidth=0.7)
                plt.plot(kappa_window_eog.flatten(), color="y", label="EOG", linewidth=0.7)
                # plt.xticks([i * 120 for i in range((hours // 120) + 1)],
                #            labels=["{}".format(i) for i in range((hours // 120) + 1)])
                plt.xlim(0,len(kappa_window.flatten()))
                plt.ylabel("Kappa", fontsize=8)
                plt.xticks([])
                plt.yticks(fontsize=8)
                plt.legend(prop={'size': 8})
                plt.show()

                # plt.subplot(712)
                # plt.plot(inter_eeg_mean_floor.flatten(), label="EEG_W_Mean", linewidth=0.3)
                # # plt.xticks([i * 120 for i in range((hours // 120) + 1)],
                # #            labels=["{}".format(i) for i in range((hours // 120) + 1)])
                # plt.xlim(0,len(inter_eeg_mean_floor.flatten()))
                # plt.ylabel("EEG W Mean", fontsize=8)
                # plt.xticks([])
                # plt.yticks(fontsize=8)


                # plt.subplot(713)
                # plt.plot(inter_eeg_std_floor.flatten(), label="EEG_W_STD", linewidth=0.3)
                # # plt.xticks([i * 120 for i in range((hours // 120) + 1)],
                # #            labels=["{}".format(i) for i in range((hours // 120) + 1)])
                # plt.xlim(0,len(inter_eeg_std_floor.flatten()))
                # plt.ylabel("EEG W STD", fontsize=8)
                # plt.xticks([])
                # plt.yticks(fontsize=8)
                #
                # plt.subplot(714)
                # plt.plot(inter_eog_mean_floor.flatten(), label="EEG_W_Mean", linewidth=0.3)
                # # plt.xticks([i * 120 for i in range((hours // 120) + 1)],
                # #            labels=["{}".format(i) for i in range((hours // 120) + 1)])
                # plt.xlim(0,len(inter_eog_mean_floor.flatten()))
                # plt.ylabel("EOG W Mean", fontsize=8)
                # plt.xticks([])
                # plt.yticks(fontsize=8)
                #
                #
                # plt.subplot(715)
                # plt.plot(inter_eog_std_floor.flatten(), label="EEG_W_STD", linewidth=0.3)
                # # plt.xticks([i * 120 for i in range((hours // 120) + 1)],
                # #            labels=["{}".format(i) for i in range((hours // 120) + 1)])
                # plt.xlim(0,len(inter_eog_std_floor.flatten()))
                # plt.ylabel("EOG W STD", fontsize=8)
                # plt.xticks([])
                # plt.yticks(fontsize=8)

                # plt.subplot(412)
                # plt.plot((views_eeg_time.flatten() - views_eeg_time.flatten().mean() )/views_eeg_time.flatten().std(), label="EEG_Time", linewidth=0.3)
                # # plt.xticks([i * 120 for i in range((hours // 120) + 1)],
                # #            labels=["{}".format(i) for i in range((hours // 120) + 1)])
                # plt.xlim(0,len(views_eeg_time.flatten()))
                #
                # plt.ylabel("EEG Time", fontsize=8)
                # plt.xticks([])
                #
                # plt.subplot(413)
                # plt.plot((views_eog_time.flatten() - views_eog_time.flatten().mean() )/views_eog_time.flatten().std(), label="EOG_Time", linewidth=0.3)
                # # plt.xticks([i * 120 for i in range((hours // 120) + 1)],
                # #            labels=["{}".format(i) for i in range((hours // 120) + 1)])
                # plt.xlim(0,len(views_eog_time.flatten()))
                #
                # plt.ylabel("EOG Time", fontsize=8)
                # plt.xticks([])


                # plt.subplot(413)
                # plt.plot(inter_distance, 'b', label='high', linewidth=0.6)
                # # plt.plot(x[low_idx_inter_dist], inter_distance[low_idx_inter_dist], 'b', label='high')
                # plt.scatter(non_matches_idx, inter_distance[non_matches_idx], marker='*', edgecolors="r", label="Mistakes", linewidth=0.4)
                # plt.xticks([])
                # plt.ylabel("Distance EEG-EOG", fontsize=8)
                # plt.xlim(0,hours)


                # plt.subplot(716)
                # t = np.linspace(0, len(views_eeg) - 1, len(views_eeg))
                # # t = np.linspace(0, len(inter_eeg) - 1, len(inter_eeg))
                # f = np.linspace(0, 129 - 1, 129)
                # plt.pcolormesh(t, f, views_eeg.transpose(), shading='auto')
                # # plt.pcolormesh(t, f, inter_eeg.transpose())
                # # plt.xticks([])
                # # plt.xticks([i * 120 for i in range((hours // 120) + 1)],
                # #            labels=["{}".format(i) for i in range((hours // 120) + 1)])
                # plt.yticks(fontsize=8)
                # plt.ylabel("F EEG")
                # plt.xlabel("Hours")
                # # plt.show()
                #
                # plt.subplot(717)
                # t = np.linspace(0, len(views_eog) - 1, len(views_eog))
                # # t = np.linspace(0, len(inter_eeg) - 1, len(inter_eeg))
                # f = np.linspace(0, 129 - 1, 129)
                # plt.pcolormesh(t, f, views_eog.transpose(), shading='auto')
                # # plt.pcolormesh(t, f, inter_eeg.transpose())
                # # plt.xticks([])
                # # plt.xticks([i * 120 for i in range((hours // 120) + 1)],
                # #            labels=["{}".format(i) for i in range((hours // 120) + 1)])
                # plt.yticks(fontsize=8)
                # plt.ylabel("F EOG")
                # plt.xlabel("Hours")
                # plt.show()


                #
                # plt.subplot(413)
                # t = np.linspace(0, len(views_eog) - 1, len(views_eog))
                # # t = np.linspace(0, len(inter_eog) - 1, len(inter_eog))
                # f = np.linspace(0, 129 - 1, 129)
                # plt.pcolormesh(t, f, views_eog.transpose(), shading='auto')
                # # plt.pcolormesh(t, f, inter_eog.transpose())
                # plt.xticks([])
                # plt.ylabel("F EOG")

                # plt.subplot(413)
                # # plt.plot(ce_loss, "r", label="CE_Loss")
                # plt.plot(x[low_idx_ce], ce_loss[low_idx_ce], 'lightblue', label='high', linewidth=0.6)
                # plt.scatter(non_matches_idx, ce_loss[non_matches_idx], marker='*', edgecolors="r", label="Mistakes", linewidth=0.4)
                # # plt.xticks([i * 120 for i in range((hours // 120) + 1)],
                # #            labels=["{}".format(i) for i in range((hours // 120) + 1)])
                # plt.ylabel("CE_Loss", fontsize=8)
                # plt.xticks([])
                # plt.xlim(0,hours)


                #
                # plt.subplot(414)
                # # x = np.linspace(0, len(match_loss) - 1, len(match_loss))
                # # plt.plot(match_loss.numpy(), "g", label="Matches")
                # # plt.plot(entropy_pred, label="Entropy")
                # plt.plot(match_loss, label="Match Loss", linewidth=0.6)
                # # plt.plot(x[low_idx_match], match_loss[low_idx_match], 'g', label='high')
                # plt.scatter(non_matches_idx, match_loss[non_matches_idx], marker='*', edgecolors="r", label="Mistakes", linewidth=0.4)
                #
                # plt.xticks([i * 120 for i in range((hours // 120) + 1)],
                #            labels=["{}".format(i) for i in range((hours // 120) + 1)])
                #
                # # plt.xticks([])
                # plt.xlim(0,hours)
                #
                # # plt.title("Patient {}".format(patient_num))
                # plt.ylabel("Align_loss", fontsize=8)
                # # # plt.ylabel("Entropy Envelope")
                #
                # # plt.subplot(414)
                # # # plt.plot(combined_loss, "k", label="Combined_Loss")
                # # plt.plot(x[low_idx_combined], combined_loss[low_idx_combined], 'k', label='high')
                # # plt.scatter(non_matches_idx, combined_loss[non_matches_idx], marker='*', edgecolors="r", label="Mistakes")
                # # plt.xticks([i * 120 for i in range((hours // 120) + 1)],
                # #            labels=["{}".format(i) for i in range((hours // 120) + 1)])
                # # plt.ylabel("Combined_Loss")
                # plt.xlabel("Hours")
                # plt.show()


                # plt.figure()
                # plt.subplot(411)
                # plt.plot(pred_plus, label="Prediction", linewidth=0.6)
                # plt.plot(target_plus, label="True Label", linewidth=0.6)
                # plt.scatter(non_matches_idx, pred_plus[non_matches_idx], marker='*', edgecolors="r", label="Mistakes", linewidth=0.6)
                # # plt.plot(non_matches_idx,"*")
                # plt.yticks([0, 1, 2, 3, 4], labels=["Wake", "N1", "REM", "N2", "N3"])
                # # plt.xticks([i * 120 for i in range((hours // 120) + 1)],
                # #            labels=["{}".format(i) for i in range((hours // 120) + 1)])
                # plt.xticks([])
                # plt.xlim(0,hours)
                #
                # # plt.legend()
                # plt.ylabel("Labels", fontsize=8)
                # # plt.xlabel("Hours")
                # plt.title("Patient {}".format(patient_num))
                #
                # plt.subplot(412)
                # plt.plot((views_eeg_time.flatten() - views_eeg_time.flatten().mean() )/views_eeg_time.flatten().std(), label="EEG_Time", linewidth=0.3)
                # # plt.xticks([i * 120 for i in range((hours // 120) + 1)],
                # #            labels=["{}".format(i) for i in range((hours // 120) + 1)])
                # plt.xlim(0,len(views_eeg_time.flatten()))
                #
                # plt.ylabel("EEG Time", fontsize=8)
                # plt.xticks([])
                #
                # plt.subplot(413)
                # plt.plot((views_eog_time.flatten() - views_eog_time.flatten().mean() )/views_eog_time.flatten().std(), label="EOG_Time", linewidth=0.3)
                # # plt.xticks([i * 120 for i in range((hours // 120) + 1)],
                # #            labels=["{}".format(i) for i in range((hours // 120) + 1)])
                # plt.xlim(0,len(views_eog_time.flatten()))
                #
                # plt.ylabel("EOG Time", fontsize=8)
                # plt.xticks([])
                #
                # plt.subplot(414)
                # # x = np.linspace(0, len(match_loss) - 1, len(match_loss))
                # # plt.plot(match_loss.numpy(), "g", label="Matches")
                # # plt.plot(entropy_pred, label="Entropy")
                # plt.plot(match_loss, label="Match Loss", linewidth=0.6)
                # # plt.plot(x[low_idx_match], match_loss[low_idx_match], 'g', label='high')
                # plt.scatter(non_matches_idx, match_loss[non_matches_idx], marker='*', edgecolors="r", label="Mistakes", linewidth=0.4)
                #
                # plt.xticks([i * 120 for i in range((hours // 120) + 1)],
                #            labels=["{}".format(i) for i in range((hours // 120) + 1)])
                #
                # # plt.xticks([])
                # plt.xlim(0,hours)
                #
                # # plt.title("Patient {}".format(patient_num))
                # plt.ylabel("Align_loss", fontsize=8)
                # # # plt.ylabel("Entropy Envelope")
                #
                # # plt.subplot(414)
                # # # plt.plot(combined_loss, "k", label="Combined_Loss")
                # # plt.plot(x[low_idx_combined], combined_loss[low_idx_combined], 'k', label='high')
                # # plt.scatter(non_matches_idx, combined_loss[non_matches_idx], marker='*', edgecolors="r", label="Mistakes")
                # # plt.xticks([i * 120 for i in range((hours // 120) + 1)],
                # #            labels=["{}".format(i) for i in range((hours // 120) + 1)])
                # # plt.ylabel("Combined_Loss")
                # plt.xlabel("Hours")
                # plt.show()

            return test_acc, test_f1, test_k, test_auc, test_conf, test_perclass_f1, test_spec, test_sens, total_entropy
        except StopIteration:
            pass
def validate_specific_patient_singlemod(data_loader, patient_num, model, device, plot_hypnogram_flag=True, return_matches=False, plot_matches=False, plot_entropy=True):
    model.eval()
    this_data_loader = copy.deepcopy(data_loader)
    this_data_loader.dataset.choose_specific_patient(patient_num)
    # this_data_loader.dataset.config.statistics["print"] = True
    # this_data_loader.dataset.print_statistics_per_patient()

    with torch.no_grad():
        try:
            tts, preds, matches, inits, views_eeg, views_eog, inter_eeg, inter_eog = [], [], [], [], [], [], [], []
            views_eeg_time, views_eog_time = [], []
            for batch_idx, (data, target, init, _) in enumerate(this_data_loader):
                views = [data[i].float().to(device) for i in range(len(data))]
                label = target.to(device)
                output = model(views)

                # pred = get_predictions_time_series(model, views, init)
                tts.append(label)
                preds.append(output)
                # preds.append(pred)
                views_eeg.append(views[0])
                views_eeg_time.append(views[1])
                inits.append(init.flatten())

            tts = torch.cat(tts).cpu().numpy()
            preds = torch.cat(preds).cpu().numpy()
            views_eeg = torch.cat(views_eeg).cpu().squeeze().flatten(start_dim=0, end_dim=1)
            views_eeg = einops.rearrange(views_eeg, "a b c -> (a c) b").numpy()
            views_eeg_time = torch.cat(views_eeg_time).cpu().squeeze().flatten(start_dim=0, end_dim=1)

            # print(views_eeg_time.shape)
            # views_eeg_time = einops.rearrange(views_eeg_time, "a c -> (a c)").numpy()
            # views_eog_time = torch.cat(views_eog_time).cpu().squeeze().flatten(start_dim=0, end_dim=1)
            # views_eog_time = einops.rearrange(views_eog_time, "a c -> (a c)").numpy()

            # inter_eeg = einops.rearrange(inter_eeg, "a b c -> (a c) b").numpy()
            # inter_eog = torch.cat(inter_eog).cpu().squeeze().flatten(start_dim=0, end_dim=1)
            # inter_eog = einops.rearrange(inter_eog, "a b c -> (a c) b").numpy()


            multiclass = False
            if preds.shape[1] > 2:
                multiclass = True

            entropy_pred = entropy(preds, axis=1)
            class_pred = preds.argmax(axis=1)
            # entropy_correct_class = entropy_pred[class_pred == tts].mean()
            # entropy_wrong_class = entropy_pred[class_pred != tts].mean()
            # total_entropy = entropy_pred.mean()
            total_entropy = 0

            # print("{} entropy for correct class class prediction is {} and for wrong class predictions {}".format(patient_num,
            #                                                                                                       entropy_correct_class,
            #                                                                                                       entropy_wrong_class))

            preds_for_loss = copy.deepcopy(preds)
            preds = preds.argmax(axis=1)
            if len(tts.shape)>2:
                tts = tts.argmax(axis=-1)
            tts = tts.flatten()

            test_acc = np.equal(tts, preds).sum() / len(tts)
            test_f1 = f1_score(preds, tts) if not multiclass else f1_score(preds, tts, average="macro")
            test_perclass_f1 = f1_score(preds, tts) if not multiclass else f1_score(preds, tts, average=None)
            test_k = cohen_kappa_score(tts, preds)
            test_auc = roc_auc_score(tts, preds) if not multiclass else 0
            test_conf = confusion_matrix(tts, preds)
            tp, fp, tn, fn = perf_measure(tts, preds)
            test_spec = tn / (tn + fp) if (tn + fp) != 0 else 0
            test_sens = tp / (tp + fn) if (tp + fn) != 0 else 0
            # print("{0} accuracy: {1:.2f}% f1 :{2:.4f}, k :{3:.4f}, sens:{4:.4f}, spec:{5:.4f}, f1_per_class :{6:40}".format(
            #     patient_num,
            #     test_acc * 100,
            #     test_f1,
            #     test_k, test_spec, test_sens,
            #     "{}".format(list(test_perclass_f1))))
            # norm_n_plot_confusion_matrix(test_conf, patient_num)

            if plot_hypnogram_flag or True:
            # if test_f1>0.85:

                pred_plus = copy.deepcopy(preds)
                pred_plus[pred_plus == 4] = 5
                pred_plus[pred_plus == 3] = 4
                pred_plus[pred_plus == 2] = 3
                pred_plus[pred_plus == 5] = 2

                target_plus = copy.deepcopy(tts)
                target_plus[target_plus == 4] = 5
                target_plus[target_plus == 3] = 4
                target_plus[target_plus == 2] = 3
                target_plus[target_plus == 5] = 2

                from_hours_to_plot = int(120*0)
                hours_to_plot = -1 #int(120*2.5)
                print(from_hours_to_plot, hours_to_plot)
                pred_plus = pred_plus[from_hours_to_plot:hours_to_plot]
                target_plus = target_plus[from_hours_to_plot:hours_to_plot]
                views_eeg_time = views_eeg_time[from_hours_to_plot:hours_to_plot]
                views_eeg = views_eeg[from_hours_to_plot:hours_to_plot]
                # match_loss = match_loss[:hours_to_plot]

                # target = target + 0.02

                non_matches = (pred_plus != target_plus).astype(int)
                non_matches_idx = non_matches.nonzero()[0]

                target_plus = target_plus + 0.02

                # print("Non matching indices are:")
                # print(non_matches_idx)
                hours = len(target_plus)

                # plt.figure()
                # plt.plot(pred_plus, label="Prediction")
                # plt.plot(target_plus, label="True Label")
                # plt.scatter(non_matches_idx, pred_plus[non_matches_idx], marker='*', edgecolors="r", label="Mistakes")
                # # plt.plot(non_matches_idx,"*")
                # plt.yticks([0, 1, 2, 3, 4], labels=["Wake", "N1", "REM", "N2", "N3"])
                # plt.xticks([i * 120 for i in range((hours // 120) + 1)],
                #            labels=["{}".format(i) for i in range((hours // 120) + 1)])
                # plt.legend()
                # plt.ylabel("Labels")
                # plt.xlabel("Hours")
                # plt.title("Patient {}".format(patient_num))
                # plt.show()

                ce_loss = nn.CrossEntropyLoss(reduction='none')(torch.from_numpy(preds_for_loss),torch.from_numpy(tts))

                # match_loss = (match_loss - match_loss.mean())/match_loss.std()
                # ce_loss = (ce_loss - ce_loss.mean())/ce_loss.std()
                # combined_loss = (combined_loss - combined_loss.mean())/combined_loss.std()

                def hl_envelopes_idx(s, dmin=-1, dmax=1, split=False):
                    """
                    Input :
                    s: 1d-array, data signal from which to extract high and low envelopes
                    dmin, dmax: int, optional, size of chunks, use this if the size of the input signal is too big
                    split: bool, optional, if True, split the signal in half along its mean, might help to generate the envelope in some cases
                    Output :
                    lmin,lmax : high/low envelope idx of input signal s
                    """

                    # locals min
                    lmin = (np.diff(np.sign(np.diff(s))) > 0).nonzero()[0] + 1
                    # locals max
                    lmax = (np.diff(np.sign(np.diff(s))) < 0).nonzero()[0] + 1

                    if split:
                        # s_mid is zero if s centered around x-axis or more generally mean of signal
                        s_mid = np.mean(s)
                        # pre-sorting of locals min based on relative position with respect to s_mid
                        lmin = lmin[s[lmin] < s_mid]
                        # pre-sorting of local max based on relative position with respect to s_mid
                        lmax = lmax[s[lmax] > s_mid]

                    # global max of dmax-chunks of locals max
                    lmin = lmin[[i + np.argmin(s[lmin[i:i + dmin]]) for i in range(0, len(lmin), dmin)]]
                    # global min of dmin-chunks of locals min
                    lmax = lmax[[i + np.argmax(s[lmax[i:i + dmax]]) for i in range(0, len(lmax), dmax)]]

                    return lmin, lmax
                high_idx_ce, low_idx_ce = hl_envelopes_idx(ce_loss.numpy())

                x = np.linspace(0, len(ce_loss) - 1, len(ce_loss))

                # plt.figure()
                # plt.subplot(211)
                # plt.plot(pred_plus, label="Prediction")
                # plt.plot(target_plus, label="True Label")
                # plt.scatter(non_matches_idx, pred_plus[non_matches_idx], marker='*', edgecolors="r", label="Mistakes")
                # # plt.plot(non_matches_idx,"*")
                # plt.yticks([0, 1, 2, 3, 4], labels=["Wake", "N1", "REM", "N2", "N3"])
                # plt.xticks([i * 120 for i in range((hours // 120) + 1)],
                #            labels=["{}".format(i) for i in range((hours // 120) + 1)])
                # plt.legend()
                # plt.ylabel("Labels")
                # plt.xlabel("Hours")
                # plt.title("Patient {}".format(patient_num))
                # plt.subplot(212)
                # # plt.plot(match_loss.numpy(), label="Matches")
                # # plt.plot(entropy_pred, label="Entropy")
                # # plt.plot(low_idx, label="Match Loss")
                # plt.plot(x[low_idx], match_loss[low_idx], 'g', label='high')
                # plt.scatter(non_matches_idx, entropy_pred[non_matches_idx], marker='*', edgecolors="r", label="Mistakes")
                # # plt.scatter(non_matches_idx, match_loss[non_matches_idx], marker='*', edgecolors="r", label="Mistakes")
                # plt.xticks([i * 120 for i in range((hours // 120) + 1)],
                #            labels=["{}".format(i) for i in range((hours // 120) + 1)])
                # # plt.title("Patient {}".format(patient_num))
                # # plt.ylabel("CE loss Envelope")
                # plt.ylabel("Entropy Envelope")
                # plt.xlabel("Hours")
                # plt.show()

            if plot_entropy:

                plt.figure()
                plt.subplot(411)
                plt.plot(pred_plus, label="Prediction", linewidth=0.6)
                plt.plot(target_plus, label="True Label", linewidth=0.6)
                plt.scatter(non_matches_idx, pred_plus[non_matches_idx], marker='*', edgecolors="r", label="Mistakes", linewidth=0.6)
                # plt.plot(non_matches_idx,"*")
                plt.yticks([0, 1, 2, 3, 4], labels=["Wake", "N1", "REM", "N2", "N3"])
                # plt.xticks([i * 120 for i in range((hours // 120) + 1)],
                #            labels=["{}".format(i) for i in range((hours // 120) + 1)])
                plt.xticks([])
                plt.xlim(0,hours)

                # plt.legend()
                plt.ylabel("Labels", fontsize=8)
                # plt.xlabel("Hours")
                plt.title("Patient {}".format(patient_num))

                # plt.subplot(412)
                # plt.plot(entropy_pred, label="Entropy", linewidth=0.6)
                # plt.scatter(non_matches_idx, entropy_pred[non_matches_idx], marker='*', edgecolors="r", label="Mistakes", linewidth=0.4)
                # # plt.xticks([i * 120 for i in range((hours // 120) + 1)],
                # #            labels=["{}".format(i) for i in range((hours // 120) + 1)])
                # plt.ylabel("Entropy", fontsize=8)
                # plt.xlim(0,hours)
                # plt.xticks([])

                # plt.subplot(412)
                # plt.plot((views_eeg_time.flatten() - views_eeg_time.flatten().mean() )/views_eeg_time.flatten().std(), label="EEG_Time", linewidth=0.3)
                # # plt.xticks([i * 120 for i in range((hours // 120) + 1)],
                # #            labels=["{}".format(i) for i in range((hours // 120) + 1)])
                # plt.xlim(0,len(views_eeg_time.flatten()))
                #
                # plt.ylabel("EEG Time", fontsize=8)
                # plt.xticks([])
                #
                # plt.subplot(413)
                # plt.plot((views_eog_time.flatten() - views_eog_time.flatten().mean() )/views_eog_time.flatten().std(), label="EOG_Time", linewidth=0.3)
                # # plt.xticks([i * 120 for i in range((hours // 120) + 1)],
                # #            labels=["{}".format(i) for i in range((hours // 120) + 1)])
                # plt.xlim(0,len(views_eog_time.flatten()))
                #
                # plt.ylabel("EOG Time", fontsize=8)
                # plt.xticks([])


                # plt.subplot(413)
                # plt.plot(inter_distance, 'b', label='high', linewidth=0.6)
                # # plt.plot(x[low_idx_inter_dist], inter_distance[low_idx_inter_dist], 'b', label='high')
                # plt.scatter(non_matches_idx, inter_distance[non_matches_idx], marker='*', edgecolors="r", label="Mistakes", linewidth=0.4)
                # plt.xticks([])
                # plt.ylabel("Distance EEG-EOG", fontsize=8)
                # plt.xlim(0,hours)


                plt.subplot(412)
                t = np.linspace(0, len(views_eeg) - 1, len(views_eeg))
                # t = np.linspace(0, len(inter_eeg) - 1, len(inter_eeg))
                f = np.linspace(0, 129 - 1, 129)
                plt.pcolormesh(t, f, views_eeg.transpose(), shading='auto')
                # plt.pcolormesh(t, f, inter_eeg.transpose())
                plt.xticks([])
                plt.ylabel("F EEG")

                plt.subplot(413)
                plt.plot((views_eeg_time.flatten() - views_eeg_time.flatten().mean() )/views_eeg_time.flatten().std(), label="EEG_Time", linewidth=0.3)
                # plt.xticks([i * 120 for i in range((hours // 120) + 1)],
                #            labels=["{}".format(i) for i in range((hours // 120) + 1)])
                plt.xlim(0,len(views_eeg_time.flatten()))

                plt.ylabel("EEG Time", fontsize=8)
                plt.xticks([])


                #
                plt.subplot(414)
                # x = np.linspace(0, len(match_loss) - 1, len(match_loss))
                # plt.plot(match_loss.numpy(), "g", label="Matches")
                # plt.plot(entropy_pred, label="Entropy")
                plt.plot(ce_loss, label="CE Loss", linewidth=0.6)
                # plt.plot(x[low_idx_match], match_loss[low_idx_match], 'g', label='high')
                plt.scatter(non_matches_idx, ce_loss[non_matches_idx], marker='*', edgecolors="r", label="Mistakes", linewidth=0.4)

                plt.xticks([i * 120 for i in range((hours // 120) + 1)],
                           labels=["{}".format(i) for i in range((hours // 120) + 1)])

                # plt.xticks([])
                plt.xlim(0,hours)

                # plt.title("Patient {}".format(patient_num))
                plt.ylabel("CE Loss", fontsize=8)
                # # plt.ylabel("Entropy Envelope")

                # plt.subplot(414)
                # # plt.plot(combined_loss, "k", label="Combined_Loss")
                # plt.plot(x[low_idx_combined], combined_loss[low_idx_combined], 'k', label='high')
                # plt.scatter(non_matches_idx, combined_loss[non_matches_idx], marker='*', edgecolors="r", label="Mistakes")
                # plt.xticks([i * 120 for i in range((hours // 120) + 1)],
                #            labels=["{}".format(i) for i in range((hours // 120) + 1)])
                # plt.ylabel("Combined_Loss")
                plt.xlabel("Hours")
                plt.show()


            return test_acc, test_f1, test_k, test_auc, test_conf, test_perclass_f1, test_spec, test_sens, total_entropy
        except StopIteration:
            pass

def save_test_results(checkpoint, save_dir, test_results):
    test_results_dict = {"post_test_results":{"accuracy":test_results[0],"f1":test_results[1],"k":test_results[2],"auc":test_results[3],
                         "conf_matrix":test_results[4], "preclass_f1":test_results[5], "spec":test_results[6], "sens":test_results[7]}}
    checkpoint.update(test_results_dict)
    try:
        torch.save(checkpoint, save_dir)
        if config.verbose:
            print("Models has saved successfully in {}".format(save_dir))
    except:
        raise Exception("Problem in model saving")

def gather_comparisons(model, data_loader, f1_comparisons, k_comparisons, entropy_comparisons, patient_map, model_eeg=None, model_eog=None):

    # with open("/esat/smcdata/users/kkontras/Image_Dataset/no_backup/Sleep_SHHS_2/patient_map_shhs1to2.pkl", "rb") as f:
    #     patient_map_shhs1to2 = pickle.load(f)

    patient_list = find_patient_list(data_loader=data_loader)
    f1_comparisons[config_name] = {}
    entropy_comparisons[config_name] = {}
    k_comparisons[config_name] = {}
    # patient_list = [55,2,91,40,23]
    # with open("/esat/smcdata/users/kkontras/Image_Dataset/no_backup/Sleep_SHHS/Version_1/patients_with_bad_eeg.pkl", "rb") as f:
    #     patients_of_interest = pickle.load(f)

    # pbar = tqdm(enumerate(patient_list), bar_format="{desc:<5}{percentage:3.0f}%|{bar}{r_bar}")
    for i, patient_num in enumerate(patient_list):
        # if patient_map["patient_{}".format(f'{patient_num:04}')] not in patients_of_interest["p"]:
        #     print("We skipped {}".format("patient_{}".format(f'{patient_num:04}')))
        #     continue
        # if patient_num not in patient_map_shhs1to2.keys():
        #     continue

        test_acc, test_f1, test_k, test_auc, test_conf, test_perclass_f1, test_spec, test_sens, total_entropy = validate_specific_patient(model=model, model_eeg=model_eeg,model_eog=model_eog, data_loader=data_loader, device=device, patient_num=patient_num, plot_hypnogram_flag=True, plot_matches=True, return_matches=False)
        # test_acc, test_f1, test_k, test_auc, test_conf, test_perclass_f1, test_spec, test_sens, total_entropy = validate_specific_patient_singlemod(model=model, data_loader=data_loader, device=device, patient_num=patient_num, plot_hypnogram_flag=True, plot_matches=True, return_matches=True)
        f1_comparisons[config_name][patient_map["patient_{}".format(f'{patient_num:04}')]] = test_f1
        k_comparisons[config_name][patient_map["patient_{}".format(f'{patient_num:04}')]] = test_k
        entropy_comparisons[config_name][patient_map["patient_{}".format(f'{patient_num:04}')]] = total_entropy
        # pbar.set_description("Patient {} has acc: {}, f1: {}, k:{} and f1_per_class: {}".format(i, round(test_acc*100,1), round(test_f1*100,1), round(test_k,3), np.round(test_perclass_f1*100,1)))
        # pbar.refresh()
        # print("Patient {} has acc: {}, f1: {}, k:{} and f1_per_class: {}".format(patient_num, round(test_acc*100,1), round(test_f1*100,1), round(test_k,3), np.round(test_perclass_f1*100,1)))


    return f1_comparisons, k_comparisons, entropy_comparisons

def find_patient_list(data_loader):
    patient_list = [int(data.split("/")[-1][1:5]) for data in data_loader.dataset.dataset[0] if data.split("/")[-1]!="empty"]
    return patient_list

def plot_hypnogram(data_loader, patient_num, model, device):
    model.eval()
    data_loader.dataset.choose_specific_patient(patient_num)
    with torch.no_grad():
        try:
            data, target, inits, idxs = next(iter(data_loader))
            views = [data[i].float().to(device) for i in range(len(data))]
            print(target.shape)
            print(target)

            if len(target.shape)>2:
                target = target.argmax(dim=-1)
            target = target.to(device).flatten()
            print(target)
            pred = get_predictions_time_series(model, views, inits)
            pred = pred.argmax(axis=1).cpu()

        except StopIteration:
            pass

    # plt.figure()
    # fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplot(3,2)
    # fig, axs = plt.subplots(3,2)
    # t=np.arange(0,29)
    # f=np.arange(0,129)
    # amp = 2 * np.sqrt(2)
    # axs[0,0].pcolormesh(t, f, views[0][1,-1,0,:,:].detach().cpu().numpy(), vmin=0, vmax=amp, shading='gouraud',label="EEG C4")
    # axs[0,1].pcolormesh(t, f, views[0][1,-1,1,:,:].detach().cpu().numpy(), vmin=0, vmax=amp, shading='gouraud',label="EEG C3")
    # axs[1,0].pcolormesh(t, f, views[1][1,-1,0,:,:].detach().cpu().numpy(), vmin=0, vmax=amp, shading='gouraud',label="EOG R")
    # axs[1,1].pcolormesh(t, f, views[1][1,-1,1,:,:].detach().cpu().numpy(), vmin=0, vmax=amp, shading='gouraud',label="EOG L")
    # axs[2,0].pcolormesh(t, f, views[2][1,-1,0,:,:].detach().cpu().numpy(), vmin=0, vmax=amp, shading='gouraud',label="EMG")
    #
    # axs[0, 0].axis("off")
    # axs[0, 1].axis("off")
    # axs[1, 0].axis("off")
    # axs[1, 1].axis("off")
    # axs[2, 0].axis("off")
    # axs[2, 1].axis("off")
    # font_dict = {'fontsize': 9,
    #  'fontweight': 50}
    #
    # axs[0, 0].set_title("EEG C4", fontdict = font_dict)
    # axs[0, 1].set_title("EEG C3", fontdict=font_dict)
    # axs[1, 0].set_title("EOG R", fontdict=font_dict)
    # axs[1, 1].set_title("EOG L", fontdict=font_dict)
    # axs[2, 0].set_title("EMG", fontdict=font_dict)
    # fig.suptitle(" N3 Epoch ")
    # plt.legend()
    # plt.show()

    c, count = np.unique(target.detach().cpu().numpy(), return_counts=True)
    s = "Patient {} has {} windows with labels ".format(patient_num, len(target))
    for i in range(len(c)):
        s += "{}-{} ".format(c[i], f'{count[i]:04}')
    print(s)
    from_hour = 0
    to_hour = 3
    target = target[int(60 * 2 * from_hour) : int(60 * 2 * to_hour)]
    pred = pred[int(60 * 2 * from_hour) : int(60 * 2 * to_hour)]
    non_matches = (pred != target).int()
    non_matches_idx = non_matches.nonzero(as_tuple=True)[0]
    print("Non matching indices are:")
    print(non_matches_idx)
    hours = len(target)

    pred = pred.detach().cpu().numpy()
    target = target.detach().cpu().numpy()
    non_matches_idx = non_matches_idx.cpu().numpy()

    pred_plus = copy.deepcopy(pred)
    pred_plus[pred_plus == 4] = 5
    pred_plus[pred_plus == 3] = 4
    pred_plus[pred_plus == 2] = 3
    pred_plus[pred_plus == 5] = 2

    target_plus = copy.deepcopy(target)
    target_plus[target_plus == 4] = 5
    target_plus[target_plus == 3] = 4
    target_plus[target_plus == 2] = 3
    target_plus[target_plus == 5] = 2

    # target = target + 0.02
    target_plus = target_plus + 0.02



    # plt.figure()
    # plt.plot(pred,label="Prediction")
    # plt.plot(target,label="True Label")
    # plt.scatter(non_matches_idx, pred[non_matches_idx], marker='*', edgecolors="r", label="Mistakes")
    # # plt.plot(non_matches_idx,"*")
    # plt.yticks([0, 1, 2, 3, 4], labels=["Wake", "N1", "N2", "N3", "REM"])
    # plt.xticks([i * 120 for i in range((hours // 120) + 1)],
    #            labels=["{}".format(i) for i in range((hours // 120) + 1)])
    # plt.legend()
    # plt.ylabel("Labels")
    # plt.xlabel("Hours")
    # plt.title("Patient {}".format(patient_num))
    # plt.show()


    plt.figure()
    plt.plot(pred_plus,label="Prediction")
    plt.plot(target_plus,label="True Label")
    plt.scatter(non_matches_idx, pred_plus[non_matches_idx], marker='*', edgecolors="r", label="Mistakes")
    # plt.plot(non_matches_idx,"*")
    plt.yticks([0, 1, 2, 3, 4], labels=["Wake", "N1", "REM", "N2", "N3"])
    plt.xticks([i * 120 for i in range((hours // 120) + 1)],
               labels=["{}".format(i) for i in range((hours // 120) + 1)])
    plt.legend()
    plt.ylabel("Labels")
    plt.xlabel("Hours")
    plt.title("Patient {}".format(patient_num))
    plt.show()
def norm_n_plot_confusion_matrix(test_conf,description):
    test_conf = test_conf.astype(float)
    for j in range(len(test_conf)):
        th_sum = test_conf[j].sum()
        for i in range(len(test_conf[j])):
            test_conf[j][i] /= th_sum
    np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
    print(test_conf)
    test_conf = np.round(test_conf, 3)
    import seaborn as sns
    import matplotlib.pyplot as plt

    ax = plt.subplot()
    sns.heatmap(test_conf, annot=True, fmt='g', ax=ax)

    # labels, title and ticks
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('{} Confusion Matrix'.format(description))
    ax.xaxis.set_ticklabels(['Wake', 'N1','N2', 'N3', 'REM'])
    ax.yaxis.set_ticklabels(['REM', 'N3','N2', 'N1', 'Wake'])
    plt.show()
def sleep_plot_losses(config, logs):
    train_loss = np.array([logs["train_logs"][i]["train_loss"] for i in logs["train_logs"]])

    # for i in range(len(train_loss)):
    #     if i%100 == 0: print("{}_{}".format(i, train_loss[i]))

    val_loss = np.array([logs["val_logs"][i]["val_loss"] for i in logs["val_logs"]])
    steps = np.array([i/logs["train_logs"][i]["validate_every"] for i in logs["train_logs"]])-1

    train_loss[train_loss>1.5] = train_loss.mean()
    plt.figure()
    plt.plot(steps, train_loss, label="Train")
    plt.plot(steps, val_loss, label="Valid")

    best_step = logs["best_logs"]["step"]/logs["train_logs"][logs["best_logs"]["step"]]["validate_every"]-1
    best_loss = logs["best_logs"]["val_loss"]

    plt.plot((best_step, best_step), (0, best_loss), linestyle="--", color="y", label="Chosen Point")
    plt.plot((0, best_step), (best_loss, best_loss), linestyle="--", color="y")

    if config.rec_test:
        test_loss = np.array([logs["test_logs"][i]["test_loss"] for i in logs["test_logs"]])
        steps = np.array([i / logs["train_logs"][i]["validate_every"] for i in logs["test_logs"]]) - 1
        best_test_step = np.argmin(test_loss)
        best_test_loss = test_loss[best_test_step]
        plt.plot(steps, test_loss, label="Test")
        plt.plot((best_test_step, best_test_step), (0, best_test_loss), linestyle="--", color="r", label="Chosen Point")
        plt.plot((0, best_test_step), (best_test_loss, best_test_loss), linestyle="--", color="r")

    plt.xlabel('Steps')
    plt.ylabel('Loss Values')
    plt.title("Loss")
    loss_min = np.min([np.min(train_loss),np.min(val_loss)])
    loss_max = np.max([np.max(train_loss),np.max(val_loss)])
    plt.ylim([loss_min-0.05,loss_max+0.05])
    plt.legend()
    # plt.savefig("/users/sista/kkontras/Documents/Sleep_Project/data/2021_data/loss.png")
    plt.show()
def sleep_plot_losses_multisupervised(config, logs):

    list_losses = ["total"]
    list_losses += ["ce_loss_{}".format(i) for i, v in config.model.args.multi_loss.multi_loss_weights.multi_supervised_loss.items() if v!=0]
    list_losses += [i for i, v in config.model.args.multi_loss.multi_loss_weights.items() if v!=0 and type(v)==int]

    train_loss = {loss_key: np.array([logs["train_logs"][i]["train_loss"][loss_key] for i in logs["train_logs"]]) for loss_key in list_losses}
    val_loss = {loss_key: np.array([logs["val_logs"][i]["val_loss"][loss_key] for i in logs["val_logs"]]) for loss_key in list_losses}

    steps = np.array([i/logs["train_logs"][i]["validate_every"] for i in logs["train_logs"]])-1
    best_step = (logs["best_logs"]["step"]/logs["train_logs"][logs["best_logs"]["step"]]["validate_every"])-1

    plt.figure()
    loss_min = 100
    loss_max = 0
    for loss_key in list_losses:
        if loss_key != "total":
            # plt.plot(steps, (train_loss[loss_key] - train_loss[loss_key].mean())/train_loss[loss_key].std(), label="Train_{}".format(loss_key))
            # plt.plot(steps, (val_loss[loss_key] - val_loss[loss_key].mean())/val_loss[loss_key].std() , label="Valid_{}".format(loss_key))

            plt.plot(steps, (train_loss[loss_key]), label="Train_{}".format(loss_key))
            plt.plot(steps, (val_loss[loss_key]) , label="Valid_{}".format(loss_key))

            loss_min = np.minimum(loss_min, np.min((val_loss[loss_key] - val_loss[loss_key].mean())/val_loss[loss_key].std()))
            loss_max = np.maximum(loss_max, np.max((val_loss[loss_key] - val_loss[loss_key].mean())/val_loss[loss_key].std()))
            best_loss = {loss_key: logs["best_logs"]["val_loss"][loss_key]}
            plt.plot((best_step, best_step), (0, best_loss[loss_key]), linestyle="--", color="y")
            plt.plot((0, best_step), (best_loss[loss_key], best_loss[loss_key]), linestyle="--", color="y")

    plt.xlabel('Steps')
    plt.ylabel('Loss Values')
    plt.title("Individual Losses")
    plt.ylim([loss_min-0.05,loss_max+0.05])
    plt.legend()
    # plt.savefig("/users/sista/kkontras/Documents/Sleep_Project/data/2021_data/loss.png")
    plt.show()

    plt.figure()
    loss_min = 100
    loss_max = 0
    loss_key = "total"
    plt.plot(steps, train_loss[loss_key], label="Train_{}".format(loss_key))
    plt.plot(steps, val_loss[loss_key], label="Valid_{}".format(loss_key))
    loss_min = np.minimum(loss_min, np.min(val_loss[loss_key]))
    loss_max = np.maximum(loss_max, np.max(val_loss[loss_key]))
    best_loss = {loss_key: logs["best_logs"]["val_loss"][loss_key]}
    plt.plot((best_step, best_step), (0, best_loss[loss_key]), linestyle="--", color="y", label="Chosen Point")
    plt.plot((0, best_step), (best_loss[loss_key], best_loss[loss_key]), linestyle="--", color="y")

    plt.xlabel('Steps')
    plt.ylabel('Loss Values')
    plt.title("Total Loss")
    plt.ylim([loss_min-0.05,loss_max+0.05])
    plt.legend()
    # plt.savefig("/users/sista/kkontras/Documents/Sleep_Project/data/2021_data/loss.png")
    plt.show()
def sleep_plot_losses_contrastive(config, logs):
    raise NotImplementedError()

def sleep_plot_k(config, logs):

    train_k = np.array([logs["train_logs"][i]["train_k"] for i in logs["train_logs"]])
    val_k = np.array([logs["val_logs"][i]["val_k"]for i in logs["val_logs"]])
    steps = np.array([i/logs["train_logs"][i]["validate_every"] for i in logs["train_logs"]])-1

    plt.figure()
    plt.plot(steps, train_k, label="Train")
    plt.plot(steps, val_k, label="Valid")

    best_step = logs["best_logs"]["step"]/logs["train_logs"][logs["best_logs"]["step"]]["validate_every"]-1
    best_k = logs["best_logs"]["val_k"]

    plt.plot((best_step, best_step), (0, best_k), linestyle="--", color="y", label="Chosen Point")
    plt.plot((0, best_step), (best_k, best_k), linestyle="--", color="y")

    if config.rec_test:
        test_k = np.array([logs["test_logs"][i]["test_k"] for i in logs["test_logs"]])
        best_test_step = np.argmax(test_k)
        best_test_k = test_k[best_test_step]
        plt.plot(steps, test_k, label="Test")
        plt.plot((best_test_step, best_test_step), (0, best_test_k), linestyle="--", color="r", label="Chosen Point")
        plt.plot((0, best_test_step), (best_test_k, best_test_k), linestyle="--", color="r")

    plt.xlabel('Steps')
    plt.ylabel('Kappa')
    plt.title("Cohen's kappa")
    plt.legend()
    kappa_min = np.min([np.min(train_k),np.min(val_k)])
    kappa_max = np.max([np.max(train_k),np.max(val_k)])
    plt.ylim([kappa_min,kappa_max+0.05])
    # plt.savefig("/users/sista/kkontras/Documents/Sleep_Project/data/2021_data/kappa.png")
    plt.show()
def sleep_plot_lr(config, logs):

    learning_rate = np.array([logs["train_logs"][i]["learning_rate"] for i in logs["train_logs"]])
    steps = np.array([i/logs["train_logs"][i]["validate_every"] for i in logs["train_logs"]])-1

    plt.figure()
    plt.plot(steps, learning_rate)
    plt.xlabel('Steps')
    plt.ylabel('Learning Rate')
    plt.title("Learning Rate during training steps")
    plt.legend()
    # plt.savefig("/users/sista/kkontras/Documents/Sleep_Project/data/2021_data/kappa.png")
    plt.show()

def sleep_load_encoder(encoders):
    encs = []
    for num_enc in range(len(encoders)):
        if encoders[num_enc]["model"] == "TF":
            layers = ["huy_pos_inner", "inner_att", "aggregation_att_contx_inner", "huy_pos_outer", "outer_att"]
            enc = Multi_Transformer(128, inner= 29, outer = 21, modalities=1, heads=8,
                                 layers = layers, num_layers=4, pos = False)
        else:
            enc_class = globals()[encoders[num_enc]["model"]]
            args = encoders[num_enc]["args"]
            enc = enc_class(args = args)
            enc = nn.DataParallel(enc, device_ids=[torch.device(0)])

        if encoders[num_enc]["pretrainedEncoder"]["use"]:
            print("Loading encoder from {}".format(encoders[num_enc]["pretrainedEncoder"]["dir"]))
            checkpoint = torch.load(encoders[num_enc]["pretrainedEncoder"]["dir"])
            enc.load_state_dict(checkpoint["encoder_state_dict"])
        encs.append(enc)
    return encs

def sleep_umap_plot(model,data_loader, description):
    print("We are in umap plot.")
    model.eval()
    with torch.no_grad():
        tts, total_features = [], []
        pbar = tqdm(enumerate(data_loader), desc=description, leave=False)
        for batch_idx, (data, target, init, _) in pbar:
            views = [data[i].float().to(device) for i in range(len(data))]
            label = target.to(device).flatten()
            features = model(views)
            # pred = get_predictions_time_series(model, views, init)
            tts.append(label.detach().cpu())
            total_features.append(features.detach().cpu())
            pbar.set_description("{} batch {}/{}".format(description, int(batch_idx), int(len(data_loader))))
            pbar.refresh()

        tts = torch.cat(tts).numpy()
        total_features = einops.rearrange(torch.cat(total_features).squeeze(),"b outer f -> (b outer) f").numpy()
    print("Our total features shape ", end="")
    print(total_features.shape)
    print("Our total labels shape ", end="")
    print(tts.shape)
    umap_train = umap.UMAP(min_dist=0.7, n_neighbors=20).fit(total_features)
    plt.scatter(umap_train.embedding_.T[0], umap_train.embedding_.T[1], c=tts, s=2, cmap='Spectral')
    plt.show()
def sleep_plot_f1(config, logs):

    train_f1 = np.array([logs["train_logs"][i]["train_f1"] for i in logs["train_logs"]])
    val_f1 = np.array([logs["val_logs"][i]["val_f1"]for i in logs["val_logs"]])
    steps = np.array([i/logs["train_logs"][i]["validate_every"] for i in logs["train_logs"]])-1

    plt.figure()
    plt.plot(steps, train_f1, label="Train")
    plt.plot(steps, val_f1, label="Valid")

    best_step = logs["best_logs"]["step"]/logs["train_logs"][logs["best_logs"]["step"]]["validate_every"]-1
    best_f1 = logs["best_logs"]["val_f1"]

    plt.plot((best_step, best_step), (0, best_f1), linestyle="--", color="y", label="Chosen Point")
    plt.plot((0, best_step), (best_f1, best_f1), linestyle="--", color="y")

    if config.rec_test:
        test_f1 = np.array([logs["test_logs"][i]["test_f1"] for i in logs["test_logs"]])
        best_test_step = np.argmax(test_f1)
        best_test_f1 = test_f1[best_test_step]
        plt.plot(steps, test_f1, label="Test")
        plt.plot((best_test_step, best_test_step), (0, best_test_f1), linestyle="--", color="r", label="Chosen Point")
        plt.plot((0, best_test_step), (best_test_f1, best_test_f1), linestyle="--", color="r")

    plt.xlabel('Steps')
    plt.ylabel('F1')
    plt.title("Validation F1 ")
    f1_min = np.min([np.min(train_f1),np.min(val_f1)])
    f1_max = np.max([np.max(train_f1),np.max(val_f1)])
    plt.ylim([f1_min-0.05,f1_max+0.05])
    plt.legend()
    # plt.savefig("/users/sista/kkontras/Documents/Sleep_Project/data/2021_data/f1.png")
    plt.show()
def sleep_plot_f1_multisupervised(config, logs):

    list_predictors = ["{}".format(i) for i, v in config.model.args.multi_loss.multi_loss_weights.multi_supervised_loss.items() if v!=0]

    train_f1 = {pred_key: np.array([logs["train_logs"][i]["train_f1"][pred_key] for i in logs["train_logs"]]) for pred_key in list_predictors}
    val_f1 = {pred_key: np.array([logs["val_logs"][i]["val_f1"][pred_key] for i in logs["val_logs"]]) for pred_key in list_predictors}

    steps = np.array([i/logs["train_logs"][i]["validate_every"] for i in logs["train_logs"]])-1
    best_step = logs["best_logs"]["step"]/logs["train_logs"][logs["best_logs"]["step"]]["validate_every"]

    plt.figure()
    loss_min = 100
    loss_max = 0
    for loss_key in list_predictors:
        plt.plot(steps, train_f1[loss_key], label="Train_{}".format(loss_key))
        plt.plot(steps, val_f1[loss_key], label="Valid_{}".format(loss_key))
        loss_min = np.minimum(loss_min, np.min(train_f1[loss_key]))
        loss_min = np.minimum(loss_min, np.min(val_f1[loss_key]))
        loss_max = np.maximum(loss_max, np.max(train_f1[loss_key]))
        loss_max = np.maximum(loss_max, np.max(val_f1[loss_key]))
        best_loss = {loss_key: logs["best_logs"]["val_f1"][loss_key]}
        plt.plot((best_step, best_step), (0, best_loss[loss_key]), linestyle="--", color="y", label="Chosen Point")
        plt.plot((0, best_step), (best_loss[loss_key], best_loss[loss_key]), linestyle="--", color="y")

    plt.xlabel('Steps')
    plt.ylabel('F1 Value')
    plt.title("F1 Multi Predictors")
    plt.ylim([loss_min-0.05,loss_max+0.05])
    plt.legend()
    # plt.savefig("/users/sista/kkontras/Documents/Sleep_Project/data/2021_data/loss.png")
    plt.show()

def sleep_plot_f1_perclass_multisupervised(config, logs):

    list_predictors = ["{}".format(i) for i, v in config.model.args.multi_loss.multi_loss_weights.multi_supervised_loss.items() if v!=0]

    train_f1_perclass = {pred_key: np.array([logs["train_logs"][i]["train_perclassf1"][pred_key] for i in logs["train_logs"]]) for pred_key in list_predictors}
    val_f1_perclass = {pred_key: np.array([logs["val_logs"][i]["val_perclassf1"][pred_key] for i in logs["val_logs"]]) for pred_key in list_predictors}

    steps = np.array([i/logs["train_logs"][i]["validate_every"] for i in logs["train_logs"]])-1
    best_step = logs["best_logs"]["step"]/logs["train_logs"][logs["best_logs"]["step"]]["validate_every"]

    for set in [{"score":train_f1_perclass,"label":"Training"}, {"score":val_f1_perclass,"label":"Validation"}]:
        plt.figure()
        score_min = 100
        score_max = 0
        colors = ["b", "k", "r"]
        color_dict = {v: colors[i] for i, v in enumerate(list_predictors)}

        for pred_key in list_predictors:
            plt.plot(steps, set["score"][pred_key][:, 0], color = color_dict[pred_key], label="{}".format(pred_key), linewidth=0.8)
            plt.plot(steps, set["score"][pred_key][:, 1], color = color_dict[pred_key], linewidth=0.8)
            plt.plot(steps, set["score"][pred_key][:, 2], color = color_dict[pred_key], linewidth=0.8)
            plt.plot(steps, set["score"][pred_key][:, 3], color = color_dict[pred_key], linewidth=0.8)
            plt.plot(steps, set["score"][pred_key][:, 4], color = color_dict[pred_key], linewidth=0.8)

            score_min = np.minimum(score_min, np.min(set["score"][pred_key]))
            score_max = np.maximum(score_max, np.max(set["score"][pred_key]))
            if set["label"] == "Validation":
                for i in range(5):
                    best_loss = logs["best_logs"]["val_perclassf1"][pred_key][i]
                    plt.plot((best_step, best_step), (0, best_loss), linestyle="--", color="y", linewidth=0.6)
                    plt.plot((0, best_step), (best_loss, best_loss), linestyle="--", color="y", linewidth=0.6)
            else:
                plt.plot((best_step, best_step), (0, score_max), linestyle="--", color="y", linewidth=0.6)

        plt.plot((0, steps[-1]), (0.8, 0.8), linestyle="--", linewidth=0.4, color="k")
        plt.plot((0, steps[-1]), (0.85, 0.85), linestyle="--", linewidth=0.4, color="k")
        plt.plot((0, steps[-1]), (0.9, 0.9), linestyle="--", linewidth=0.4, color="k")
        plt.plot((0, steps[-1]), (0.95, 0.95), linestyle="--", linewidth=0.4, color="k")

        plt.xlabel('Steps')
        plt.ylabel('F1 Value')
        plt.title("F1 Multi Predictors on {}".format(set["label"]))
        plt.yticks([0.4,0.45,0.5,0.55,0.8,0.85,0.9,0.95])
        plt.ylim([score_min-0.05,score_max+0.05])
        plt.legend()
        # plt.savefig("/users/sista/kkontras/Documents/Sleep_Project/data/2021_data/loss.png")
        plt.show()

def sleep_plot_f1_perclass(config, logs):

    val_f1 = np.array([logs["val_logs"][i]["val_perclassf1"] for i in logs["val_logs"]])
    steps = np.array([i/logs["train_logs"][i]["validate_every"] for i in logs["train_logs"]])-1

    plt.figure()
    plt.plot(steps, val_f1[:,0], label="Wake",linewidth=0.8)
    plt.plot(steps, val_f1[:,1], label="N1",linewidth=0.8)
    plt.plot(steps, val_f1[:,2], label="N2",linewidth=0.8)
    plt.plot(steps, val_f1[:,3], label="N3",linewidth=0.8)
    plt.plot(steps, val_f1[:,4], label="REM",linewidth=0.8)

    f1_min = np.min(val_f1)
    f1_max = np.max(val_f1)
    best_step = logs["best_logs"]["step"]/logs["train_logs"][logs["best_logs"]["step"]]["validate_every"]-1
    best_f1 = logs["best_logs"]["val_f1"]

    plt.plot((best_step, best_step), (0, f1_max+0.05), linestyle="--", color="y", label="Chosen Point")
    plt.plot((0, steps[-1]), (0.8, 0.8), linestyle="--", linewidth=0.4, color="k")
    plt.plot((0, steps[-1]), (0.85, 0.85), linestyle="--", linewidth=0.4, color="k")
    plt.plot((0, steps[-1]), (0.9, 0.9), linestyle="--", linewidth=0.4, color="k")
    plt.plot((0, steps[-1]), (0.95, 0.95), linestyle="--", linewidth=0.4, color="k")
    # plt.plot((0, best_step), (best_f1, best_f1), linestyle="--", color="b")

    plt.xlabel('Steps')
    plt.ylabel('F1')
    plt.title("Validation F1 per class")

    plt.ylim([f1_min-0.05,1.01])
    plt.legend()
    # plt.savefig("/users/sista/kkontras/Documents/Sleep_Project/data/2021_data/f1_perclass.png")
    plt.show()

def plot_comparisons(f1_comparisons, k_comparisons, entropy_comparisons, config_list):

    k_comparisons = {config_list[0]:[k_comparisons[config_list[0]][i] for i in k_comparisons[config_list[0]].keys()],
               config_list[1]:[k_comparisons[config_list[1]][i] for i in k_comparisons[config_list[1]].keys()]}

    f1_comparisons = {config_list[0]:[f1_comparisons[config_list[0]][i] for i in f1_comparisons[config_list[0]].keys()],
               config_list[1]:[f1_comparisons[config_list[1]][i] for i in f1_comparisons[config_list[1]].keys()]}

    entropy_comparisons = {config_list[0]:[entropy_comparisons[config_list[0]][i] for i in entropy_comparisons[config_list[0]].keys()],
               config_list[1]:[entropy_comparisons[config_list[1]][i] for i in entropy_comparisons[config_list[1]].keys()]}

    patient_sortargs = np.argsort(k_comparisons[config_list[0]])

    entropy_comparisons[config_list[0]] = [entropy_comparisons[config_list[0]][i] for i in patient_sortargs]
    entropy_comparisons[config_list[1]] = [entropy_comparisons[config_list[1]][i] for i in patient_sortargs]

    f1_comparisons[config_list[0]] = [f1_comparisons[config_list[0]][i] for i in patient_sortargs]
    f1_comparisons[config_list[1]] = [f1_comparisons[config_list[1]][i] for i in patient_sortargs]

    k_comparisons[config_list[0]] = [k_comparisons[config_list[0]][i] for i in patient_sortargs]
    k_comparisons[config_list[1]] = [k_comparisons[config_list[1]][i] for i in patient_sortargs]


    colors = np.array(f1_comparisons[config_list[0]]) > np.array(f1_comparisons[config_list[1]])
    colors = ["orange" if i else "lightblue" for i in colors]
    plt.figure(figsize=(25, 5))
    x = np.linspace(0, len(f1_comparisons[config_list[0]]) - 1, len(f1_comparisons[config_list[0]]))
    plt.xlabel("Patient")
    plt.ylabel("F1")
    plt.title("F1 Comparison")
    plt.plot(x, f1_comparisons[config_list[0]], 'o', color='orange', label="EEG")
    plt.plot(x, f1_comparisons[config_list[1]], 'o', color='lightblue', label="EEG-EOG")
    # for i in range(len(x)):
    #     plt.axvline(x[i], 0, 1, color=colors[i])
    plt.legend()
    plt.show()

    colors = np.array(k_comparisons[config_list[0]]) > np.array(k_comparisons[config_list[1]])
    colors = ["orange" if i else "lightblue" for i in colors]
    plt.figure(figsize=(25, 5))
    x = np.linspace(0, len(k_comparisons[config_list[0]]) - 1, len(k_comparisons[config_list[0]]))
    plt.xlabel("Patient")
    plt.ylabel("Cohens Kappa")
    plt.title("K Comparison")
    plt.plot(x, k_comparisons[config_list[0]], 'o', color='orange', label="EEG")
    plt.plot(x, k_comparisons[config_list[1]], 'o', color='lightblue', label="EEG-EOG")
    # for i in range(len(x)):
    #     plt.axvline(x[i], 0, 1, color=colors[i])
    plt.legend()
    plt.show()

    colors = np.array(entropy_comparisons[config_list[0]]) > np.array(entropy_comparisons[config_list[1]])
    colors = ["orange" if i else "lightblue" for i in colors]
    plt.figure(figsize=(25, 5))
    x = np.linspace(0, len(entropy_comparisons[config_list[0]]) - 1, len(entropy_comparisons[config_list[0]]))
    plt.xlabel("Patient")
    plt.ylabel("Entropy")
    plt.title("Entropy Comparison")
    plt.plot(x, entropy_comparisons[config_list[0]], 'o', color='orange', label="EEG")
    plt.plot(x, entropy_comparisons[config_list[1]], 'o', color='lightblue', label="EEG-EOG")
    # for i in range(len(x)):
    #     plt.axvline(x[i], 0, 1, color=colors[i])

    plt.legend()
    plt.show()

def load_models(config, device, checkpoint, only_model=False):

    model_class = globals()[config.model.model_class]
    # config.pretrainedEncoder = [False]
    enc = sleep_load_encoder(encoders=config.model.encoders)
    model = model_class(enc, args = config.model.args)
    # model = model.to('cpu')
    # model = nn.DataParallel(model, device_ids='cpu')
    model = model.to(device)
    model = nn.DataParallel(model, device_ids=[torch.device(i) for i in config.training_params.gpu_device])

    #
    if only_model:
        return model

    # config.pretrainedEncoder = [True]
    # enc = sleep_load_encoder(encoder_models=config.encoder_models,pretrainedEncoder=config.pretrainedEncoder,save_dir_encoder=config.savetrainedEncoder)
    # best_model = model_class(enc, channel = config.channel)
    # best_model = best_model.to(device)
    # best_model = nn.DataParallel(best_model, device_ids=[torch.device(i) for i in config.gpu_device])

    best_model = copy.deepcopy(model)
    # best_model = best_model.to('cpu')
    # best_model = nn.DataParallel(best_model, device_ids='cpu')
    model.load_state_dict(checkpoint["model_state_dict"])
    best_model.load_state_dict(checkpoint["best_model_state_dict"])

    return model, best_model

def print_trained_files():
    filename = "/esat/smcdata/users/kkontras/Image_Dataset/no_backup/data/2021_data/shhs1/single_channel"
    for path, subdirs, files in os.walk(filename):
        for name in files:
            f_name = os.path.join(path, name)
            if "cp" not in f_name and "contrastive" not in f_name:
                print(f_name)
                checkpoint = torch.load(f_name, map_location="cpu")
                logs = checkpoint['logs']
                print("-- Best Validation --")
                print(logs["best_logs"])


config_list = [
    # "./configs/neonatal/fourier_transformer_eeg_mat.json",
    # "./configs/neonatal/fourier_transformer_eeg_mat.json",
    # "./configs/shhs/single_channel/replicate.json",
    # "./configs/shhs/single_channel/replicate_nosch.json",
    # "./configs/shhs/single_channel/fourier_transformer_eeg_mat_contrastive.json",
    # "./configs/shhs/contrastive_training/contrastive_pretraining.json",
    # "./configs/shhs/single_channel/fourier_transformer_eeg_mat.json",
    # "./configs/shhs/single_channel/fourier_transformer_eog_mat.json",
    # "./configs/shhs/single_channel/fourier_transformer_eeg_mat_ch2s.json",
    # "./configs/shhs/single_channel/fourier_transformer_eeg_mat_w.json",
    # "./configs/shhs/single_channel/fourier_transformer_eog_mat.json",
    # "./configs/shhs/single_channel/fourier_transformer_eog_mat_ch2.json",
    # "./configs/shhs/single_channel/fourier_transformer_emg_mat.json",
    # "./configs/shhs/multi_channel/fourier_transformer_eeg_mat.json",
    # "./configs/shhs/multi_channel/fourier_transformer_eog_mat.json",
    # "./configs/shhs/multi_channel/fourier_transformer_eeg_eog_emg_mat_concat.json",
    # "./configs/shhs/multi_channel/fourier_transformer_eeg_eog_emg_mat_merged.json",
    # "./configs/shhs/multi_channel/fourier_transformer_eeg_eog_emg_mat_late.json",
    # "./configs/shhs/multi_channel/fourier_transformer_eeg_eog_emg_mat_late_late.json",
    # "./configs/shhs/multi_channel/fourier_transformer_eeg_eog_emg_mat_merged_with_diff_FC.json",
    # "./configs/shhs/multi_channel/fourier_transformer_eeg_eog_emg_mat_merged_io_with_diff_FC.json",

    # "./configs/shhs/multi_channel/fourier_transformer_eeg_eog_mat_bottleneck.json",
    # "./configs/shhs/multi_channel/fourier_transformer_eeg_eog_mat_merged_outermod.json",
    # "./configs/shhs/multi_channel/fourier_transformer_eeg_eog_mat_merged_with_diff_FC.json",
    # "./configs/shhs/multi_channel/fourier_transformer_eeg_eog_mat_merged_io_with_diff_FC.json",
    # "./configs/shhs/multi_channel/fourier_transformer_eeg_eog_mat_merged.json",
    # "./configs/shhs/multi_channel/fourier_transformer_eeg_eog_mat_concat.json",
    # "./configs/shhs/multi_channel/fourier_transformer_eeg_eog_mat_late.json",
    # "./configs/shhs/single_channel/replicate_avg.json",
    # "./configs/shhs/single_channel/fourier_transformer_cls_eeg_mat.json",
    # "./configs/shhs/single_channel/fourier_transformer_cls_emg_mat.json",
    # "./configs/shhs/single_channel/fourier_transformer_cls_emg_mat.json",
    # "./configs/shhs/single_channel/replicate_avg_vnov.json",

    # "./configs/shhs/single_channel/fourier_transformer_cls_eeg_mat.json",
    "./configs/shhs/single_channel/fourier_transformer_cls_eeg_mat_adv.json",
    # "./configs/shhs/single_channel/fourier_transformer_cls_eeg_mat_vdec.json",
    # "./configs/shhs/single_channel/fourier_transformer_cls_eeg_mat_noadv.json",
    # "./configs/shhs/single_channel/fourier_transformer_cls_eog_mat.json",
    # "./configs/shhs/single_channel/fourier_transformer_cls_emg_mat.json",
    # "./configs/shhs/multi_modal/eeg_eog/fourier_transformer_eeg_eog_mat_concat.json",
    # "./configs/shhs/multi_modal/eeg_eog/fourier_transformer_eeg_eog_mat_merged.json",
    # "./configs/shhs/multi_modal/eeg_eog/fourier_transformer_eeg_eog_mat_merged_rpos.json",

    # "./configs/shhs/multi_modal/eeg_eog/fourier_transformer_eeg_eog_mat_late.json",
    # "./configs/shhs/multi_modal/eeg_eog/fourier_transformer_eeg_eog_mat_late_late.json",
    # "./configs/shhs/multi_modal/eeg_eog_emg/fourier_transformer_eeg_eog_emg_mat_merged.json",
    # "./configs/shhs/multi_modal/eeg_eog_emg/fourier_transformer_eeg_eog_emg_mat_late.json",

    # "./configs/shhs/single_channel/fourier_transformer_cls_eeg_mat_RA.json",
    # "./configs/shhs/single_channel/fourier_transformer_cls_eeg_mat_Sparse_RA.json",

    # "./configs/shhs/single_channel/fourier_transformer_cls_eog_mat_RA.json",
    # "./configs/shhs/multi_modal/eeg_eog/fourier_transformer_eeg_eog_mat_concat_l1_RA.json",
    # "./configs/shhs/multi_modal/eeg_eog/fourier_transformer_eeg_eog_mat_concat_l1_h2_RA.json",
    # "./configs/shhs/multi_modal/eeg_eog/fourier_transformer_eeg_eog_mat_merged_RA.json",
    # "./configs/shhs/multi_modal/eeg_eog/fourier_transformer_eeg_eog_mat_merged_l1_RA.json",
    # "./configs/shhs/multi_modal/eeg_eog/fourier_transformer_eeg_eog_mat_late_l1_RA.json",
    # "./configs/shhs/multi_modal/eeg_eog/fourier_transformer_eeg_eog_mat_late_late_l1_RA.json",
    # "./configs/shhs/multi_modal/eeg_eog/bottleneck/fourier_transformer_eeg_eog_mat_bottleneck_l1_c1_RA.json",
    # "./configs/shhs/multi_modal/eeg_eog/bottleneck/fourier_transformer_eeg_eog_mat_bottleneck_l1_c5_RA.json",

    # "./configs/shhs/multi_modal/eeg_eog/established_models/fourier_transformer_eeg_eog_mat_merged.json",
    # "./configs/shhs/multi_modal/eeg_eog/established_models/fourier_transformer_eeg_eog_mat_BIOBLIP.json",
    # "./configs/shhs/multi_modal/eeg_eog/established_models/fourier_transformer_eeg_eog_mat_BIOBLIP_test.json",
    # "./configs/shhs/multi_modal/eeg_eog/established_models/fourier_transformer_eeg_eog_mat_BIOBLIP_lossw.json",
    # "./configs/shhs/multi_modal/eeg_eog/established_models/fourier_transformer_eeg_eog_mat_BIOBLIP_lossw_masked.json",
    # "./configs/shhs/multi_modal/eeg_eog/established_models/fourier_transformer_eeg_eog_mat_BIOBLIP_lossw_frozen_LE.json",
    # "./configs/shhs/multi_modal/eeg_eog/established_models/fourier_transformer_eeg_eog_mat_BIOBLIP_order.json",
    # "./configs/shhs/multi_modal/eeg_eog/established_models/fourier_transformer_eeg_eog_mat_BIOBLIP_twote.json",
    # "./configs/shhs/multi_modal/eeg_eog/established_models/fourier_transformer_eeg_eog_mat_BIOBLIP_adv.json",
    # "./configs/shhs/multi_modal/eeg_eog/established_models/fourier_transformer_eeg_eog_mat_BIOBLIP_outer.json",
    # "./configs/shhs/multi_modal/eeg_eog/established_models/fourier_transformer_eeg_eog_mat_bottleneck_lim0.json",
    # "./configs/shhs/multi_modal/eeg_eog/established_models/fourier_transformer_eeg_eog_mat_late_glearnedbiasedm_outerplus.json",
    # "./configs/shhs/multi_modal/eeg_eog/established_models/fourier_transformer_eeg_eog_mat_merged.json",
    # "./configs/shhs/multi_modal/eeg_eog/established_models/fourier_transformer_eeg_eog_mat_οbottleneck.json",
    # "./configs/shhs/multi_modal/eeg_eog/established_models/fourier_transformer_eeg_eog_mat_BIOBLIP_rpos_simple.json",
    # "./configs/shhs/multi_modal/eeg_eog/established_models/fourier_transformer_eeg_eog_mat_BIOBLIP_outer_rpos_simple.json",
    # "./configs/shhs/multi_modal/eeg_eog/established_models/fourier_transformer_eeg_eog_mat_BIOBLIP_double.json",
    # "./configs/shhs/multi_modal/eeg_eog/established_models/fourier_transformer_eeg_eog_mat_BIOBLIP_shared.json",

    # "./configs/shhs/multi_modal/eeg_eog/established_models/fourier_transformer_eeg_eog_mat_BIOBLIP_rpos_adv_simple.json",
    # "./configs/shhs/multi_modal/eeg_eog/established_models/fourier_transformer_eeg_eog_mat_BIOBLIP_rpos_simple.json",

    # "./configs/shhs/multi_modal/eeg_eog/established_models/fourier_transformer_eeg_eog_mat_late_glearnedbiasedm_outerplus_rpos_adv.json",
    # "./configs/shhs/multi_modal/eeg_eog/established_models/fourier_transformer_eeg_eog_mat_BIOBLIP_rpos_adv.json",
    # "./configs/shhs/multi_modal/eeg_eog/established_models/fourier_transformer_eeg_eog_mat_merged_rpos_adv.json",
    # "./configs/shhs/multi_modal/eeg_eog/established_models/fourier_transformer_eeg_eog_mat_bottleneck_lim0_rpos_adv.json",
    # "./configs/shhs/multi_modal/eeg_eog/established_models/fourier_transformer_eeg_eog_mat_BIOBLIP_rpos_adv_neigh.json",

    # "./configs/shhs/multi_modal/eeg_eog/established_models/fourier_transformer_eeg_eog_mat_COCA_sep_multisupervised.json"
    # "./configs/shhs/multi_modal/eeg_eog/established_models/fourier_transformer_eeg_eog_mat_sep_multisupervised.json"
    # "./configs/shhs/multi_modal/eeg_eog/established_models/fourier_transformer_eeg_eog_mat_sep_combined_multisupervised.json"

    # "./configs/shhs/multi_modal/eeg_eog/established_models/fourier_transformer_eeg_eog_mat_BIOBLIP_lossw_pretrainedNCH_onlyalign.json",
    # "./configs/nch/multi_modal/established_models/fourier_transformer_eeg_eog_mat_BIOBLIP_lossw.json",

    # "./configs/shhs/myprepro/multi_modal/eeg_eog/fourier_transformer_eeg_eog_mat_merged_rpos_adv.json",
    # "./configs/shhs/single_channel/fourier_transformer_cls_eeg_mat_adv_perrecording.json",

    # "./configs/shhs2/fourier_transformer_eeg_eog_mat_BIOBLIP.json"

    # "./configs/shhs/single_channel/fourier_transformer_cls_eeg_mat_adv.json",
    # "./configs/shhs/single_channel/fourier_transformer_cls_eog_mat.json",

    # "./configs/shhs/single_channel/time_cnn_transformer_eeg.json",
    # "./configs/shhs/single_channel/time_cnn_eeg.json",
    # "./configs/shhs/single_channel/time_cnn_eog.json",
    # "./configs/shhs/single_channel/time_cnn_emg.json",
    # "./configs/shhs/multi_modal/eeg_eog/time_cnn_late_eeg_eog.json",
    # "./configs/shhs/multi_modal/eeg_eog/time_cnn_mid_eeg_eog.json",
    # "./configs/shhs/multi_modal/eeg_eog/time_cnn_early_eeg_eog.json",
    # # "./configs/shhs/multi_modal/eeg_eog/time_cnn_late_late_eeg_eog.json",
    # "./configs/shhs/multi_modal/eeg_eog/time_cnn_mid_shared_eeg_eog.json",
    #
    # "./configs/shhs/multi_modal/eeg_eog/bottleneck/fourier_transformer_eeg_eog_mat_bottleneck_c5.json",
    # "./configs/shhs/multi_modal/eeg_eog/bottleneck/fourier_transformer_eeg_eog_mat_bottleneck_lim0.json",
    # "./configs/shhs/multi_modal/eeg_eog/bottleneck/fourier_transformer_eeg_eog_mat_bottleneck_lim1_c1.json",
    # "./configs/shhs/multi_modal/eeg_eog/bottleneck/fourier_transformer_eeg_eog_mat_bottleneck_lim2_c5.json",
    # "./configs/shhs/multi_modal/eeg_eog/bottleneck/fourier_transformer_eeg_eog_mat_bottleneck_lim2_c1.json",
    # "./configs/shhs/multi_modal/eeg_eog/bottleneck/fourier_transformer_eeg_eog_mat_ibottleneck_lim2_c1.json",
    # "./configs/shhs/multi_modal/eeg_eog/bottleneck/fourier_transformer_eeg_eog_mat_obottleneck_lim2_c1.json",
    # "./configs/shhs/multi_modal/eeg_eog/bottleneck/fourier_transformer_eeg_eog_mat_οbottleneck.json",
    # "./configs/shhs/multi_modal/eeg_eog/bottleneck/fourier_transformer_eeg_eog_mat_ibottleneck_merged.json",
    # "./configs/shhs/multi_modal/eeg_eog_emg/bottleneck/fourier_transformer_eeg_eog_emg_mat_bottleneck.json",


    # "./configs/shhs/multi_modal/eeg_eog/suppmod/fourier_transformer_eeg_eog_mat_suppmod_lim2_c5.json",

    # "./configs/shhs/multi_modal/eeg_eog/dex/fourier_transformer_eeg_eog_mat_real_dex.json",
    # "./configs/shhs/multi_modal/eeg_eog/dex/fourier_transformer_eeg_eog_mat_dex.json",
    # "./configs/shhs/multi_modal/eeg_eog/dex/fourier_transformer_eeg_eog_mat_dex_pcommon.json",

    # "./configs/shhs/contrastive_training/contrastive_pretraining.json",

    # "./configs/sleep_edf/transformers/sleeptransformer_cls.json",
    # "./configs/sleep_edf/transformers/sleeptransformer_cls.json",
    # "./configs/sleep_edf/transformers/sleeptransformer_cls.json",
    # "./configs/sleep_edf/transformers/sleeptransformer_cls.json",

    # "./configs/shhs/multi_modal/eeg_eog/contrastive_pre/fourier_transformer_eeg_eog_mat_late.json",
    # "./configs/shhs/multi_modal/eeg_eog/contrastive_pre/fourier_transformer_eeg_eog_mat_late_rpos.json",
    # "./configs/shhs/multi_modal/eeg_eog/contrastive_pre/fourier_transformer_eeg_eog_mat_late_moddrop.json",
    # "./configs/shhs/multi_modal/eeg_eog/contrastive_pre/fourier_transformer_eeg_eog_mat_late_concat_pos.json",
    # "./configs/shhs/multi_modal/eeg_eog/contrastive_pre/fourier_transformer_eeg_eog_mat_late_only_inner.json",
    # "./configs/shhs/multi_modal/eeg_eog/contrastive_pre/fourier_transformer_eeg_eog_mat_late_no_pos.json",
    # "./configs/shhs/multi_modal/eeg_eog/contrastive_pre/fourier_transformer_eeg_eog_mat_late_learnable_pos.json",
    # "./configs/shhs/multi_modal/eeg_eog/contrastive_pre/fourier_transformer_eeg_eog_mat_late_randshuffle.json",
    # "./configs/shhs/multi_modal/eeg_eog/contrastive_pre/fourier_transformer_eeg_eog_mat_late_every.json",
    # "./configs/shhs/multi_modal/eeg_eog/contrastive_pre/fourier_transformer_eeg_eog_mat_late_gbiased.json",
    # "./configs/shhs/multi_modal/eeg_eog/contrastive_pre/fourier_transformer_eeg_eog_mat_late_gbiasedm_temp10.json",

    # "./configs/shhs/multi_modal/eeg_eog/contrastive_pre/fourier_transformer_eeg_eog_mat_late_gbiasedm.json",
    # "./configs/shhs/multi_modal/eeg_eog/contrastive_pre/fourier_transformer_eeg_eog_mat_late_gbiasedm_plus.json",
    # "./configs/shhs/multi_modal/eeg_eog/contrastive_pre/fourier_transformer_eeg_eog_mat_late_neighbiasedm5.json",
    # "./configs/shhs/multi_modal/eeg_eog/contrastive_pre/fourier_transformer_eeg_eog_mat_late_neighbiasedm3.json",
    # "./configs/shhs/multi_modal/eeg_eog/contrastive_pre/fourier_transformer_eeg_eog_mat_late_neighbiasedm5.json",
    # "./configs/shhs/multi_modal/eeg_eog/contrastive_pre/fourier_transformer_eeg_eog_mat_late_neighbiasedm3_plus.json",
    # "./configs/shhs/multi_modal/eeg_eog/contrastive_pre/fourier_transformer_eeg_eog_mat_late_neighbiasedm5_outerplus.json",
    # "./configs/shhs/multi_modal/eeg_eog/contrastive_pre/fourier_transformer_eeg_eog_mat_late_glearnedbiasedm.json",
    # "./configs/shhs/multi_modal/eeg_eog/contrastive_pre/fourier_transformer_eeg_eog_mat_late_glearnedbiasedm_plus.json",
    # "./configs/shhs/multi_modal/eeg_eog/contrastive_pre/fourier_transformer_eeg_eog_mat_late_glearnedbiasedm_outerplus.json",

    # "./configs/shhs/single_channel/fourier_transformer_eeg_mat_rpos_glearnedbiasedm_outer_plus.json",
    # "./configs/shhs/single_channel/fourier_transformer_eeg_mat_rpos_glearnedbiasednodiagm_outer_plus.json",
    # "./configs/shhs/single_channel/fourier_transformer_lstm_eeg_mat_rpos.json",
    # "./configs/shhs/single_channel/fourier_transformer_intlstm_eeg_mat_rpos.json",
    # "./configs/shhs/single_channel/fourier_transformer_io_intlstm_eeg_mat_rpos.json",

    # "./configs/shhs/multi_modal/eeg_eog/bottleneck/fourier_transformer_eeg_eog_mat_contextproc_RA.json",
    # "./configs/shhs/multi_modal/eeg_eog/bottleneck/fourier_transformer_eeg_eog_mat_contextproc_rpos_RA.json",

    # "./configs/shhs/tempshuffle/fourier_transformer_cls_eeg_mat_tempshuffle_pretrained.json",

    # "./configs/shhs/multi_modal/eeg_eog/fourier_transformer_eeg_eog_mat_merged_v2.json",

    # "./configs/shhs/tempshuffle/fourier_transformer_cls_eeg_mat_tempshuffle_p5.json",
    # "./configs/shhs/tempshuffle/fourier_transformer_cls_eeg_mat_no_pretrained_benchmark_5.json",
    # "./configs/shhs/tempshuffle/fourier_transformer_cls_eeg_mat_tempshuffle_p10.json",
    # "./configs/shhs/tempshuffle/fourier_transformer_cls_eeg_mat_no_pretrained_benchmark_10.json",
    # "./configs/shhs/tempshuffle/fourier_transformer_cls_eeg_mat_tempshuffle_p100.json",
    # "./configs/shhs/tempshuffle/fourier_transformer_cls_eeg_mat_no_pretrained_benchmark_100.json",
    # "./configs/shhs/tempshuffle/fourier_transformer_cls_eeg_mat_tempshuffle_p500.json",
    # "./configs/shhs/tempshuffle/fourier_transformer_cls_eeg_mat_no_pretrained_benchmark_500.json",
    # "./configs/shhs/tempshuffle/fourier_transformer_cls_eeg_mat_tempshuffle_pfull.json",
    # "./configs/shhs/tempshuffle/fourier_transformer_cls_eeg_mat_no_pretrained_benchmark.json",

    # "./configs/shhs/tempshuffle/fourier_transformer_cls_eeg_mat_tempshuffle_order_pretrained.json",
    # "./configs/shhs/tempshuffle/fourier_transformer_cls_eeg_mat_tempshuffle_order_pretrained_10.json",
    # "./configs/shhs/tempshuffle/fourier_transformer_cls_eeg_mat_tempshuffle_order_pretrainedinner_10.json",
    # "./configs/shhs/tempshuffle/fourier_transformer_cls_eeg_mat_tempshuffle_order_pretrained_100.json",
    # "./configs/shhs/tempshuffle/fourier_transformer_cls_eeg_mat_tempshuffle_order_pretrained_500.json",

    # "./configs/shhs/single_channel/fourier_transformer_eeg_connepoch.json",
    # "./configs/shhs/single_channel/fourier_transformer_eeg_decoder.json",
    # "./configs/shhs/multi_modal/eeg_eog/fourier_transformer_eeg_eog_mat_merged_HPFC.json",
    # "./configs/shhs/multi_modal/eeg_eog/fourier_transformer_eeg_eog_mat_locglob.json",
    # "./configs/shhs/multi_modal/eeg_eog/fourier_transformer_eeg_eog_mat_BIOBLIP.json",
    # "./configs/shhs/multi_modal/eeg_eog/fourier_transformer_eeg_eog_mat_BIOBLIP_outer.json",

    # "./configs/shhs/border/border_pretraining.json",

    # "./configs/nch/single_channel/fourier_transformer_eeg_mat_emphasisonN1.json",
    # "./configs/nch/single_channel/fourier_transformer_eog_mat_emphasisonN1.json",
    # "./configs/nch/single_channel/fourier_transformer_eog_mat_emphasisonN1.json",
    # "./configs/nch/single_channel/fourier_transformer_emg_mat_emphasisonN1.json",
    # "./configs/nch/single_channel/fourier_transformer_eeg_mat.json",
    # "./configs/nch/single_channel/fourier_transformer_eeg_mat_tempw.json",
    # "./configs/nch/single_channel/fourier_transformer_eeg_mat_pretrained.json",
    # "./configs/nch/single_channel/fourier_transformer_eog_mat.json",
    # "./configs/nch/multi_modal/fourier_transformer_eeg_eog_merged_mat_tempw.json",
    # "./configs/nch/multi_modal/fourier_transformer_eeg_eog_merged_mat.json",
    # "./configs/nch/multi_modal/fourier_transformer_eeg_eog_late_mat.json",
    # "./configs/nch/multi_modal/fourier_transformer_eeg_eog_bottleneck_mat.json",
    # "./configs/nch/multi_modal/fourier_transformer_eeg_eog_merged_channels_mat.json",
    # "./configs/nch/multi_channel/fourier_transformer_multichannel_eeg.json",
    # "./configs/nch/multi_channel/fourier_transformer_multichannel_eeg_w.json",
    # "./configs/nch/multi_channel/fourier_transformer_multichannel_eog.json",
    # "./configs/nch/multi_channel/fourier_transformer_multichannel_eog_w.json",
    # "./configs/nch/multi_channel/fourier_transformer_multichannel_eog_w_v2.json"
    # "./configs/nch/multi_modal/fourier_transformer_eeg_eog_BIOBLIP.json",
    # "./configs/nch/multi_modal/fourier_transformer_eeg_eog_BIOBLIP_rpos_adv.json"
    # "./configs/nch/multi_modal/fourier_transformer_eeg_eog_BIOBLIP_rpos_adv.json"


    # "./configs/nch/multi_modal/established_models/fourier_transformer_eeg_eog_mat_BIOBLIP_rpos_adv.json",
    # "./configs/nch/multi_modal/established_models/fourier_transformer_eeg_eog_mat_BIOBLIP_rpos_adv_temp.json",
    # "./configs/nch/multi_modal/established_models/fourier_transformer_eeg_eog_mat_BIOBLIP_rpos_adv_neigh.json",
    # "./configs/nch/multi_modal/established_models/fourier_transformer_eeg_eog_mat_merged_rpos_adv.json",
    # "./configs/nch/multi_modal/established_models/fourier_transformer_eeg_eog_mat_late_glearnedbiasedm_outerplus_rpos_adv.json",
    # "./configs/nch/multi_modal/established_models/fourier_transformer_eeg_eog_mat_bottleneck_lim0_rpos_adv.json",

    # "./configs/nch/multi_modal/established_models/fourier_transformer_eeg_eog_mat_BIOBLIP.json",
    # "./configs/nch/multi_modal/established_models/fourier_transformer_eeg_eog_mat_BIOBLIP_rpos_adv.json",
    # "./configs/nch/multi_modal/established_models/fourier_transformer_eeg_eog_mat_BIOBLIP_rpos_adv_shared.json",
    # "./configs/nch/multi_modal/established_models/fourier_transformer_eeg_eog_mat_BIOBLIP_rpos_adv_neigh.json",
    # "./configs/nch/multi_modal/established_models/fourier_transformer_eeg_eog_mat_BIOBLIP_rpos_adv_temp.json",
    # "./configs/nch/multi_modal/established_models/fourier_transformer_eeg_eog_mat_BIOBLIP_rpos_adv_temp.json",
    # "./configs/nch/multi_modal/established_models/fourier_transformer_eeg_eog_mat_BIOBLIP_rpos_adv_shared_temp.json",
    # "./configs/nch/multi_modal/established_models/fourier_transformer_eeg_eog_mat_bottleneck_lim0_rpos_adv.json",
    # "./configs/nch/multi_modal/established_models/fourier_transformer_eeg_eog_mat_late_glearnedbiasedm_outerplus_rpos_adv.json",
    # "./configs/nch/multi_modal/established_models/fourier_transformer_eeg_eog_mat_merged_rpos_adv.json",
    # "./configs/nch/multi_modal/established_models/fourier_transformer_eeg_eog_mat_merged_rpos_adv_temp.json",
    # "./configs/nch/multi_modal/established_models/fourier_transformer_eeg_eog_mat_late_rpos_adv_temp.json",
    # "./configs/nch/multi_modal/fourier_transformer_eeg_eog_merged_mat_tempw.json",

    # "./configs/nch/multi_modal/established_models/fourier_transformer_eeg_eog_mat_BIOBLIP_rpos_adv_temp.json",
    # "./configs/nch/multi_modal/established_models/fourier_transformer_eeg_eog_mat_bottleneck_lim0_rpos_adv.json",
    # "./configs/nch/multi_modal/established_models/fourier_transformer_eeg_eog_mat_late_glearnedbiasedm_outerplus_rpos_adv_temp.json",
    # "./configs/nch/multi_modal/established_models/fourier_transformer_eeg_eog_mat_merged_rpos_adv_temp.json",



    # "./configs/sleep_edf/multi_modal/eeg_eog/fourier_transformer_eeg_eog_mat_BIOBLIP.json",
    # "./configs/sleep_edf/multi_modal/eeg_eog/fourier_transformer_eeg_eog_mat_BIOBLIP_shared.json",
    # "./configs/sleep_edf/multi_modal/eeg_eog/fourier_transformer_eeg_eog_mat_merged.json",
    # "./configs/sleep_edf/multi_modal/eeg_eog/fourier_transformer_eeg_eog_mat_late.json",
    # "./configs/sleep_edf/single_channel/fourier_transformer_eeg_mat.json",
    # "./configs/sleep_edf/single_channel/fourier_transformer_eog_mat.json",

    # "./configs/shhs/single_channel/fourier_transformer_eeg_tf_rn.json",
    # "./configs/shhs/single_channel/fourier_transformer_cls_eeg_mat.json",
    # "./configs/shhs/single_channel/fourier_transformer_cls_eeg_overparam.json",
    # "./configs/shhs/single_channel/fourier_transformer_cls_eeg_conv.json",

    # "./configs/shhs/multi_modal/eeg_eog/contrastive_pre/fourier_transformer_eeg_eog_mat_late_neighbiasedm_bigseql.json",
    # "./configs/shhs/multi_modal/eeg_eog/contrastive_pre/fourier_transformer_eeg_eog_mat_late_neighbiasedm5_bigseql.json",
    # "./configs/shhs/multi_modal/eeg_eog/contrastive_pre/fourier_transformer_eeg_eog_mat_late_gbiased_diag.json",
    # "./configs/shhs/multi_modal/eeg_eog/contrastive_pre/fourier_transformer_eeg_eog_mat_rep_bottleneck_gbiased.json",
    # "./configs/shhs/multi_modal/eeg_eog/contrastive_pre/fourier_transformer_eeg_eog_mat_merged_gbiased.json",
    # "./configs/shhs/multi_modal/eeg_eog/contrastive_pre/fourier_transformer_eeg_eog_mat_merged_fc_gbiased.json",
    # "./configs/shhs/multi_modal/eeg_eog/contrastive_pre/fourier_transformer_eeg_eog_mat_late_hf_norm.json",
    # "./configs/shhs/multi_modal/eeg_eog/contrastive_pre/fourier_transformer_eeg_eog_mat_merged_hf_norm.json",
    # "./configs/shhs/multi_modal/eeg_eog/contrastive_pre/fourier_transformer_eeg_eog_mat_late_randshuffle_b.json",

]

f1_comparisons = {}
k_comparisons = {}
entropy_comparisons = {}
multi_fold_results = {}
for i, config_name in enumerate(config_list):
    for fold in range(1):

        config = process_config(config_name, False)

        # config.seq_length = [21,0]
        # config.batch_size = 32
        # config.test_batch_size = 32
        # config.model.save_dir = config.model.save_dir.format(fold)

        # config.huy_data= False
        # config.data_roots = "/esat/smcdata/users/kkontras/Image_Dataset/no_backup/Sleep_SHHS/Version_1/train/"
        # config.data_view_dir = [
        #     {"list_dir": "stft_eeg_file_map.txt", "data_type": "stft", "mod": "eeg", "num_ch": 1},
        #     # {"list_dir": "stft_eog_file_map.txt", "data_type": "stft", "mod": "eog", "num_ch": 1},
        #     {"list_dir": "time_eeg_file_map.txt", "data_type": "time", "mod": "eog", "num_ch": 1},
        #     # {"list_dir": "time_eog_file_map.txt", "data_type": "time", "mod": "eog", "num_ch": 1}
        # ]

        # config.data_view_dir = [
        #     {"list_dir" : "patient_mat_list.txt", "data_type": "stft", "mod": "eeg", "num_ch": 1},
        #     {"list_dir" :"patient_eog_mat_list.txt", "data_type": "stft", "mod": "eog", "num_ch": 1},
        #     {"list_dir" : "patient_mat_list.txt", "data_type": "time", "mod": "eeg", "num_ch": 1},
        #     {"list_dir" : "patient_eog_mat_list.txt", "data_type": "time", "mod": "eog", "num_ch": 1}
        # ]

        # config.data_view_dir = [
        #     {"list_dir": "patient_mat_list.txt", "data_type": "stft", "mod": "eeg", "num_ch": 1},
        #     {"list_dir": "patient_mat_list.txt", "data_type": "time", "mod": "eeg", "num_ch": 1}
        # ]


        # config.huy_data= False

        # config.data_roots = "/esat/smcdata/users/kkontras/Image_Dataset/no_backup/Sleep_SHHS_2/"
        # config.data_view_dir = [
        #     {"list_dir": "stft_eeg_file_map_empties.txt", "data_type": "stft", "mod": "eeg", "num_ch": 1},
        #     {"list_dir": "stft_emg_file_map_empties.txt", "data_type": "stft", "mod": "eog", "num_ch": 1}
        # ]
        # config.calculate_metrics = False
        # config.model.load_ongoing = False
        # config.normalization.dir= "/esat/smcdata/users/kkontras/Image_Dataset/no_backup/Sleep_SHHS_2/metrics_eeg_eog_emg_stft.pkl"

        # config.model.save_dir = "/esat/smcdata/users/kkontras/Image_Dataset/no_backup/data/2021_data/sleep_edf/multi_modal/fourier_transformer_cls_eeg_eog_BIOBLIP_folds_{}.pth.tar".format(
        #     fold)
        # if fold == 0:
        #     config.model.save_dir = "/esat/smcdata/users/kkontras/Image_Dataset/no_backup/data/2021_data/sleep_edf/multi_modal/fourier_transformer_cls_eeg_eog_BIOBLIP.pth.tar"


        # config.data_loader_workers = 0

        # config.random_shuffle_data = False
        # config.random_shuffle_data_batch = True
        # deterministic(config.seed)
        # config.save_dir = "/users/sista/kkontras/Documents/Sleep_Project/data/2021_data/shhs1/shhs1_sleepTransormer_eeg_bfc_eqweights.pth.tar"
        # device = "cuda:{}".format(config.gpu_device[0])
        # device = "cpu"
        device = "cuda:0"

        if "model" not in config:
            print("Loading from {}".format(config.save_dir))
            checkpoint = torch.load(config.save_dir, map_location="cpu")
        else:
            print("Loading from {}".format(config.model.save_dir))
            checkpoint = torch.load(config.model.save_dir, map_location="cpu")
        logs = checkpoint['logs']
        print("-- Best Validation --")
        print(logs["best_logs"])
        if config.training_params.rec_test:
            print("-- Best Test --")
            print(logs["test_logs"][logs["best_logs"]["step"]])

        if "post_test_results" in checkpoint:
            multi_fold_results[fold] = checkpoint["post_test_results"]
            print("-- Best Test --")
            print("Acc: {0:.1f}, Kappa: {1:.3f}, F1: {2:.1f}, f1_per_class: {3:.1f} {4:.1f} {5:.1f} {6:.1f} {7:.1f}".format(
                checkpoint["post_test_results"]["accuracy"]*100,
                checkpoint["post_test_results"]["k"],
                checkpoint["post_test_results"]["f1"]*100,
                checkpoint["post_test_results"]["preclass_f1"][0]*100,
                checkpoint["post_test_results"]["preclass_f1"][1]*100,
                checkpoint["post_test_results"]["preclass_f1"][2]*100,
                checkpoint["post_test_results"]["preclass_f1"][3]*100,
                checkpoint["post_test_results"]["preclass_f1"][4]*100
            ))

        if logs["steps_no_improve"] > 5000:
            print("This models has finished training")
        else:
            print("Steps without improvement are  {}".format(int(logs["steps_no_improve"])))

        #
        dataloader = globals()[config.dataset.dataloader_class]
        data_loader = dataloader(config=config)
        #
        data_loader.load_metrics_ongoing(checkpoint["metrics"])
        data_loader.weights = logs["weights"]
        weights = data_loader.weights
        model, best_model = load_models(config=config, device=device, checkpoint=checkpoint)
        # model = load_models(config=config, device=device, checkpoint={}, only_model=True)



        # config_eeg = process_config(config_list[1], False)
        # config_eog = process_config(config_list[2], False)
        # print(config_eeg)
        # checkpoint_eeg = torch.load(config_eeg.model.save_dir, map_location="cpu")
        # checkpoint_eog = torch.load(config_eog.model.save_dir, map_location="cpu")
        # model_eeg, best_model_eeg = load_models(config=config_eeg, device=device, checkpoint=checkpoint_eeg)
        # model_eog, best_model_eog = load_models(config=config_eog, device=device, checkpoint=checkpoint_eeg)

        print("---- Running Model ----")
        # validate(config, model, data_loader.valid_loader, "Validation")
        # test(model, data_loader)
        print("---- Best Model ----")
        # validate(config, best_model, data_loader.valid_loader, "Validation", device)
        # validate(config, best_model, data_loader.train_loader, "Train", device)
        # validate(config, best_model, data_loader.test_loader, "Test", device)
        # validate(config, best_model, data_loader.total_loader, "Total", device)

        # multi_fold_results[fold] = validate(config, best_model, data_loader.valid_loader, "Val", device)
        # multi_fold_results[fold] = validate(config, best_model, data_loader.test_loader, "Test", device)
        # save_test_results(checkpoint, config.model.save_dir, multi_fold_results[fold])

        # validate_borders(config, best_model, data_loader.test_loader, "Test")
        # validate_borders(config, best_model, data_loader.train_loader, "Train")
        validate_borders(config, best_model, data_loader.valid_loader, "Validation")



        # for patient_num in [55, 203, 273]:
        #     test_acc, test_f1, test_k, test_auc, test_conf, test_perclass_f1, test_spec, test_sens, total_entropy = validate_specific_patient(model=best_model, data_loader=data_loader.test_loader, device=device, patient_num=patient_num)


        # with open("/esat/smcdata/users/kkontras/Image_Dataset/no_backup/Sleep_SHHS/Version_1/patient_map.pkl","rb") as f: patient_map = pickle.load(f)
        # with open("/esat/smcdata/users/kkontras/Image_Dataset/no_backup/Sleep_SHHS_2/patient_map.pkl", "rb") as f: patient_map = pickle.load(f)
        # with open("/esat/smcdata/users/kkontras/Image_Dataset/no_backup/Sleep_NCH/Version_1/patient_map.pkl", "rb") as f: patient_map = pickle.load(f)

        # patient_map = {"patient_{}".format(f'{patient_num:04}'): "patient_{}".format(f'{patient_num:04}') for patient_num in range(7000)}

        # with open("/esat/smcdata/users/kkontras/Image_Dataset/no_backup/Sleep_SHHS_2/f1kentrpy_results_shhs1.pkl", "rb") as f:
        #     f1kentrpy_results_shhs1 = pickle.load(f)
        # f1_comparisons = f1kentrpy_results_shhs1["f1"]
        # k_comparisons = f1kentrpy_results_shhs1["k"]
        # entropy_comparisons = f1kentrpy_results_shhs1["entropy"]

        # f1, k, entropy = gather_comparisons(model=best_model, model_eeg=best_model_eeg, model_eog=best_model_eog, data_loader=data_loader.test_loader, f1_comparisons=f1_comparisons, k_comparisons=k_comparisons, entropy_comparisons=entropy_comparisons, patient_map=patient_map)
        # f1, k, entropy = gather_comparisons(model=best_model, data_loader=data_loader.test_loader, f1_comparisons=f1_comparisons, k_comparisons=k_comparisons, entropy_comparisons=entropy_comparisons, patient_map=patient_map)

        # with open("/esat/smcdata/users/kkontras/Image_Dataset/no_backup/Sleep_SHHS/Version_1/patient_perf_merged_shhs1.pkl","wb") as f:
        # with open("/esat/smcdata/users/kkontras/Image_Dataset/no_backup/Sleep_SHHS/Version_1/patient_perf_eeg_shhs1.pkl","wb") as f:
        # with open("/esat/smcdata/users/kkontras/Image_Dataset/no_backup/Sleep_NCH/Version_1/f1kentrpy_results_nch_eeg.pkl", "wb") as f:
        #     pickle.dump({"f1":f1, "k":k, "entropy":entropy},f)
        # a = get_attention_weights(model=best_model, device=device, batch = config.test_batch_size, seq_l=config.seq_length[0], data_loader=data_loader.valid_loader, description="Validation")
        # a = get_attention_weights_merged(model=best_model, device=device, batch = config.test_batch_size, seq_l=config.seq_length[0], data_loader=data_loader.valid_loader, description="Validation")
        # a = get_attention_weights_concat(model=best_model, device=device, data_loader=data_loader.valid_loader, description="Validation")
        # a = get_attention_weights_bottleneck(model=best_model, device=device, data_loader=data_loader.valid_loader, description="Validation", context_points=1)
        # a = get_attention_weights_late(model=best_model, device=device, data_loader=data_loader.valid_loader, description="Validation")
        # a = get_attention_weights_late_contrastive(model=best_model, device=device, batch = config.test_batch_size, seq_l=config.seq_length[0], data_loader=data_loader.valid_loader, description="Validation")
        # a = get_attention_weights_late_contrastive(model=model, device=device, batch = config.test_batch_size, seq_l=config.seq_length[0], data_loader=data_loader.valid_loader, description="Validation")
        # a = get_attention_weights_late_retarded_norm(model=best_model, device=device, batch = config.test_batch_size, seq_l=config.seq_length[0], data_loader=data_loader.valid_loader, description="Validation")
        # a = get_attention_weights_late_norm(model=best_model, device=device, batch = config.test_batch_size, seq_l=config.seq_length[0], data_loader=data_loader.valid_loader, description="Validation")
        # a = get_attention_weights_merged_norm(model=best_model, device=device, batch = config.test_batch_size, seq_l=config.seq_length[0], data_loader=data_loader.valid_loader, description="Validation")
        # a = get_learnable_pos(model=best_model, device=device, batch = config.test_batch_size, seq_l=config.seq_length[0], data_loader=data_loader.valid_loader, description="Validation")
        # plot_hypnogram(data_loader.test_loader, 30, best_model, device)

        #ENSEMBLES
        # import copy
        # best_model_1 = copy.deepcopy(best_model)
        # best_model_2 = copy.deepcopy(best_model)
        # config.save_dir =  "/esat/smcdata/users/kkontras/Image_Dataset/no_backup/data/2021_data/shhs1/single_channel/replicate_nosch_adv.pth.tar"
        # print("Loading from {}".format(config.save_dir))
        # checkpoint_1 = torch.load(config.save_dir, map_location="cpu")
        # config.save_dir =  "/esat/smcdata/users/kkontras/Image_Dataset/no_backup/data/2021_data/shhs1/single_channel/replicate_v4.pth.tar"
        # print("Loading from {}".format(config.save_dir))
        # checkpoint_2 = torch.load(config.save_dir, map_location="cpu")
        # best_model_1.load_state_dict(checkpoint_1["best_model_state_dict"])
        # best_model_2.load_state_dict(checkpoint_2["best_model_state_dict"])
        #
        # validate_ensembles([best_model,best_model_1, best_model_2] , data_loader.valid_loader, "Validation")

        # data_loader.valid_loader.dataset.print_statistics_per_patient()
        # data_loader.test_loader.dataset.print_statistics_per_patient()
        # data_loader.train_loader.dataset.print_statistics_per_patient()

        # sleep_umap_plot(model, data_loader.valid_loader, "Validation")
        # sleep_umap_plot(best_model, data_loader.valid_loader, "Validation")
        #
        # print(data_loader.test_loader.dataset.dataset)
        # plot_hypnogram(data_loader.test_loader, 7, best_model, device)
        # plot_hypnogram(data_loader.train_loader, 20, best_model, device)
        # sleep_plot_losses(config, logs)

        sleep_plot_losses_multisupervised(config, logs)
        sleep_plot_f1_multisupervised(config, logs)
        sleep_plot_f1_perclass_multisupervised(config, logs)

        # sleep_plot_additional_losses(config, logs, 3)
        # sleep_plot_f1_perclass(config, logs)
        # sleep_plot_k(config, logs)
        # sleep_plot_f1(config, logs)
        # sleep_plot_lr(config, logs)

        # sleep_plot_losses_contrastive(config, logs)d

# config_list = ["a", "b"]
# f1_comparisons = {"a":[0.1,0.6,0.7],"b":[0.4,0.3,0.5]}
# entropy_comparisons = {"a":[0.1,0.6,0.7],"b":[0.4,0.3,0.5]}

# f1, acc, k, f1_perclass = [], [], [], []
# for fold_metrics in multi_fold_results:
#     this_fold_metrics = multi_fold_results[fold_metrics]
#
#     acc.append(this_fold_metrics["accuracy"])
#     f1.append(this_fold_metrics["f1"])
#     k.append(this_fold_metrics["k"])
#     f1_perclass.append(this_fold_metrics["preclass_f1"])
# print("Total")
# print("acc: {0:.1f}".format(np.array(acc).mean()*100),end=" ")
# print("f1: {0:.1f}".format(np.array(f1).mean()*100),end=" ")
# print("k: {0:.3f}".format(np.array(k).mean()),end=" ")
# print("f1 per class: {}".format(np.round(np.array(f1_perclass).mean(axis=0)*100,1)),end=" ")
#
# results_of_comparison = {"f1":f1_comparisons, "k":k_comparisons, "entropy":entropy_comparisons,"config_list":config_list}
# # a_file = open("./results_of_comparisons.pkl", "rb")
# # results_of_comparison_prev = pickle.load(a_file)
# # a_file.close()
# # results_of_comparison_prev.update(results_of_comparison)
# # a_file = open("./results_of_comparisons.pkl", "wb")
# # pickle.dump(results_of_comparison, a_file)
# # a_file.close()
# plot_comparisons(f1_comparisons, k_comparisons, entropy_comparisons, config_list)

# b = np.array([0.667, 0.723, 0.762, 0.800])
# a = np.array([ 0.617, 0.481, 0.564, 0.701])
#
# a = np.array([0.710, 0.781, 0.813, 0.829])
# b = np.array([0.685, 0.778, 0.809, 0.831])
#
# plt.plot(np.array([0,1,2,3]), b, label="Random weights")
# plt.scatter(np.array([0,1,2,3]), b)
# plt.plot(np.array([0,1,2,3]), a, label="Temporal Shuffling pretrained")
# plt.scatter(np.array([0,1,2,3]), a)
# plt.xticks([0, 1, 2, 3], ['10p', '100p', '500p', 'Full'],rotation=20)  # Set text labels and properties.
# plt.axhline(y = 0.7, color = 'r', linestyle = 'dashed')
# plt.xlim(-.5,3.5)
# plt.ylim(0.6,0.85)
# plt.legend()
# plt.xlabel("#Patients")
# plt.ylabel("Macro F1 Score")
# plt.title("Temporal Shuffling Comparison")
# plt.savefig("./temp_shuffle_results_shhs.jpg")
# plt.show()
