import torch
import torch.nn as nn
from colorama import init, Fore, Back, Style
from sklearn.metrics import f1_score, cohen_kappa_score, roc_auc_score, confusion_matrix
import numpy as np
from collections import defaultdict
from scipy.stats import entropy
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm
from scipy.special import softmax
import einops

class Generate_n_Compare():
    def __init__(self, model, data_loader, config, device):
        self.config = config
        self.device = device
        self.model = model
        self.data_loader = data_loader

    def reconstruct(self, set: str = "Validation", plot_comparison: bool=False):

        if set == "Validation":
            self.reconstruct_n_plot(data_loader=self.data_loader.valid_loader, description=set, plot_comparison=plot_comparison)
        elif set == "Test":
            self.reconstruct_n_plot(data_loader=self.data_loader.test_loader, description=set, plot_comparison=plot_comparison)
        elif set == "Train":
            self.reconstruct_n_plot(data_loader=self.data_loader.train_loader, description=set, plot_comparison=plot_comparison)
        elif set == "Total":
            self.reconstruct_n_plot(data_loader=self.data_loader.total_loader, description=set, plot_comparison=plot_comparison)
        else:
            raise ValueError('This set {} does not exist, options are "Validation", "Test", Train" "Total"'.format(set))

        return None

    def reconstruct_n_plot(self, data_loader, description, plot_comparison=False):
            self.model.eval()
            with torch.no_grad():
                tts, preds, inits, batch_loss = [], [], [], []
                pbar = tqdm(enumerate(data_loader), desc=description, leave=False)
                for batch_idx, (data, target, init, _) in pbar:

                    views = [data[i].float().to(self.device) for i in range(len(data))]

                    pred = self.model(views)
                    output_losses = self.model.module.loss_function(pred[0], pred[1], pred[2], pred[3], reduction="none")

                    output_losses["total"] = output_losses["total"].flatten(start_dim=1).mean(dim=1)
                    output_losses["total"] = output_losses["total"].cpu().numpy()

                    output_losses["reconstruction_Loss"] = torch.from_numpy(output_losses["reconstruction_Loss"]).flatten(start_dim=1).mean(dim=1)
                    output_losses["reconstruction_Loss"] = output_losses["reconstruction_Loss"].numpy()

                    batch_loss.append(output_losses)

                    if plot_comparison:
                        for i in range(32):
                            output = einops.rearrange(pred[0][i].squeeze(), "inner f t -> (inner t) f").cpu().numpy()
                            input = einops.rearrange(pred[1][i].squeeze(), "inner f t -> (inner t) f").cpu().numpy()

                            labels = ["W", "N1", "N2", "N3", "R"]
                            plt.figure()
                            plt.subplot(211)
                            t = np.linspace(0, len(input) - 1, len(input))
                            f = np.linspace(0, 128 - 1, 128)
                            plt.pcolormesh(t, f, input.transpose())
                            for x in range(0, len(t), 29):
                                plt.axvline(x)
                                plt.text(x + 5, 10, labels[target[i][int(x / 29)].item()])
                            plt.yticks(fontsize=8)
                            plt.ylabel("F EEG")
                            plt.xlabel("Hours")
                            plt.subplot(212)
                            t = np.linspace(0, len(output) - 1, len(output))
                            f = np.linspace(0, 128 - 1, 128)
                            plt.pcolormesh(t, f, output.transpose())
                            for x in range(0, len(t), 29):
                                plt.axvline(x)
                                plt.text(x + 5, 10, labels[target[i][int(x / 29)].item()])
                            plt.yticks(fontsize=8)
                            plt.ylabel("F EEG Reconstructed")
                            plt.xlabel("Hours")
                            plt.show()


                    # preds.append(pred)
                    inits.append(init.flatten())
                    pbar.set_description("{} batch {}/{}".format(description, int(batch_idx), int(len(data_loader))))
                    pbar.refresh()

            mean_batch = self._calc_mean_batch_loss(batch_loss=batch_loss)

            print(mean_batch)

    def _calc_mean_batch_loss(self, batch_loss):
        mean_batch = defaultdict(list)
        for b_i in batch_loss:
            for loss_key in b_i:
                for j in b_i[loss_key]:
                    mean_batch[loss_key].append(j)
        for key in mean_batch:
            mean_batch[key] = np.array(mean_batch[key])
            if key=="total":
                plt.figure()
                plt.title("Reconstruction loss per batch")
                plt.ylabel("Reconstruction loss - Val")
                plt.xlabel("Batch")
                plt.bar(np.arange(len(mean_batch[key])), np.array(mean_batch[key]))
                plt.ylim([0,6])
                plt.show()
            mean_batch[key] = np.array(mean_batch[key]).mean(axis=0)
        return mean_batch