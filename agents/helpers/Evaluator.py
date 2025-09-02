import torch
import logging
import copy
from torchmetrics import F1Score, CohenKappa, Accuracy
from collections import defaultdict

class All_Evaluator:
    def __init__(self, config, dataloaders: dict):
        self.train_evaluator = General_Evaluator(config, len(dataloaders.train_loader.dataset), use_missing_idx=True)
        self.val_evaluator = General_Evaluator(config, len(dataloaders.train_loader.dataset))
        if hasattr(dataloaders, "test_loader"):
            self.test_evaluator = General_Evaluator(config, len(dataloaders.test_loader.dataset))

class General_Evaluator:
    def __init__(self, config,  total_instances: int, use_missing_idx: bool=False):
        self.config = config
        self.use_missing_idx = use_missing_idx
        self.total_instances = total_instances
        self.num_classes = config.model.args.num_classes
        self.reset()

        self.early_stop = False

        self.best_acc = 0.0
        self.best_loss = 0.0

    def set_best(self, best_acc, best_loss):
        self.best_acc = best_acc
        self.best_loss = best_loss
        logging.info("Set current best acc {}, loss {}".format(self.best_acc, self.best_loss))

    def reset(self):
        self.losses = []
        self.preds = {pred_key.lower(): [] for pred_key in self.config.model.args.multi_loss.supervised_losses}
        self.labels = []
        self.missing_mod_idx = {pred_key.lower(): [] for pred_key in self.config.model.args.multi_loss.supervised_losses}
        self.processed_instances = 0

    def process(self, losses:dict, preds: dict, label: torch.Tensor, missing_mod_idx: dict=None):

        num_instances = label.shape[0]
        self.labels.append(label)

        for pred_key in self.preds:
            if pred_key in preds:
                assert (len(preds[pred_key].shape) == 2)
                self.preds[pred_key].append(preds[pred_key])

            if missing_mod_idx is not None and pred_key in missing_mod_idx:
                assert (missing_mod_idx[pred_key].shape[0]*missing_mod_idx[pred_key].shape[1] == num_instances), "missing_mod_idx index has to have the same length as the batch*sequence length"
                self.missing_mod_idx[pred_key].append(missing_mod_idx[pred_key].flatten())
            else:
                self.missing_mod_idx[pred_key].append(torch.Tensor([True]).repeat(num_instances))

        self.processed_instances += num_instances
        self.losses.append(losses)

    def get_early_stop(self):
        return self.early_stop

    def enable_early_stop(self):
        self.early_stop = True

    def mean_batch_loss(self):
        """
        Calculate the mean loss of the batches
        """
        if len(self.losses)==0:
            return None, ""

        mean_batch_loss = defaultdict(lambda: 0)
        count = defaultdict(lambda: 0)
        for i in range(len(self.losses)):
            for loss_key in self.losses[i]:
                mean_batch_loss[loss_key] += self.losses[i][loss_key]
                count[loss_key] += 1
        for loss_key in mean_batch_loss:
            mean_batch_loss[loss_key] /= count[loss_key]
            mean_batch_loss[loss_key] = mean_batch_loss[loss_key].item()

        message = ""
        for mean_key in mean_batch_loss: message += "{}: {:.3f} ".format(mean_key, mean_batch_loss[mean_key])

        return dict(mean_batch_loss), message

    def evaluate(self):
        targets_tens = torch.concatenate(self.labels).cpu().flatten()

        mean_batch_loss, _ = self.mean_batch_loss()

        total_preds, metrics  = {}, defaultdict(dict)
        if mean_batch_loss is not None:
            metrics["loss"] = mean_batch_loss
        for pred_key in self.preds:
            if len(self.preds[pred_key]) == 0: continue
            total_preds = torch.concatenate(self.preds[pred_key]).cpu()#[:self.processed_instances]

            if self.use_missing_idx and pred_key in self.missing_mod_idx and len(self.missing_mod_idx[pred_key]) > 0:
                missing_mod_idx = torch.concatenate(self.missing_mod_idx[pred_key]).cpu().flatten()
                this_target_tens = targets_tens[missing_mod_idx.bool()]
            else:
                this_target_tens = targets_tens

            metrics["acc"][pred_key] = Accuracy(task="multiclass", num_classes=self.num_classes)(total_preds,this_target_tens).item()
            metrics["f1"][pred_key] = F1Score( task="multiclass", num_classes=self.num_classes, average='macro')(total_preds, this_target_tens).item()
            metrics["f1_mi"][pred_key] = F1Score( task="multiclass", num_classes=self.num_classes, average='micro')(total_preds, this_target_tens).item()
            metrics["k"][pred_key] = CohenKappa(task="multiclass", num_classes=self.num_classes)(total_preds, this_target_tens).item()
            metrics["f1_perclass"][pred_key] = F1Score(task="multiclass", num_classes=self.num_classes, average=None)(total_preds, this_target_tens)

        metrics = dict(metrics) #Avoid passing empty dicts to logs, better return an error!

        return metrics

    def is_best(self, metrics = None, best_logs=None):
        if metrics is None:
            metrics = self.evaluate()

        validate_with = self.config.early_stopping.get("validate_with", "loss")
        if validate_with == "loss":
            is_best = (metrics["loss"]["total"] < best_logs["loss"]["total"])
        elif validate_with == "accuracy":
            is_best = (metrics["acc"]["combined"] > best_logs["acc"]["combined"])
        else:
            raise ValueError("self.agent.config.early_stopping.validate_with should be either loss or accuracy")
        return is_best
