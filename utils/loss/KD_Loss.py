import torch.nn as nn
import torch.nn.functional as F

class KD_Loss(nn.Module):
    def __init__(self, alpha, temp):
        super(KD_Loss, self).__init__()
        self.alpha = alpha
        self.T = temp

    def forward(self, outputs, labels, teacher_outputs):
        """
        Compute the knowledge-distillation (KD) loss given outputs, labels.
        "Hyperparameters": temperature and alpha
        NOTE: the KL Divergence for PyTorch comparing the softmaxs of teacher
        and student expects the input tensor to be log probabilities! See Issue #2
        """

        KD_loss = nn.KLDivLoss()(F.log_softmax(outputs/self.T, dim=1), F.softmax(teacher_outputs/self.T, dim=1)) * (self.alpha * self.T * self.T) + F.cross_entropy(outputs, labels) * (1. - self.alpha)

        return KD_loss