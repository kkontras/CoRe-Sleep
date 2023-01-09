import torch
import numpy as np

class CrossEntropyLoss(torch.nn.Module):

    def __init__(self, weight=None):
        super().__init__()

        self.loss = torch.nn.CrossEntropyLoss(weight=weight)

    def forward(self, outputs, targets):
        # a = torch.nn.functional.log_softmax(outputs, dim=1)
        return self.loss(outputs, targets)

    def pixel_CrossEntropyLoss(self, outputs, targets, indices):
        one_hot = torch.nn.functional.one_hot(targets, outputs.shape[1])
        one_hot = one_hot.permute(0, 3, 1, 2)
        log_softmax = torch.nn.functional.log_softmax(outputs, dim=1)
        cross_entropy = -torch.sum(one_hot * log_softmax, axis=1)
        if type(indices) is np.ndarray:
            indices = torch.from_numpy(indices)
        mask = indices.ge(0.5).cuda()
        cross_entropy_mean = torch.mean(torch.masked_select(cross_entropy,mask))
        return cross_entropy_mean

