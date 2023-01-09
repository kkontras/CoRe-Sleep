import torch

class regr_nlloss(torch.nn.Module):
    def __init__(obj):
        super().__init__()
        obj.message_ = "This is a custom loss function to resemble nll on regression"
    def __call__(obj,sigma,mean,target):#sigma, mean, targets):
        loss = 0.5*torch.log(sigma) + 0.5*torch.div(torch.pow((target.squeeze(dim=1) - mean), 2),sigma) + 5
        return loss.mean()