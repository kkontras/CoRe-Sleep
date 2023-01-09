import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE_Loss(nn.Module):
    def __init__(self, kld_weight=0, reduction="mean"):
        super(VAE_Loss, self).__init__()
        self.kld_weight = kld_weight
        self.reduction = reduction

    def forward(self, reconstruction, input, mu=None, log_var=None) -> dict:

        recons_loss = F.mse_loss(reconstruction, input, reduction=self.reduction)
        output_loss = { 'reconstruction_Loss': recons_loss.detach().cpu().numpy()}

        loss = recons_loss
        if self.kld_weight != 0:
            kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)
            loss +=self.kld_weight * kld_loss

            output_loss.update({"kld_loss": kld_loss.detach().cpu().numpy()})
        output_loss.update({'total': loss})

        return output_loss

