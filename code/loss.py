import torch
import torch.nn as nn



#Loss for Jarzynski and PCD, for CD the loss would be the same of PCD but xw are resampled from the data 


class WLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, xpred, weights=None):
        """
        If weights is None then returns the usual loss. 
        Otherwise computes the mean of xpred with the weigths. 
        """
        if weights is None:
            pred_mean = xpred.mean()
        else:
            pred_mean = (xpred*weights[:,None]).sum()
        
        return x.mean() - pred_mean 