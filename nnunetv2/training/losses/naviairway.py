# https://github.com/AntonotnaWang/NaviAirway/blob/main/func/loss_func.py

from typing import Callable

import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch
import numpy as np

from nnunetv2.utilities.ddp_allgather import AllGatherGrad

class NaviAirwayLoss(nn.Module):
    def __init__(self, device, ddp: bool = True):
        super(NaviAirwayLoss, self).__init__()
        self.device = device
        self.ddp = ddp
        # self.ske = SkeletonDiceLoss(ddp=self.ddp, apply_nonlin=torch.sigmoid)
        self.pen = PenaltyLoss(alpha=2, ddp=self.ddp, apply_nonlin=torch.sigmoid)

    def forward(self, output:torch.Tensor, target:torch.Tensor):
        weights = self.calculate_weights(target).to(self.device)
        # self.ske_loss = self.ske(output, target, weights)
        self.pen_loss = self.pen(output, target, weights)
        # return self.ske_loss + self.pen_loss
        return self.pen_loss
        
    # @torch.no_grad()
    def calculate_weights(self, target:torch.Tensor):
        background = 1 - target

        fg_pix_num = torch.sum(target)
        bg_pix_num = torch.sum(background)
        fg_pix_per = fg_pix_num / (fg_pix_num + bg_pix_num)
        bg_pix_per = bg_pix_num / (fg_pix_num + bg_pix_num)
        weight_fg = torch.exp(bg_pix_per) / (torch.exp(fg_pix_per)+torch.exp(bg_pix_per))
        weight_bg = torch.exp(fg_pix_per) / (torch.exp(fg_pix_per)+torch.exp(bg_pix_per))
        weights = weight_fg*torch.eq(target,1).float() + weight_bg*torch.eq(target,0).float()

        return weights

# class SkeletonDiceLoss(nn.Module):
#     """
#     dice_loss_weights
#     """
#     def __init__(self, smooth: float=1e-2, ddp: bool = True, apply_nonlin: Callable = None):
#         super(SkeletonDiceLoss, self).__init__()
#         self.smooth = smooth
#         self.ddp = ddp
#         self.apply_nonlin = apply_nonlin
    
#     def forward(self, output:torch.Tensor, target:torch.Tensor, weights:torch.Tensor):
#         axes = list(range(2, len(output.shape))) # compute for each sample individually

#         if self.apply_nonlin is not None:
#             output = self.apply_nonlin(output)
        
#         assert output.dim() == 4 or output.dim() == 5, "Only 2D and 3D supported"
#         assert(
#             output.dim() == target.dim() == weights.dim()
#         ), "Prediction, target and weights should have the same dimensions"
#         assert torch.all((target == 0) | (target == 1)), "Target should be binary"

#         with torch.no_grad():
#             if output.ndim != target.ndim:
#                 target = target.view((target.shape[0], 1, *target.shape[1:]))
            
#             if output.shape == target.shape:
#                 y_onehot = target
#             else:
#                 y_onehot = torch.zeros(output.shape, device=output.device)
#                 y_onehot.scatter_(1, target.long(), 1)
        
#         weighted_tp = weights * output * y_onehot
#         tp = output * y_onehot
#         fp = output * (1 - y_onehot)
#         fn = (1 - output) * y_onehot

#         if len(axes) > 0:
#             weighted_tp = weighted_tp.sum(dim=axes, keepdim=False)
#             tp = tp.sum(dim=axes, keepdim=False)
#             fp = fp.sum(dim=axes, keepdim=False)
#             fn = fn.sum(dim=axes, keepdim=False)
        
#         if self.ddp:
#             weighted_tp = AllGatherGrad.apply(weighted_tp).sum(0)
#             tp = AllGatherGrad.apply(tp).sum(0)
#             fp = AllGatherGrad.apply(fp).sum(0)
#             fn = AllGatherGrad.apply(fn).sum(0)

#         numerator = 2. * weighted_tp
#         denominator = 2. * tp + fp + fn
#         dice = (numerator + self.smooth) / (torch.clip(denominator + self.smooth, 1e-8))
#         dice = dice.mean()
#         return 1 - dice

class PenaltyLoss(nn.Module):
    """
    dice_loss_power_weights
    """
    def __init__(self, alpha: float=0.5, ddp: bool = True, apply_nonlin: Callable = None):
        super(PenaltyLoss, self).__init__()
        self.alpha = alpha
        self.delta = 1e-5
        self.smooth = 1.
        self.ddp = ddp
        self.apply_nonlin = apply_nonlin
    
    def forward(self, output:torch.Tensor, target:torch.Tensor, weights:torch.Tensor):
        axes = list(range(2, len(output.shape))) # compute for each sample individually

        if self.apply_nonlin is not None:
            output = self.apply_nonlin(output)
        
        assert output.dim() == 4 or output.dim() == 5, "Only 2D and 3D supported"
        assert(
            output.dim() == target.dim() == weights.dim()
        ), "Prediction, target and weights should have the same dimensions"
        assert torch.all((target == 0) | (target == 1)), "Target should be binary"

        with torch.no_grad():
            if output.ndim != target.ndim:
                target = target.view((target.shape[0], 1, *target.shape[1:]))
            
            if output.shape == target.shape:
                y_onehot = target
            else:
                y_onehot = torch.zeros(output.shape, device=output.device)
                y_onehot.scatter_(1, target.long(), 1)
        
        weighted_tp = weights * torch.pow(output+self.delta, self.alpha) * y_onehot
        tp = torch.pow(output+self.delta, self.alpha) * y_onehot
        fp = output * (1 - y_onehot)
        fn = (1 - output) * y_onehot

        if len(axes) > 0:
            weighted_tp = weighted_tp.sum(dim=axes, keepdim=False)
            tp = tp.sum(dim=axes, keepdim=False)
            fp = fp.sum(dim=axes, keepdim=False)
            fn = fn.sum(dim=axes, keepdim=False)
        
        if self.ddp:
            weighted_tp = AllGatherGrad.apply(weighted_tp).sum(0)
            tp = AllGatherGrad.apply(tp).sum(0)
            fp = AllGatherGrad.apply(fp).sum(0)
            fn = AllGatherGrad.apply(fn).sum(0)

        numerator = 2. * weighted_tp
        denominator = 2. * tp + fp + fn
        dice = (numerator + self.smooth) / (torch.clip(denominator + self.smooth, 1e-8))
        dice = dice.mean()
        return 1 - dice

