# https://github.com/Nandayang/YangLab_FANN/blob/main/utils/losses.py

from typing import Callable, List

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from nnunetv2.utilities.ddp_allgather import AllGatherGrad
from nnunetv2.training.loss.robust_ce_loss import RobustCrossEntropyLoss

class Dice_BCE_CL(nn.Module):
    def __init__(self, weight_dc, weight_ce, weight_cont, ce_kwargs, ignore_label = None, ddp: bool = True):
        super(Dice_BCE_CL, self).__init__()
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label
        self.weight_dc = weight_dc
        self.weight_ce = weight_ce
        self.weight_cont = weight_cont
        self.ddp = ddp
        self.ignore_label = ignore_label

        self.dice = DiceLoss(ddp=self.ddp, apply_nonlin=torch.sigmoid)
        # self.bce = nn.BCEWithLogitsLoss(**bce_kwargs)
        self.ce = RobustCrossEntropyLoss(**ce_kwargs)
        self.continuity = ContinuityLoss(ddp=self.ddp, apply_nonlin=torch.sigmoid)

    def forward(self, pred: torch.Tensor, target: torch.Tensor, cl: torch.Tensor):
        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'ignore label is not implemented for one hot encoded target variables ' \
                                         '(DC_and_CE_loss)'
            mask = target != self.ignore_label
            # remove ignore label from target, replace with one of the known labels. It doesn't matter because we
            # ignore gradients in those areas anyway
            target_dice = torch.where(mask, target, 0)
            num_fg = mask.sum()
        else:
            target_dice = target
            mask = None
        
        dice = self.dice(pred, target) \
            if self.weight_dc != 0 else 0
        # if mask is not None:
        #     bce = (self.bce(pred, target_regions) * mask).sum() / torch.clip(mask.sum(), min=1e-8)
        # else:
        #     bce = self.bce(pred, target_regions)
        ce_loss = self.ce(pred, target[:, 0]) \
            if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0) else 0
        cont = self.continuity(pred, cl) \
            if self.weight_cont != 0 else 0
        
        out_loss = self.weight_dc * dice + self.weight_ce * ce_loss + self.weight_cont * cont
        return out_loss

class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1e-5, ddp: bool = True, apply_nonlin: Callable = None):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.ddp = ddp
        self.apply_nonlin = apply_nonlin
    
    def forward(self, output: torch.Tensor, target: torch.Tensor):
        axes = list(range(2, len(output.shape))) # compute for each sample individually

        if self.apply_nonlin is not None:
            output = self.apply_nonlin(output)
        
        assert output.dim() == 4 or output.dim() == 5, "Only 2D and 3D supported"
        assert(
            output.dim() == target.dim()
        ), "Prediction, and target should have the same dimensions"
        assert torch.all((target == 0) | (target == 1)), "Target should be binary"

        with torch.no_grad():
            if output.ndim != target.ndim:
                target = target.view((target.shape[0], 1, *target.shape[1:]))
            
            if output.shape == target.shape:
                y_onehot = target
            else:
                y_onehot = torch.zeros(output.shape, device=output.device)
                y_onehot.scatter_(1, target.long(), 1)
        
        tp = output * y_onehot
        fp = output * (1 - y_onehot)
        fn = (1 - output) * y_onehot

        if len(axes) > 0:
            tp = tp.sum(dim=axes, keepdim=False)
            fp = fp.sum(dim=axes, keepdim=False)
            fn = fn.sum(dim=axes, keepdim=False)
        
        if self.ddp:
            tp = AllGatherGrad.apply(tp).sum(0)
            fp = AllGatherGrad.apply(fp).sum(0)
            fn = AllGatherGrad.apply(fn).sum(0)

        numerator = 2. * tp + self.smooth
        denominator = 2. * tp + fp + fn
        dice = (numerator + self.smooth) / (torch.clip(denominator + self.smooth, 1e-8))
        dice = dice.mean()
        return 1 - dice
    
class ContinuityLoss(nn.Module):
    def __init__(self, smooth: float = 1e-5, ddp: bool = True, apply_nonlin: Callable = None):
        super(ContinuityLoss, self).__init__()
        self.smooth = smooth
        self.ddp = ddp
        self.apply_nonlin = apply_nonlin

    def forward(self, output: torch.Tensor, cl: torch.Tensor):
        axes = list(range(2, len(output.shape)))

        if self.apply_nonlin is not None:
            output = self.apply_nonlin(output)
        
        assert output.dim() == 4 or output.dim() == 5, "Only 2D and 3D supported"
        assert(
            output.dim() == cl.dim()
        ), "Prediction, and target cl should have the same dimensions"
        assert torch.all((cl == 0) | (cl == 1)), "Target cl should be binary"

        with torch.no_grad():
            if output.ndim != cl.ndim:
                cl = cl.view((cl.shape[0], 1, *cl.shape[1:]))
            
            if output.shape == cl.shape:
                y_onehot = cl
            else:
                y_onehot = torch.zeros(output.shape, device=output.device)
                y_onehot.scatter_(1, cl.long(), 1)

        true_cl = output * y_onehot
        if len(axes) > 0:
            true_cl = true_cl.sum(dim=axes, keepdim=False)
        if self.ddp:
            true_cl = AllGatherGrad.apply(true_cl).sum(0)

        cl_pred_num = torch.sum(true_cl) # num of correctly predicted centrelines
        cl_true_num = torch.sum(y_onehot) # num of ground truth centrelines

        continuity = (cl_pred_num + self.smooth) / (cl_true_num + self.smooth)
        continuity = continuity.mean()
        return 1 - continuity
