# GUL from WingsNet: https://github.com/haozheng-sjtu/3d-airway-segmentation/blob/main/loss%20functions.py

from typing import Callable

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from scipy import ndimage
from scipy.ndimage import distance_transform_edt
import cupy as cp
from cucim.core.operations import morphology

from nnunetv2.utilities.ddp_allgather import AllGatherGrad

class GUL(nn.Module):
    def __init__(self, apply_nonlin: Callable = None, do_bg: bool = True, alpha: float = 0.1, 
                 smooth: float =1., sigma1: float = 1e-3, sigma2: float = 1e-3, ddp: bool = True):
        super(GUL, self).__init__()
        self.apply_nonlin = apply_nonlin
        self.do_bg = do_bg
        self.alpha = alpha
        self.beta = 1 - alpha
        self.smooth = smooth
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.ddp = ddp

        if self.smooth is not None:
            if self.smooth < 0 or self.smooth > 1.0:
                raise ValueError('smooth value should be in [0,1]')
    
    # converting to np too slow (GPU -> CPU -> GPU)
    @torch.no_grad()
    def compute_dist_ratio(self, seg: torch.Tensor, cl: torch.Tensor):
        """
        :param seg: label
        :param cl: centreline
        """
        num_batches = seg.shape[0]
        seg_squeezed = cp.asarray(seg).squeeze(axis=1)
        cl_squeezed = cp.asarray(cl).squeeze(axis=1)
        dist = cp.zeros(seg.shape)
        if num_batches > 1:
            for b in range(num_batches):
                dist[b,0,...] = morphology.distance_transform_edt(
                    cp.asarray(seg_squeezed[b,...]) * (1 - cp.asarray(cl_squeezed[b,...])))
        else: 
            dist[b,0,...] = morphology.distance_transform_edt(
                cp.asarray(seg_squeezed) * (1 - cp.asarray(cl_squeezed)))
                            
        d_i = torch.as_tensor(cp.expand_dims(dist, axis=1), device='cuda')
        d_max = torch.as_tensor(cp.amax(dist)) # max d_i in one case
        dist_ratio = (d_i+self.smooth) / (d_max+self.smooth)
        return dist_ratio

    def forward(self, pred: torch.Tensor, target: torch.Tensor, cl: torch.Tensor):
        """
        :param pred: output from network (b, c, z, y, x)
        :param target: label (b, c, z, y, x)
        :param cl: centreline/skeleton (b, c, z, y, x)
        """

        shp_pred = pred.shape
        if self.apply_nonlin is not None:
            pred = self.apply_nonlin(pred)
        axes = list(range(2, len(shp_pred))) # compute for each sample individually

        assert pred.dim() == 4 or pred.dim() == 5, "Only 2D and 3D supported"
        assert (
            pred.dim() == target.dim()
        ), "Prediction and target need to be of same dimension"
        assert torch.all((target == 0) | (target == 1)), "Target is not binary"

        # tp, fp, fn, _, tp_root = get_tp_fp_fn_tn(pred, target, axes)
        with torch.no_grad():
            if pred.ndim != target.ndim:
                target = target.view((target.shape[0], 1, *target.shape[1:]))

            if pred.shape == target.shape:
                # if this is the case then gt is probably already a one hot encoding
                y_onehot = target
            else:
                y_onehot = torch.zeros(pred.shape, device=pred.device)
                y_onehot.scatter_(1, target.long(), 1)

        dist = self.compute_dist_ratio(target, cl)
        m = (1 - 2*self.alpha) / (1 - self.alpha)
        weight = (1 - m*(dist**0.5)) * target + (1 - target) # r_d = 0.5

        weighted_tp = weight * pred * y_onehot
        weighted_tp_root = weight * (pred ** 0.7) * y_onehot
        weighted_fp = weight * pred * (1 - y_onehot)
        weighted_fn = weight * (1 - pred) * y_onehot

        if len(axes) > 0:
            weighted_tp = weighted_tp.sum(dim=axes, keepdim=False)
            weighted_tp_root = weighted_tp_root.sum(dim=axes, keepdim=False)
            weighted_fp = weighted_fp.sum(dim=axes, keepdim=False)
            weighted_fn = weighted_fn.sum(dim=axes, keepdim=False)

        if self.ddp:
            weighted_tp = AllGatherGrad.apply(weighted_tp).sum(0)
            weighted_tp_root = AllGatherGrad.apply(weighted_tp_root).sum(0)
            weighted_fp = AllGatherGrad.apply(weighted_fp).sum(0)
            weighted_fn = AllGatherGrad.apply(weighted_fn).sum(0)


        numerator = weighted_tp_root
        denominator = (weighted_tp + self.alpha * weighted_fp + self.beta * weighted_fn)

        loss = (numerator+self.smooth) / (torch.clip(denominator + self.smooth, 1e-8))
        loss = loss.mean()

        return 1 - loss

def get_tp_fp_fn_tn(net_output, gt, axes=None, mask=None, square=False):
    """
    net_output must be (b, c, x, y(, z)))
    gt must be a label map (shape (b, 1, x, y(, z)) OR shape (b, x, y(, z))) or one hot encoding (b, c, x, y(, z))
    if mask is provided it must have shape (b, 1, x, y(, z)))
    :param net_output:
    :param gt:
    :param axes: can be (, ) = no summation
    :param mask: mask must be 1 for valid pixels and 0 for invalid pixels
    :param square: if True then fp, tp and fn will be squared before summation
    :return:
    """
    if axes is None:
        axes = tuple(range(2, net_output.ndim))

    with torch.no_grad():
        if net_output.ndim != gt.ndim:
            gt = gt.view((gt.shape[0], 1, *gt.shape[1:]))

        if net_output.shape == gt.shape:
            # if this is the case then gt is probably already a one hot encoding
            y_onehot = gt
        else:
            y_onehot = torch.zeros(net_output.shape, device=net_output.device)
            y_onehot.scatter_(1, gt.long(), 1)

    tp = net_output * y_onehot
    fp = net_output * (1 - y_onehot)
    fn = (1 - net_output) * y_onehot
    tn = (1 - net_output) * (1 - y_onehot)
    tp_root = (net_output ** 0.7) * y_onehot

    if mask is not None:
        with torch.no_grad():
            mask_here = torch.tile(mask, (1, tp.shape[1], *[1 for _ in range(2, tp.ndim)]))
        print('mask_here from get_tp') # not run
        tp *= mask_here
        fp *= mask_here
        fn *= mask_here
        tn *= mask_here
        tp_root *= mask_here

    if square:
        print('square from get_tp') # not run
        tp = tp ** 2
        fp = fp ** 2
        fn = fn ** 2
        tn = tn ** 2

    if len(axes) > 0:
        print('sum from get_tp')
        tp = tp.sum(dim=axes, keepdim=False)
        fp = fp.sum(dim=axes, keepdim=False)
        fn = fn.sum(dim=axes, keepdim=False)
        tn = tn.sum(dim=axes, keepdim=False)
        tp_root = tp_root.sum(dim=axes, keepdim=False)

    return tp, fp, fn, tn, tp_root
    