# https://github.com/Puzzled-Hui/Connectivity-Aware-Airway-Segmentation/blob/main/networks/CAS.py
# https://github.com/Puzzled-Hui/Connectivity-Aware-Airway-Segmentation/blob/main/networks/utils.py

import torch
import torch.nn as nn
import numpy as np

import cupy as cp
from cucim.core.operations import morphology

class TverskyLoss(nn.Module):
    def __init__(self, apply_nonlin=None, batch_dice=False, do_bg=True, smooth=1., square=False, alpha=0.1, beta=0.9):
        super(TverskyLoss, self).__init__()
        self.square = square
        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth
        self.alpha = alpha
        self.beta = beta

    def forward(self, x, y, loss_mask=None):
        shp_x = x.shape
        if self.batch_dice:
            axes = [0] + list(range(2, len(shp_x)))
        else:
            axes = list(range(2, len(shp_x)))
        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)
        tp, fp, fn = get_tp_fp_fn(x, y, axes, loss_mask, self.square)
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        if not self.do_bg:
            if self.batch_dice:
                tversky = tversky[1:]
            else:
                tversky = tversky[:, 1:]
        tversky = tversky.mean()

        return -tversky


class CAS_COM(nn.Module):
    def __init__(self, apply_nonlin=None, alpha=0.1, beta=0.9, batch_dice=False, do_bg=True, smooth=1., square=False,
                 ddp: bool = True):
        super(CAS_COM, self).__init__()
        self.crossentropy_loss = nn.CrossEntropyLoss(reduction='none')
        self.tversky = TverskyLoss(apply_nonlin=apply_nonlin, batch_dice=batch_dice, do_bg=do_bg, smooth=smooth,
                                   square=square,
                                   alpha=alpha, beta=beta)
        self.smooth = smooth
        self.lambda_fg = 10
        self.epsilon = 1e-5
        self.ddp = ddp
    
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
    
    @torch.no_grad()
    def compute_weighting_map(self, seg: torch.Tensor, cl: torch.Tensor):
        dist_ratio = self.compute_dist_ratio(seg, cl) 
        alpha = (-self.lambda_fg * torch.log(dist_ratio+self.epsilon)) * seg + (1-seg)
        return alpha

    def forward(self, input: torch.Tensor, target: torch.Tensor, target_ce: torch.Tensor, cl: torch.Tensor, # added cl
                lambda1: float = 1., lambda2: float = 1.):
        # weight_pixelmap is the distance transform map.
        weight_pixelmap = self.compute_weighting_map(target, cl)
        print(f"weight pixelmap shape: {weight_pixelmap.shape}") # [B, B, D, H, W]

        y1 = self.tversky(input, target)
        y2 = torch.mean(torch.mul(self.crossentropy_loss(input, target_ce.long().squeeze(1)), weight_pixelmap))
        CAS_com = lambda1 * y1 + lambda2 * y2
        return CAS_com


class CAS_COR(nn.Module):
    def __init__(self, device, weights=None, *args, **kwargs):
        super(CAS_COR, self).__init__()
        self.device = device
        self.precision_range_lower = precision_range_lower = 0.001
        self.precision_range_upper = precision_range_upper = 1.0
        self.num_classes = 2 # C
        self.num_anchors = 10 # A

        self.precision_range = (
            self.precision_range_lower,
            self.precision_range_upper,
        )
        self.precision_values, self.delta = range_to_anchors_and_delta(
            self.precision_range, self.num_anchors, self.device
        )
        self.biases = nn.Parameter(
            FloatTensor(self.device, self.num_classes, self.num_anchors).zero_()
        )
        self.lambdas = nn.Parameter(
            FloatTensor(self.device, self.num_classes, self.num_anchors).data.fill_(
                1.0
            )
        )

    def forward(self, logits, targets, reduce=True, size_average=True, weights=None):
        logits = logits.view(-1, logits.shape[1]) # flatten -> [B*D*H*W,C]
        targets = targets.long().contiguous().view(-1)
        C = 1 if logits.dim() == 1 else logits.size(1)
        labels, weights = CAS_COR._prepare_labels_weights(
            logits, targets, device=self.device, weights=weights
        ) # labels: [B*D*H*W,C], weights: [B*D*H*W,1]
        lambdas = lagrange_multiplier(self.lambdas) # [C, A]
        hinge_loss = weighted_hinge_loss(
            labels.unsqueeze(-1),
            logits.unsqueeze(-1) - self.biases,
            positive_weights=1.0 + lambdas * (1.0 - self.precision_values),
            negative_weights=lambdas * self.precision_values
        ) # [B*D*H*W, C, A]
        class_priors = build_class_priors(labels, weights=weights) # [C]
        lambda_term = class_priors.unsqueeze(-1) * (
                lambdas * (1.0 - self.precision_values)
        ) # [C, A]
        per_anchor_loss = weights.unsqueeze(-1) * hinge_loss - lambda_term # [B*D*H*W, C, A]
        loss = per_anchor_loss.sum(2) * self.delta
        loss /= self.precision_range[1] - self.precision_range[0]

        if not reduce:
            return loss
        elif size_average:
            return loss.mean() # scalar
        else:
            return loss.sum()

    @staticmethod
    def _prepare_labels_weights(logits, targets, device, weights=None):
        N, C = logits.size()
        print(f"N: {N}, C: {C}")
        print(f"logits: {logits.shape}, targets: {targets.shape}")
        labels = FloatTensor(device, N, C).zero_().scatter(1, targets.unsqueeze(1).data, 1)
        if weights is None:
            weights = FloatTensor(device, N).data.fill_(1.0)
        if weights.dim() == 1:
            weights = weights.unsqueeze(-1)
        return labels, weights


class CAS(nn.Module):
    def __init__(self, device):
        super(CAS, self).__init__()
        self.COM = CAS_COM(apply_nonlin=torch.sigmoid)
        self.COR = CAS_COR(device)
        self.lambda1 = 1.
        self.lambda2 = 1.

    def forward(self, input, target, target_ce, cl): # moved lambda to init
        self.loss_1 = self.COM(input, target, target_ce, cl) # replaced weight_pixelmap with cl
        self.loss_2 = self.COR(input, target_ce)
        return self.lambda1 * self.loss_1 + self.lambda2 * self.loss_2

def sum_tensor(inp, axes, keepdim=False):
    axes = np.unique(axes).astype(int)
    if keepdim:
        for ax in axes:
            inp = inp.sum(int(ax), keepdim=True)
    else:
        for ax in sorted(axes, reverse=True):
            inp = inp.sum(int(ax))
    return inp


def get_tp_fp_fn(net_output, gt, axes=None, mask=None, square=False):
    if axes is None:
        axes = tuple(range(2, len(net_output.size())))

    shp_x = net_output.shape
    shp_y = gt.shape

    with torch.no_grad():
        if len(shp_x) != len(shp_y):
            gt = gt.view((shp_y[0], 1, *shp_y[1:]))

        if all([i == j for i, j in zip(net_output.shape, gt.shape)]):
            y_onehot = gt
        else:
            gt = gt.long()
            y_onehot = torch.zeros(shp_x)
            if net_output.device.type == "cuda":
                y_onehot = y_onehot.cuda(net_output.device.index)
            y_onehot.scatter_(1, gt, 1)

    tp = net_output * y_onehot
    fp = net_output * (1 - y_onehot)
    fn = (1 - net_output) * y_onehot

    if mask is not None:
        tp = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(tp, dim=1)), dim=1)
        fp = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(fp, dim=1)), dim=1)
        fn = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(fn, dim=1)), dim=1)

    if square:
        tp = tp ** 2
        fp = fp ** 2
        fn = fn ** 2

    tp = sum_tensor(tp, axes, keepdim=False)
    fp = sum_tensor(fp, axes, keepdim=False)
    fn = sum_tensor(fn, axes, keepdim=False)

    return tp, fp, fn


CUDA_ENABLED = True


def FloatTensor(device, *args):
    if CUDA_ENABLED:
        return torch.FloatTensor(*args).to(device)
    else:
        return torch.FloatTensor(*args)


def range_to_anchors_and_delta(precision_range, num_anchors, device):
    precision_values = np.linspace(
        start=precision_range[0], stop=precision_range[1], num=num_anchors + 1
    )[1:]

    delta = (precision_range[1] - precision_range[0]) / num_anchors
    return FloatTensor(device, precision_values), delta


def build_class_priors(
        labels,
        class_priors=None,
        weights=None,
        positive_pseudocount=1.0,
        negative_pseudocount=1.0,
):
    if class_priors is not None:
        return class_priors
    N, C = labels.size()
    weighted_label_counts = (weights * labels).sum(0)
    weight_sum = weights.sum(0)
    class_priors = torch.div(
        weighted_label_counts + positive_pseudocount,
        weight_sum + positive_pseudocount + negative_pseudocount
    )
    return class_priors


def weighted_hinge_loss(labels, logits, positive_weights=1.0, negative_weights=1.0):
    positive_term = (1 - logits).clamp(min=0) * labels
    negative_term = (1 + logits).clamp(min=0) * (1 - labels)
    positive_weights_is_tensor = torch.is_tensor(positive_weights)
    if positive_weights_is_tensor and positive_term.dim() == 2:
        return (
                positive_term.unsqueeze(-1) * positive_weights
                + negative_term.unsqueeze(-1) * negative_weights
        )
    else:
        return positive_term * positive_weights + negative_term * negative_weights


def true_positives_lower_bound(labels, logits, weights):
    loss_on_positives = weighted_hinge_loss(labels, logits, negative_weights=0.0)
    weighted_loss_on_positives = (
        weights.unsqueeze(-1) * (labels - loss_on_positives)
        if loss_on_positives.dim() > weights.dim()
        else weights * (labels - loss_on_positives)
    )
    return weighted_loss_on_positives.sum(0)


def false_postives_upper_bound(labels, logits, weights):
    loss_on_negatives = weighted_hinge_loss(labels, logits, positive_weights=0)
    weighted_loss_on_negatives = (
        weights.unsqueeze(-1) * loss_on_negatives
        if loss_on_negatives.dim() > weights.dim()
        else weights * loss_on_negatives
    )
    return weighted_loss_on_negatives.sum(0)


class LagrangeMultiplier(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg()


def lagrange_multiplier(x):
    return LagrangeMultiplier.apply(x)

