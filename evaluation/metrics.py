import os
import torch
import numpy as np
import nibabel as nib
import skimage.measure as measure
from skimage.morphology import skeletonize_3d
# from utils.tree_parse import get_parsing
from eval_utils import get_parsing

EPSILON = 1e-32

def jaccard(x, y):
    """
    x: prediction
    y: ground truth
    """
    intersection = np.sum(x * y) + EPSILON
    union = np.sum(x) + np.sum(y)
    jaccard_idx = intersection / (union - intersection + EPSILON)
    return jaccard_idx

def dice(x, y):
    """
    x: prediction
    y: ground truth
    """
    intersection = 2 * x * y
    union = x + y
    dice_score = (np.sum(intersection) + EPSILON) / (np.sum(union) + EPSILON)
    return dice_score

def continuity(x, y_cl):
    intersection = np.sum(x * y_cl) + EPSILON
    cont_idx = intersection / (np.sum(y_cl) + EPSILON)
    return cont_idx

def ccf(x, y, y_cl, w):
    """
    x: pred
    y: gt
    y_cl: gt centreline
    w: preference param [0,1]
    """
    weight = 1 + w**2
    jac = jaccard(x, y)
    cont = continuity(x, y_cl)
    intersection = jac * cont
    union = (w**2 * jac) + cont
    ccf_score = weight * ((intersection + EPSILON) / (union + EPSILON))
    return ccf_score

def evaluate_airway_metrics(fid, pred, gt, gt_cl):
    """
    """
    parsing_gt = get_parsing(gt, refine=False) # compute tree parsing
    cd, num = measure.label(pred, return_num=True, connectivity=1) # find largest connected component

    if num == 0: # no connected components found
        print(f"{fid}: No labeled objects found in prediction. Skipping.")
        return None, None, None, None, None, None, None, None, None, None, None
    
    volume = np.zeros([num])
    for k in range(num):
        volume[k] = ((cd == (k + 1)).astype(np.uint8)).sum()
    volume_sort = np.argsort(volume)
    large_cd = (cd == (volume_sort[-1] + 1)).astype(np.uint8)

    # compute jaccard index
    jac = jaccard(large_cd, gt)

    # to ensure extracted largest component is correct
    jj = -1
    while jac < 0.1:
        print(fid, " failed need post-processing")
        jj -= 1
        if abs(jj) > len(volume_sort):  # Check if jj exceeds bounds
            print(f"{fid}: Exhausted all components. Skipping.")
            return None, None, None, None, None, None, None, None, None, None, None
        large_cd = (cd == (volume_sort[jj] + 1)).astype(np.uint8)
        jac = jaccard(large_cd, gt)
        if jj == -5:
            break
    
    skeleton = skeletonize_3d(gt)
    skeleton = (skeleton > 0)
    skeleton = skeleton.astype('uint8')

    dice_score = dice(large_cd, gt)
    TD = (large_cd * skeleton).sum() / skeleton.sum()
    precision = (large_cd * gt).sum() / large_cd.sum()
    ALR = ((large_cd - gt)==1).sum() / gt.sum() # FP: (pred==1) & (gt==0), ratio of FPs to GT
    AMR = ((gt - large_cd)==1).sum() / gt.sum() # FN: (gt==1) & (pred==0), ratop of FNs to GT
    FPR = ((large_cd - gt)==1).sum() / (gt == 0).sum() # FP: (pred==1) & (gt==0), ratio of FPs to GT
    FNR = ((gt - large_cd)==1).sum() / (gt == 1).sum() # FN: (gt==1) & (pred==0), ratio of FNs to GT

    numBranch = parsing_gt.max()
    detectedNum = 0 
    for i in range(numBranch):
        branchLabel = ((parsing_gt == (i + 1)).astype(np.uint8)) * skeleton
        if (large_cd * branchLabel).sum() / branchLabel.sum() >= 0.8:
            detectedNum += 1
    BD = detectedNum / numBranch

    CCF = ccf(large_cd, gt, gt_cl, 0.9) # following paper (preference param w = 0.9)

    return jac, dice_score, TD, BD, precision, ALR, AMR, FPR, FNR, CCF, large_cd


