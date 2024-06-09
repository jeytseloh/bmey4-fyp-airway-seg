# Adopted from FANN
# https://github.com/Nandayang/YangLab_FANN/blob/main/Compute_metric.py

import numpy as np
import os
import re
import nibabel as nib
from skimage.morphology import skeletonize_3d
# from eval_metrics import dice_similarity_coefficient, false_positive_rate, branches_detected, tree_length_detected, false_negative_rate
import SimpleITK as sitk
import csv
import skimage.measure as measure

# from eval_utils import get_parsing
from metrics import evaluate_airway_metrics


def compute_metrics(gt_path: str, pred_path: str, cl_path: str, save_csv: bool, 
                    save_path: str, model: str, tsDataset: str):
    """
    pred: {ATM/Aero}_{FID}.nii.gz (eval_results)
    gt: {ATM/Aero}_{FID}.nii.gz (eval_dataset)
    cl: {ATM/Aero}_{FID}_cl.nii.gz (eval_dataset)
    """
    pred_fids = os.listdir(pred_path) # {ATM/Aero}_{FID}.nii.gz
    metrics = []

    for pred_fid in pred_fids:
        if pred_fid.endswith(".nii.gz") and not pred_fid.startswith("._"):
            filename = pred_fid.split(".nii.gz")
            gt_fid = f"{filename[0]}.nii.gz"
            cl_fid = f"{filename[0]}_cl.nii.gz"
            print("Assessing ", f"{filename[0]}.nii.gz")

            gt = sitk.ReadImage(os.path.join(gt_path, gt_fid))
            gt_array = sitk.GetArrayFromImage(gt)
            gt_voi = np.where(gt_array > 0) # volume of interest
            z, y, x = gt_array.shape
            # get min and max coordinates of VOI
            z_min, z_max = min(gt_voi[0]), max(gt_voi[0])
            y_min, y_max = min(gt_voi[1]), max(gt_voi[1])
            x_min, x_max = min(gt_voi[2]), max(gt_voi[2])
            z_min, z_max = max(0,z_min-20), z_max
            y_min, y_max = max(0, y_min - 20), min(y, y_max + 20)
            x_min, x_max = max(0, x_min - 20), min(x, x_max + 20)

            try:
                pred = sitk.ReadImage(os.path.join(pred_path, pred_fid))
            except:
                print("Pred file not found: ", pred_fid)
                continue

            try:
                cl = sitk.ReadImage(os.path.join(cl_path, cl_fid))
            except:
                print("Cl file not found: ", cl_fid)
                continue
            
            pred_array = sitk.GetArrayFromImage(pred)
            cl_array = sitk.GetArrayFromImage(cl)
            gt_ = gt_array[z_min:z_max,y_min:y_max,x_min:x_max]
            pred_ = pred_array[z_min:z_max,y_min:y_max,x_min:x_max]
            cl_ = cl_array[z_min:z_max,y_min:y_max,x_min:x_max]

            result = evaluate_airway_metrics(pred_fid, pred_, gt_, cl_)
            # jac, dice_score, TD, BD, precision, ALR, FNR, CCF, large_cd
            if result is not None:
                metrics.append(result[:-1]) # exclude large_cd
                # save metrics for each image to csv
                if save_csv:
                    file_path = save_path + f"_{model}.csv"
                    file_exists = os.path.exists(file_path) and os.path.getsize(file_path) > 0
                    with open(save_path + f"_{model}.csv", 'a+', newline='') as f:
                        writer = csv.writer(f)
                        if not file_exists: # column names
                            writer.writerow(["FID", "IoU", "DSC", "TD", "BD", "Pre", "ALR", "AMR", "FPR", "FNR", "CCF"])
                        writer.writerow([pred_fid] + list(result[:-1]))
                        f.close()
            
            print(f"{filename[0]}_{filename[1]}", 
                  " IoU: ", result[0], 
                  "DSC: ", result[1], 
                  "TD : ", result[2],
                   "BD: ", result[3], 
                   "Precision: ", result[4], 
                   "ALR: ", result[5], 
                   "AMR: ", result[6], 
                   "FPR: ", result[7],
                   "FNR: ", result[8],
                   "CCF: ", result[9])
    
    metrics_array = np.array([[float(value) if value is not None else np.nan for value in metric] for metric in metrics])
    metrics_mean = np.nanmean(metrics_array, axis=0)
    metrics_std = np.nanstd(metrics_array, axis=0)
    print("************** Overall metric: **********\n IoU, DSC, TD, BD, Pre, ALR, FNR, CCF\n", metrics_mean)
    print(metrics_std)

    # save summary metrics to txt file
    with open(save_path + f"summary_{model}_{tsDataset}.txt", 'w') as f:
        f.write("Mean (IoU, DSC, TD, BD, Pre, ALR, AMR, FPR, FNR, CCF):\n")
        f.write(", ".join(map(str, metrics_mean)) + "\n")
        f.write("Standard Deviation (IoU, DSC, TD, BD, Pre, ALR, AMR, FPR, FNR, CCF):\n")
        f.write(", ".join(map(str, metrics_std)) + "\n")
        f.close()


if __name__ == "__main__":

    # Dataset111_ATM22Aero, Dataset112_Aero, Dataset113_SmallATM22, Dataset114_FullATM22
    tsDataset = ["Aero", "ATM22"] # dataset used to evaluate model on {Aero, ATM22}
    root_dir = "/Volumes/TOSHIBA EXT/FYP/"
    model = "Untuned"

    # evaluate on Aero test dataset
    gt_path_aero = os.path.join(root_dir, "eval_dataset_full", "Aero_gt")
    pred_path_aero = save_path_aero = os.path.join(root_dir, "Results_v2", "MedSAM", model, tsDataset[0])
    cl_path_aero = os.path.join(root_dir, "eval_dataset_full", "Aero_cl")
    compute_metrics(gt_path_aero, pred_path_aero, cl_path_aero, True, save_path_aero, model, tsDataset[0])
    print(f"Evaluation on {tsDataset[0]} for {model} done.")

    # evalute on ATM22 test dataset
    gt_path_atm22 = os.path.join(root_dir, "eval_dataset_full", "ATM22_gt")
    pred_path_atm22 = save_path_atm22 = os.path.join(root_dir, "Results_v2", "MedSAM", model, tsDataset[1])
    cl_path_atm22 = os.path.join(root_dir, "eval_dataset_full", "ATM22_cl")
    compute_metrics(gt_path_atm22, pred_path_atm22, cl_path_atm22, True, save_path_atm22, model, tsDataset[1])
    print(f"Evaluation on {tsDataset[1]} for {model} done.")