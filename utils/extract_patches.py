# SPS from FANN: https://github.com/Nandayang/YangLab_FANN/blob/main/data/extract_patches.py

import numpy as np
import SimpleITK as sitk
import os
from tqdm import tqdm
from tree_parse import get_parsing
from skimage.morphology import skeletonize_3d
import matplotlib.pyplot as plt
import json

def zscore_normalization(images, p995,p005, mean=None, std=None):
    if mean==None or std==None:
        raise Exception("compute the mean and std of training set first!")
    images = np.clip(images, p005, p995)
    images = (images - mean)/std
    return images

def adjust_window(image, window_center=-300, window_width=1800):
    win_min = window_center - window_width // 2
    win_max = window_center + window_width // 2
    image = 255.0 * (image - win_min) / (win_max - win_min)
    image[image>255] = 255
    image[image<0] = 0
    return image

def smart_patch_sampling(img_path, label_path, cl_path, centerline=True):
    patch_size = [128, 96, 144] # from FANN
    root_dir = "/Volumes/TOSHIBA EXT/FYP/ExtractPatches2/"
    metadata = {} # to save metadat for reconstruction

    if centerline:
        dir_list = ['img', 'label', 'cl', 'cl_thin']
    else:
        dir_list = ['img', 'label']
    for dir in dir_list:
        if not os.path.exists(os.path.join(root_dir, "Train", dir)):
            os.makedirs(os.path.join(root_dir, "Train", dir))
        if not os.path.exists(os.path.join(root_dir, "Test", dir)):
            os.makedirs(os.path.join(root_dir, "Test", dir))
        print("Directory " + dir + " created.")


    path_save_img = os.path.join(root_dir, "Train", "img")
    path_save_label = os.path.join(root_dir, "Train", "label")
    # path_save_cl = os.path.join(root_dir, "Train", "cl")
    # path_save_cl_thin = os.path.join(root_dir, "Train", "cl_thin")

    fids = [f for f in os.listdir(label_path) if not f.startswith('.')]
    for fid in tqdm(fids):
        cid = fid.split(".nii.gz")[0]
        case_metadata = metadata.setdefault(cid, {"patches": [], "volume_dimensions": None})
        
        volume_img = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(img_path, cid + "_0000.nii.gz")))
        volume_img = adjust_window(volume_img, window_center=-300, window_width=1800)
        volume_label = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(label_path, fid)))
        gt_v = np.sum(volume_label)
        # print(gt_v)

        case_metadata["volume_dimensions"] = list(volume_img.shape)

        if centerline:
            volume_cl = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(cl_path, cid + "_cl.nii.gz")))
            volume_cl_thin = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(cl_path, cid + "_cl.nii.gz")))
            cl_v = np.sum(volume_cl)
            # print(cl_v)
        assert volume_img.shape == volume_label.shape
        # print(volume_img.shape)
        z, y, x = volume_img.shape
        index = 0
        # overlapping patches
        for i in range(0,z,64):
            for j in range(0,y,48):
                for k in range(0,x,72):
                    if z < i+patch_size[0] or y < j + patch_size[1] or x < k + patch_size[2]:
                        continue
                    z_start_idex = min(i,z-patch_size[0])
                    z_end_idex = min(z, i+patch_size[0])
                    y_start_idex = min(j, y - patch_size[1])
                    y_end_idex = min(y, j + patch_size[1])
                    x_start_idex = min(k, x - patch_size[2])
                    x_end_idex = min(x, k + patch_size[2])

                    # starting coordinates - for metadata
                    coords = [z_start_idex, y_start_idex, x_start_idex]
                    # size of patch to be extracted - for metadata
                    patch_dims = [z_end_idex - z_start_idex,
                                  y_end_idex - y_start_idex,
                                  x_end_idex - x_start_idex]

                    patch_img = volume_img[z_start_idex:z_end_idex,
                                        y_start_idex:y_end_idex,
                                        x_start_idex:x_end_idex]
                    patch_label = volume_label[z_start_idex:z_end_idex,
                                            y_start_idex:y_end_idex,
                                            x_start_idex:x_end_idex]
                    
                    if patch_label.shape != tuple(patch_size) or patch_img.shape != tuple(patch_size):
                        print('wrong')
                        continue

                    if centerline:
                        patch_cl = volume_cl[z_start_idex:z_end_idex,y_start_idex:y_end_idex,x_start_idex:x_end_idex]
                        patch_cl_thin = volume_cl_thin[z_start_idex:z_end_idex,y_start_idex:y_end_idex,x_start_idex:x_end_idex]
                        # keep or discard patch based on cl or volume ratio
                        if np.sum(patch_cl) < cl_v*0.15 and np.sum(patch_label)<gt_v*0.10:
                            continue

                        # cl_out = sitk.GetImageFromArray(patch_cl)
                        # cl_out_thin = sitk.GetImageFromArray(patch_cl_thin)

                    img_out = sitk.GetImageFromArray(patch_img)
                    label_out = sitk.GetImageFromArray(patch_label)

                    sitk.WriteImage(img_out, path_save_img + cid + "_img_{}.nii.gz".format(index))
                    sitk.WriteImage(label_out, path_save_label + cid + "_label_{}.nii.gz".format(index))

                    # metadata for reconstruction
                    case_metadata["patches"].append({
                        "patch_index": index,
                        "coordinates": coords,
                        "dimensions": patch_size
                    })

                    index +=1
        
        # save metadata to json file
        with open(os.path.join(root_dir, "Train", "metadata.json"), 'w') as f:
            json.dump(metadata, f, indent=4)
        
        print("patches extracted for " + cid)

def test_cl_extraction(img_path, label_path, output_path):
    image = sitk.GetArrayFromImage(sitk.ReadImage(img_path))
    label = sitk.GetArrayFromImage(sitk.ReadImage(label_path))

    label = (label > 0).astype(np.uint8)

    # cl = get_parsing(label, refine=False)
    # cl_sitk = sitk.GetImageFromArray(cl)
    skeleton = skeletonize_3d(label)
    skeleton_sitk = sitk.GetImageFromArray(skeleton)

    # sitk.WriteImage(cl_sitk, output_path)
    sitk.WriteImage(skeleton_sitk, output_path)

    print("centreline extracted")

def extract_cl(label_path, flag):
    fids = [f for f in os.listdir(label_path) if not f.startswith('.')]
    
    for fid in tqdm(fids):
        cid = fid.split(".nii.gz")[0]
        save_path = "/Volumes/Expansion/FYP/ATM22_Aero_Dataset/{}/cl/{}_cl.nii.gz".format(flag,cid)
        label = sitk.GetArrayFromImage(sitk.ReadImage(label_path + fid))
        label = (label > 0).astype(np.uint8)
        skeleton_sitk = sitk.GetImageFromArray(skeletonize_3d(label))
        sitk.WriteImage(skeleton_sitk, save_path)
        print("Centreline extracted for " + cid)


if __name__ == "__main__":
    root_dir = "/Volumes/TOSHIBA EXT/FYP/ATM22_Aero_Dataset/"

    imgTr_path = os.path.join(root_dir, "train", "imagesTr")
    labelTr_path = os.path.join(root_dir, "train", "labelsTr")
    clTr_path = os.path.join(root_dir, "train", "cl")

    imgTs_path = os.path.join(root_dir, "test", "imagesTs")
    labelTs_path = os.path.join(root_dir, "test", "labelsTs")
    clTs_path = os.path.join(root_dir, "test", "cl")

    smart_patch_sampling(imgTr_path, labelTr_path, clTr_path) # train
    smart_patch_sampling(imgTs_path, labelTs_path, clTs_path) # test

