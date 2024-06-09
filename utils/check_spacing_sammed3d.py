import SimpleITK as sitk
import os
import numpy as np

def check_spacing(volume, new_spacing=[1., 1., 1.]):
    """
    Check spacing and resample volume to new spacing
    """
    original_spacing = volume.GetSpacing()
    original_size = volume.GetSize()
    new_size = [int(round(osz * ospc / nspc)) for osz, ospc, nspc in zip(original_size, original_spacing, new_spacing)]
    return sitk.Resample(volume, new_size, sitk.Transform(), sitk.sitkLinear,
                  volume.GetOrigin(), new_spacing, volume.GetDirection(), 0, volume.GetPixelID())

if __name__ == '__main__':
    # img_dir = "/Volumes/TOSHIBA EXT/FYP/ATM22_Aero_Dataset/train/imagesTr/"
    gt_dir = "/Volumes/TOSHIBA EXT/FYP/eval_dataset_full/Aero_gt/"
    cl_dir = "/Volumes/TOSHIBA EXT/FYP/eval_dataset_full/Aero_cl/"
    # output_img_dir = "/Volumes/TOSHIBA EXT/FYP//ATM22AeroFull_resampled/imagesTr"
    output_gt_dir = "/Volumes/TOSHIBA EXT/FYP/SAM-Med3D/SAM-Med3D_raw/Aero_test_resampled/Aero_gt_resampled"
    output_cl_dir = "/Volumes/TOSHIBA EXT/FYP/SAM-Med3D/SAM-Med3D_raw/Aero_test_resampled/Aero_cl_resampled"

    # image_files = os.listdir(img_dir)
    gt_files = [f for f in os.listdir(gt_dir) if not f.startswith('._')]
    cl_files = [f for f in os.listdir(cl_dir) if not f.startswith('._')]

    # image_files.sort()
    gt_files.sort()
    cl_files.sort()

    assert len(gt_files) == len(cl_files), "The number of image files and label files are not equal."

    # for img, gt in zip(image_files, gt_files):
    for gt, cl in zip(gt_files, cl_files):
        # img_path = os.path.join(img_dir, img)
        gt_path = os.path.join(gt_dir, gt)
        cl_path = os.path.join(cl_dir, cl)

        # volume = sitk.ReadImage(img_path)
        label = sitk.ReadImage(gt_path)
        cl = sitk.ReadImage(cl_path)
        # print(f"Volume: {img}")
        # print("original spacing:",  volume.GetSpacing(), ',size:', sitk.GetArrayFromImage(volume).shape)
        print(f"Label: {gt}")
        print("original spacing:",  label.GetSpacing(), ',size:', sitk.GetArrayFromImage(label).shape)
        print(f"CL: {cl}")
        print("original spacing:",  cl.GetSpacing(), ',size:', sitk.GetArrayFromImage(cl).shape)

        # resample to 1.5mm spacing for SAM-Med3D input
        # new_volume = check_spacing(volume, [1.5,1.5,1.5])
        new_label = check_spacing(label, [1.5,1.5,1.5])
        new_cl = check_spacing(cl, [1.5,1.5,1.5])

        # print("resampled spacing (volume):", new_volume.GetSpacing(), ',size:', sitk.GetArrayFromImage(new_volume).shape)
        print("resampled spacing (label):", new_label.GetSpacing(), ',size:', sitk.GetArrayFromImage(new_label).shape)
        print("resampled spacing (CL):", new_cl.GetSpacing(), ',size:', sitk.GetArrayFromImage(new_cl).shape)

        # resampled_img_path = os.path.join(output_img_dir, img)
        resampled_gt_output = os.path.join(output_gt_dir, gt)
        resampled_cl_output = os.path.join(output_cl_dir, cl)

        # sitk.WriteImage(new_volume, resampled_img_path)
        sitk.WriteImage(new_label, resampled_gt_output)
        sitk.WriteImage(new_cl, resampled_cl_output)
        # print(f"Saved resampled volume to {resampled_img_path}")
        print(f"Saved resampled label to {resampled_gt_output}")
        print(f"Saved resampled CL to {resampled_cl_output}")