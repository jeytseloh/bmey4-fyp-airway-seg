import os
import json
import SimpleITK as sitk

def extract_metadata(volume):
    """
    Save original spacing and size of volume as metadata for resampling after inference
    """
    original_spacing = volume.GetSpacing()
    original_size = volume.GetSize()
    return {
        'original_spacing': original_spacing,
        'original_size': original_size
    }

if __name__ == '__main__':
    img_dir = "/rds/general/user/jtl20/ephemeral/MedSAM_raw/ATM22Aero_train/imagesTr"
    gt_dir = "/rds/general/user/jtl20/ephemeral/MedSAM_raw/ATM22Aero_train/labelsTr"
    metadata_path = "/rds/general/user/jtl20/ephemeral/MedSAM_raw/ATM22Aero_train/metadata.json"

    image_files = os.listdir(img_dir)
    gt_files = os.listdir(gt_dir)

    image_files.sort()
    gt_files.sort()

    assert len(image_files) == len(gt_files), "The number of image files and label files are not equal."

    metadata = {}

    for img, gt in zip(image_files, gt_files):
        print(f"Processing {img}")
        img_path = os.path.join(img_dir, img)
        gt_path = os.path.join(gt_dir, gt)

        volume = sitk.ReadImage(img_path)
        label = sitk.ReadImage(gt_path)

        volume_metadata = extract_metadata(volume)
        label_metadata = extract_metadata(label)

        metadata[img] = {
            'original_volume_spacing': volume_metadata['original_spacing'],
            'original_volume_size': volume_metadata['original_size'],
            'original_label_spacing': label_metadata['original_spacing'],
            'original_label_size': label_metadata['original_size']
        }

    with open(metadata_path, 'w') as f:
        json.dump(metadata, f)

    print(f"Metadata saved to {metadata_path}")