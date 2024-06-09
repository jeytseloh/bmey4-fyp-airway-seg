import SimpleITK as sitk
import json
import os

def resample_volume_to_original_size(segmentation, original_size):
    current_size = segmentation.GetSize()
    
    # no adjustment needed
    if current_size == tuple(original_size):
        return segmentation

    # adjust segmentation to match original size
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(segmentation.GetSpacing())
    resampler.SetSize(original_size)
    resampler.SetOutputDirection(segmentation.GetDirection())
    resampler.SetOutputOrigin(segmentation.GetOrigin())
    resampler.SetTransform(sitk.Transform())
    resampler.SetDefaultPixelValue(segmentation.GetPixelIDValue())
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)  # nearest neighbour for segmentation

    adjusted_segmentation = resampler.Execute(segmentation)
    return adjusted_segmentation

def resample_volume_to_original_spacing(segmentation_path, original_spacing, original_size, output_path):
    segmentation = sitk.ReadImage(segmentation_path)
    current_spacing = segmentation.GetSpacing()
    current_size = segmentation.GetSize()
    
    new_size = [int(round(cs * cs_spacing / os_spacing)) \
                for cs, cs_spacing, os_spacing in zip(current_size, current_spacing, original_spacing)]
    
    resampled_segmentation = sitk.Resample(
        segmentation, 
        new_size, 
        sitk.Transform(), 
        sitk.sitkNearestNeighbor,  # Using nearest neighbor for segmentation
        segmentation.GetOrigin(), 
        original_spacing, 
        segmentation.GetDirection(), 
        0,  # Padding value
        segmentation.GetPixelID()
    )

    # to ensure output matches original size
    final_segmentation = resample_volume_to_original_size(resampled_segmentation, original_size)

    print(f"Resampled segmentation from size {current_size} and spacing {current_spacing} "
          f"to size {new_size} and spacing {original_spacing}, "
          f"then adjusted to final size {original_size}")
    
    sitk.WriteImage(final_segmentation, output_path)
    print(f"Resampled segmentation saved to {output_path}")

def process_all_segmentations(segmentation_dir, metadata_path, output_dir):
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    os.makedirs(output_dir, exist_ok=True)
    
    for filename in os.listdir(segmentation_dir):
        if filename.endswith("_pred0.nii.gz") and not filename.startswith("._"):
            segmentation_path = os.path.join(segmentation_dir, filename)
            base_name = filename.replace("_pred0.nii.gz", "_0000.nii.gz") # specific to SAM-Med3D inference output naming convention
            if base_name in metadata:
                original_spacing = metadata[base_name]["original_volume_spacing"]
                original_size = metadata[base_name]["original_volume_size"]
                output_path = os.path.join(output_dir, filename.replace("_pred0.nii.gz", ".nii.gz"))
                resample_volume_to_original_spacing(segmentation_path, original_spacing, original_size, output_path)
                print(f"Resampled and saved {filename} to {output_path}")
            else:
                print(f"Metadata for {base_name} not found.")


if __name__ == '__main__':
    results_root_dir = "/Volumes/TOSHIBA EXT/FYP/SAM-Med3D/SAM-Med3D_results/"
    raw_root_dir = "/Volumes/TOSHIBA EXT/FYP/SAM-Med3D/SAM-Med3D_raw/"
    model = "Turbo_untuned"
    segmentation_dir = os.path.join(results_root_dir, model, "ATM22Aero")
    aero_metadata_path = os.path.join(raw_root_dir, "Aero_test_resampled", "metadata.json")
    atm22_metadata_path = os.path.join(raw_root_dir, "ATM22_test_resampled", "metadata.json")
    output_dir = os.path.join(results_root_dir, model, "ATM22_ori_spacing")

    # aero
    process_all_segmentations(segmentation_dir, aero_metadata_path, output_dir)

    #atm22
    process_all_segmentations(segmentation_dir, atm22_metadata_path, output_dir)