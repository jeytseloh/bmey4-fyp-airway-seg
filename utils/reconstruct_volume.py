import numpy as np
import SimpleITK as sitk
import json
import os
import argparse

def load_patch(patch_path):
    return sitk.GetArrayFromImage(sitk.ReadImage(patch_path))

def reconstruct_volume(metadata, patch_dir, case_id):
    """
    Reconstruct full volume from patches based on metadata.json file saved during extraction step
    """
    volume_dim = metadata.get("volume_dimensions")
    if volume_dim is None:
        raise ValueError("Metadata does not contain volume dimensions")
    
    full_volume = np.zeros(volume_dim, dtype=np.float32)

    for patch_info in metadata.get("patches", []):
        patch_idx = patch_info.get("patch_index")
        patch_coords = patch_info.get("coordinates")
        patch_data = load_patch(os.path.join(patch_dir, f"{case_id}_{patch_idx}.nii.gz"))
        z, y, x = patch_coords
        dz, dy, dx = patch_info["dimensions"]

        full_volume[z:z+dz, y:y+dy, x:x+dx] = patch_data

    return full_volume

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_dir", 
                        type=str, 
                        required=True, 
                        help="Directory containing predicted patches")
    parser.add_argument("--metadata_path",
                        type=str,
                        required=True,
                        help="Path to metadata file")
    parser.add_argument("--output_path", 
                        type=str, 
                        required=True, 
                        help="Directory to save reconstructed volume")
    args = parser.parse_args()

    # load metadata
    with open(args.metadata_path, "r") as f:
        all_metadata = json.load(f)
    
    fids = list(all_metadata.keys()) # case_ids
    print(f"There are {len(fids)} cases in test dataset")

    for fid in fids:
        case_metadata = all_metadata.get(fid)
        if case_metadata is None:
            raise ValueError(f"Case {fid} not found in metadata")
        full_volume = reconstruct_volume(case_metadata, args.pred_dir, fid)
        # save full volume
        full_vol_img = sitk.GetImageFromArray(full_volume)
        sitk.WriteImage(full_vol_img, os.path.join(args.output_path, f"{fid}.nii.gz"))
        print(f"{fid} reconstructed")


if __name__ == "__main__":
    main()


