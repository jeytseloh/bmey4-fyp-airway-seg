import os

directory = "/PATH_TO/imagesTr/" # img
# directory = "/PATH_TO/labelsTr/" # label

for filename in os.listdir(directory):
    parts = filename.split("_")
    if filename.endswith(".nii.gz"):
        img_num = parts[3].split(".")[0]
        new_filename = f"{parts[0]}_{parts[1]}_{img_num}_0000.nii.gz" # img
        # new_filename = f"{parts[0]}_{parts[1]}_{img_num}.nii.gz" # label
        old_path = os.path.join(directory, filename)
        new_path = os.path.join(directory, new_filename)

        os.rename(old_path, new_path)
        print(f"Renamed '{filename}' to '{new_filename}'")
