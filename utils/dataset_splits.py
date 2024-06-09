from sklearn.model_selection import train_test_split
import os
from tqdm import tqdm
import shutil
import json
import random


atm22_img_path = "/Volumes/Expansion/ATM'22 Dataset/TrainAll/imagesTr"
atm22_label_path = "/Volumes/Expansion/ATM'22 Dataset/TrainAll/labelsTr"
aeropath_path = "/Volumes/Expansion/AeroPath"

def atm22_splits():
    """
    Splits ATM22 dataset into 80:20 train:test
    """
    atm22_fids = [f for f in os.listdir(atm22_img_path) if not f.startswith('.')]
    atm22_train, atm22_test = train_test_split(atm22_fids, test_size=0.2, train_size=0.8, random_state=0)

    # print(len(atm22_train))
    # print(len(atm22_test))

    for fid in tqdm(atm22_fids):
        if fid in atm22_test:
            split_flag = "test"
            abbrev = "Ts"
            print("Moved to test folder")
        else: # if fid in atm22_train
            split_flag = "train"
            abbrev = "Tr"
            print("Moved to train folder")
        path_save_img = "/Volumes/Expansion/FYP/ATM22_Aero_Dataset/{}/images{}/".format(split_flag, abbrev)
        path_save_label = "/Volumes/Expansion/FYP/ATM22_Aero_Dataset/{}/labels{}/".format(split_flag, abbrev)

        os.makedirs(path_save_img, exist_ok=True)
        os.makedirs(path_save_label, exist_ok=True)
        shutil.copy(os.path.join(atm22_img_path, fid), path_save_img)
        shutil.copy(os.path.join(atm22_label_path, fid), path_save_label)

def aero_splits():
    aero_cases = [d for d in os.listdir(aeropath_path) 
                  if os.path.isdir(os.path.join(aeropath_path, d)) 
                  if not d.startswith('.')]
    aero_train, aero_test = train_test_split(aero_cases, test_size=0.2, train_size=0.8, random_state=0)

    print(aero_train)
    print(aero_test)
    print(len(aero_train))
    print(len(aero_test))

    for cid in tqdm(aero_cases):
        if cid in aero_test:
            split_flag = "test"
            abbrev = "Ts"
            print("Moved to test folder")
        else: # if fid in atm22_train
                split_flag = "train"
                abbrev = "Tr"
                print("Moved to train folder")
        path_save_img = "/Volumes/Expansion/FYP/ATM22_Aero_Dataset/{}/images{}/".format(split_flag, abbrev)
        path_save_label = "/Volumes/Expansion/FYP/ATM22_Aero_Dataset/{}/labels{}/".format(split_flag, abbrev)

        os.makedirs(path_save_img, exist_ok=True)
        os.makedirs(path_save_label, exist_ok=True)

        for f in os.listdir(os.path.join(aeropath_path, cid)):
            if "airways" in f:
                label_filename = "{}/{}_CT_HR_label_airways.nii.gz".format(cid, cid)
                shutil.copy(os.path.join(aeropath_path, label_filename), path_save_label)
            elif "CT_HR.nii.gz" in f:
                img_filename = "{}/{}_CT_HR.nii.gz".format(cid, cid)
                shutil.copy(os.path.join(aeropath_path, img_filename), path_save_img)

def rename_img_files(directory):
    """
    Rename image files to match nnunet format: {Aero/ATM}_{CID}_0000.nii.gz
    """
    fids = [f for f in os.listdir(directory) if not f.startswith('.')]
    # print(fids)
    # print(len(fids))
    for fid in tqdm(fids):
        if not fid.startswith("ATM") and fid.endswith(".nii.gz"):
            cid = fid.split("_")[0].zfill(3) # 3 digits
            # print(cid)
            new_fid = "Aero_{}_0000.nii.gz".format(cid)
            os.rename(os.path.join(directory, fid), os.path.join(directory, new_fid))
            print(f"Renamed ''{fid}' to '{new_fid}'")

def rename_label_files(directory):
    """
    Rename label files to match nnunet format: {Aero/ATM}_{CID}.nii.gz
    """
    fids = [f for f in os.listdir(directory) if not f.startswith('.')]
    for fid in tqdm(fids):
        if fid.endswith("_0000.nii.gz"): # rename ATM22 labels
            new_fid = fid.replace("_0000", "")
            os.rename(os.path.join(directory, fid), os.path.join(directory, new_fid))
            print(f"Renamed ''{fid}' to '{new_fid}'")
        elif fid.endswith("_CT_HR_label_airways.nii.gz"):
            cid = fid.split("_")[0].zfill(3) # 3 digits
            new_fid = "Aero_{}.nii.gz".format(cid)
            os.rename(os.path.join(directory, fid), os.path.join(directory, new_fid))
            print(f"Renamed ''{fid}' to '{new_fid}'")

# def small_atm22_splits(json_file, num_train, num_test, output_file):
#     with open (json_file, 'r') as f:
#         image_files = json.load(f)
    
#     atm22_train = [f for f in image_files['train'] if f.startswith('ATM')]
#     atm22_test = [f for f in image_files['test'] if f.startswith('ATM')]
    
#     trainSelected = random.sample(atm22_train, num_train)
#     testSelected = random.sample(atm22_test, num_test)

#     selectedFiles = {
#         'train': trainSelected,
#         'test': testSelected
#     }

#     with open(output_file, 'w') as f:
#         json.dump(selectedFiles, f, indent=4)

#     print(f"Selected filenames written to {output_file}")

if __name__ == "__main__":
    atm22_splits()
    aero_splits()

    rename_img_files("/Volumes/Expansion/FYP/ATM22_Aero_Dataset/train/imagesTr/")
    rename_label_files("/Volumes/Expansion/FYP/ATM22_Aero_Dataset/train/labelsTr/")

    rename_img_files("/Volumes/Expansion/FYP/ATM22_Aero_Dataset/test/imagesTs/")
    rename_label_files("/Volumes/Expansion/FYP/ATM22_Aero_Dataset/test/labelsTs/")

    # jsonFile = "/Volumes/Expansion/FYP/ATM22_Aero_Dataset/dataset_split.json"
    # outputFile = "/Volumes/Expansion/FYP/ATM22_Aero_Dataset/small_atm22_split.json"
    # small_atm22_splits(jsonFile, 21, 6, outputFile)