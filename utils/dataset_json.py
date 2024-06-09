import os
import json

def generate_json(trainDir, testDir, outputFile):
    """
    Generate JSON file of filenames in train and test directories
    """
    image_files = {'train': [], 'test': []}

    for filename in os.listdir(trainDir):
        if filename.endswith(".nii.gz") and not filename.startswith("._"):
            image_files['train'].append(filename)

    for filename in os.listdir(testDir):
        if filename.endswith(".nii.gz") and not filename.startswith("._"):
            image_files['test'].append(filename)

    with open(outputFile, "w") as f:
        json.dump(image_files, f, indent=4)

    print(f"JSON file {outputFile} created")

if __name__ == "__main__":
    train_dir = "/Volumes/Expansion/FYP/ATM22_Aero_Dataset/train/imagesTr/"
    test_dir = "/Volumes/Expansion/FYP/ATM22_Aero_Dataset/test/imagesTs/"
    output_file = "/Volumes/Expansion/FYP/ATM22_Aero_Dataset/dataset_split.json"

    generate_json(train_dir, test_dir, output_file)