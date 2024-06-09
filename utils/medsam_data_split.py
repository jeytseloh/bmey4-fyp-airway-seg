import json
import random

def select_filenames(input_json, x, y, a, b, output_json):
    """
    Create split for MedSAM from JSON file of dataset filenames
    x: number of ATM files for training
    y: number of Aero files for training
    a: number of ATM files for testing
    b: number of Aero files for testing
    """
    with open(input_json, 'r') as file:
        data = json.load(file)
    
    train_atm_files = [f for f in data['train'] if f.startswith('ATM')]
    train_aero_files = [f for f in data['train'] if f.startswith('Aero')]
    test_atm_files = [f for f in data['test'] if f.startswith('ATM')]
    test_aero_files = [f for f in data['test'] if f.startswith('Aero')]
    
    selected_train_atm = random.sample(train_atm_files, min(x, len(train_atm_files)))
    selected_train_aero = random.sample(train_aero_files, min(y, len(train_aero_files)))
    
    selected_test_atm = random.sample(test_atm_files, min(a, len(test_atm_files)))
    selected_test_aero = random.sample(test_aero_files, min(b, len(test_aero_files)))
    
    output_data = {
        'train': {
            'ATM': selected_train_atm,
            'Aero': selected_train_aero
        },
        'test': {
            'ATM': selected_test_atm,
            'Aero': selected_test_aero
        }
    }
    
    with open(output_json, 'w') as file:
        json.dump(output_data, file, indent=4)
    
    print(f"Selected {len(selected_train_atm)} ATM and {len(selected_train_aero)} Aero files for train.")
    print(f"Selected {len(selected_test_atm)} ATM and {len(selected_test_aero)} Aero files for test.")
    print(f"Output written to {output_json}")

if __name__ == '__main__':
    input_json = '/Volumes/TOSHIBA EXT/FYP/MedSAM/ATM22Aero_small/dataset_split.json'  # Path to your input JSON file
    atm_tr_num = 30
    aero_tr_num = 6
    atm_ts_num = 10
    aero_ts_num = 10
    output_json = '/Volumes/TOSHIBA EXT/FYP/MedSAM/ATM22Aero_small/medsam_test.json'  # Path to your output JSON file

    select_filenames(input_json, atm_tr_num, aero_tr_num, atm_ts_num, aero_ts_num, output_json)