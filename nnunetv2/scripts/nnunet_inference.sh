#!/bin/bash

export nnUNet_raw="PATH_TO_nnUNet_raw"
export nnUNet_preprocessed="PATH_TO_nnUNet_preprocessed"
export nnUNet_results="PATH_TO_nnUNet_results"

nnUNetv2_predict -i PATH_TO_INPUT_FOLDER \
        -o PATH_TO_OUTPUT_FOLDER \
        -d DATASET_NAME_OR_ID \
        -tr nnUNetTrainer_LOSS \
        -c 3d_fullres_cl_preprocessor \
        -chk checkpoint_final.pth \
        -f 0 1 2 3 4
