#!/bin/bash

export nnUNet_raw="PATH_TO_nnUNet_raw"
export nnUNet_preprocessed="PATH_TO_nnUNet_preprocessed"
export nnUNet_results="PATH_TO_nnUNet_results"

nnUNetv2_train DATASET_NAME_OR_ID 3d_fullres_cl_preprocessor 0 -tr nnUNetTrainer_LOSS
