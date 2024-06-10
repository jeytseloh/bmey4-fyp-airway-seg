#!/bin/bash

export nnUNet_raw="PATH_TO_nnUNet_raw"
export nnUNet_preprocessed="PATH_TO_nnUNet_preprocessed"
export nnUNet_results="PATH_TO_nnUNet_results"

nnUNetv2_preprocess -d DATASET_ID -c 3d_fullres_cl_preprocessor --verify_dataset_integrity
