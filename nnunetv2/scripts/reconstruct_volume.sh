#!/bin/bash

python reconstruct_volume.py --pred_dir PATH_TO_NNUNET_INFERENCE_OUTPUT_FOLDER \
                --metadata_path PATH_TO_DATASET/metadata.json \
                --output_path OUTPUT_FOLDER