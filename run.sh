#!/bin/bash
SD_INPAINTING_PATH="" # Download from https://huggingface.co/runwayml/stable-diffusion-inpainting/

# Dataset and checkpoint see https://share.multcloud.link/share/74f20132-4269-41d0-8355-0c39106de6b0
DATASET_PATH="" # json file
CHECKPOINT_PATH=""
python roomeditor_test.py \
    --pretrained_sd_inpainting_path $SD_INPAINTING_PATH \
    --dataset_path $DATASET_PATH\
    --batch_size 32 \
    --num_inference_steps 50 \
    --precision "fp16" \
    --guidance_scale 2.0 \
    --save_path "test_roombench" \
    --checkpoint_path $CHECKPOINT_PATH
