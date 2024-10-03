#!/bin/bash

python inference.py \
--ckpt_path ohayonguy/PMRF_blind_face_image_restoration \
--ckpt_path_is_huggingface \
--lq_data_path /home/ohayonguy/projects/mmse_rectified_flow/data/celeba_512_validation_lq \
--output_dir ./results_huggingface \
--batch_size 64 \
--num_flow_steps 25
