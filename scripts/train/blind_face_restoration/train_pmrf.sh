#!/bin/bash

python train.py \
--precision "bf16-mixed" \
--stage "flow" \
--degradation "difface" \
--train_data_root "/path/to/data/ffhq512" \
--val_data_root "/path/to/data/celebahq_test_512" \
--arch "hdit_ImageNet256Sp4" \
--num_flow_steps 50 \
--num_gpus 16 \
--num_workers 80 \
--check_val_every_n_epoch 50 \
--train_batch_size 256 \
--val_batch_size 32 \
--img_size 512 \
--max_epochs 3850 \
--ema_decay 0.9999 \
--eps 0.0 \
--t_schedule "stratified_uniform" \
--weight_decay 1e-2 \
--lr 5e-4 \
--source_noise_std 0.1 \
--wandb_project_name "PMRF" \
--wandb_group "Blind face restoration PMRF" \
--mmse_model_arch "swinir_L" \
--mmse_model_ckpt_path "./checkpoints/swinir_restoration512_L1.pth"  # Path to the DifFace checkpoint of SwinIR
