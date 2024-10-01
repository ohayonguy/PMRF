#!/bin/bash

python train.py \
--precision "bf16-mixed" \
--stage "flow" \
--degradation "random_inpainting_gaussian_noise_01" \
--train_data_root "/path/to/data/ffhq256" \
--val_data_root "/path/to/data/celebahq_test_256" \
--arch "hdit_XL2" \
--num_flow_steps 50 \
--num_gpus 4 \
--num_workers 40 \
--check_val_every_n_epoch 10 \
--train_batch_size 256 \
--val_batch_size 32 \
--img_size 256 \
--max_epochs 1000 \
--ema_decay 0.9999 \
--eps 0.0 \
--t_schedule "stratified_uniform" \
--weight_decay 1e-2 \
--lr 5e-4 \
--wandb_project_name "Posterior-Mean Rectified Flow" \
--wandb_group "PMRF" \
--mmse_model_arch "swinir_M" \
--mmse_model_ckpt_path "/path/to/checkpoint/last.ckpt" \
--source_noise_std 0.1
