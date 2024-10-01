#!/bin/bash


for method in "mmse" "naive_flow" "pmrf" "posterior_conditioned_on_mmse" "posterior_conditioned_on_y"; do
python test.py \
--precision "32" \
--degradation "colorization_gaussian_noise_025" \
--test_data_root "/path/to/celebahq_test_256" \
--num_gpus 1 \
--batch_size 128 \
--num_workers 10 \
--img_size 256 \
--ckpt_path "./checkpoints/controlled_experiments/colorization_gaussian_noise_025/${method}/epoch=999-step=273000.ckpt" \
--results_path "./controlled_experiments_results/" \
--num_flow_steps 100
#--num_flow_steps 5 10 20 50 100

done
