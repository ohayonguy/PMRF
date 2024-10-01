#!/bin/bash

python inference.py \
--lq_data_path /path/to/lq/images \
--output_dir /path/to/results/dir \
--batch_size 64 \
--num_flow_steps 25
