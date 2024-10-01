#!/bin/bash

python compute_metrics_controlled_experiments.py \
--parent_ffhq_256_path /path/to/parent/of/ffhq256 \
--rec_path /path/to/restored/images \
--gt_path /path/to/celeba-256-test/ground-truth/images
