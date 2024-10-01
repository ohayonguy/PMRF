#!/bin/bash

# CelebA-Test or any other data set with ground-truth images
python compute_metrics_blind.py \
--parent_ffhq_512_path /path/to/parent/of/ffhq512 \
--rec_path /path/to/celeba-512-test/restored/images \
--gt_path /path/to/celeba-512-test/ground-truth/images


# Real-world degraded images (no ground-truth)
python compute_metrics_blind.py \
--parent_ffhq_512_path /path/to/parent/of/ffhq512 \
--rec_path /path/to/real-world/restored/images \
--mmse_rec_path /path/to/mmse/restored/images