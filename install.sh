#!/bin/bash

conda create -n pmrf python=3.10 -y
conda activate pmrf
conda install pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda=11.8 -c pytorch -c nvidia -y
conda install lightning==2.3.3 -c conda-forge -y

pip install opencv-python==4.10.0.84 timm==1.0.8 wandb==0.17.5 lovely-tensors==0.1.16 torch-fidelity==0.3.0 einops==0.8.0 dctorch==0.1.2 torch-ema==0.3 --no-input
pip install natten==0.17.1+torch230cu118 -f https://shi-labs.com/natten/wheels --no-input
pip install nvidia-cuda-nvcc-cu11 --no-input
pip install basicsr==1.4.2  --no-input
pip install git+https://github.com/toshas/torch-fidelity.git --no-input
pip install lpips==0.1.4 --no-input
pip install piq==0.8.0 --no-input