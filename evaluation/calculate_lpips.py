import cv2
import glob
import numpy as np
import os.path as osp
from torchvision.transforms.functional import normalize
from basicsr.utils import img2tensor
from tqdm import tqdm
import argparse
import torch

torch.set_grad_enabled(False)

import lpips



def calculate_lpips(gt_folder, restored_folder):
    # Configurations
    loss_fn_vgg = lpips.LPIPS(net='vgg').cuda()  # RGB, normalized to [-1,1]
    lpips_all = []

    img_list = sorted(glob.glob(osp.join(gt_folder, '*')))
    restored_list = sorted(glob.glob(osp.join(restored_folder, '*')))

    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
    for i, (restored_path, img_path) in enumerate(tqdm(zip(restored_list, img_list))):
        img_gt = cv2.imread(img_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.
        img_restored = cv2.imread(restored_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.

        img_gt, img_restored = img2tensor([img_gt, img_restored], bgr2rgb=True, float32=True)
        # norm to [-1, 1]
        normalize(img_gt, mean, std, inplace=True)
        normalize(img_restored, mean, std, inplace=True)

        # calculate lpips
        lpips_val = loss_fn_vgg(img_restored.unsqueeze(0).cuda(), img_gt.unsqueeze(0).cuda())
        lpips_val = lpips_val.cpu().item()
        # print(f'{i+1:3d}: {basename:25}. \tLPIPS: {lpips_val:.6f}.')
        lpips_all.append(lpips_val)
        if i % 100 == 0:
            print(sum(lpips_all) / len(lpips_all))
    return sum(lpips_all) / len(lpips_all)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt', type=str, default='', help='Path to gt (Ground-Truth)')
    parser.add_argument('--restored', type=str, help='Path to restored images')
    args = parser.parse_args()
    calculate_lpips(args.gt, args.restored)
