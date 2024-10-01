import math
from functools import partial

import cv2
import numpy as np
import torch
from basicsr.data import degradations as degradations
from basicsr.data.transforms import augment
from basicsr.utils import img2tensor
from torch.nn.functional import interpolate
from torchvision.transforms import Compose
from utils.basicsr_custom import (
    random_mixed_kernels,
    random_add_gaussian_noise,
    random_add_jpg_compression,
)


def create_degradation(degradation):
    if degradation == 'sr_bicubic_x8_gaussian_noise_005':
        return Compose([
            partial(down_scale, scale_factor=1.0 / 8.0, mode='bicubic'),
            partial(add_gaussian_noise, std=0.05),
            partial(interpolate, scale_factor=8.0, mode='nearest-exact'),
            partial(torch.clip, min=0, max=1),
            partial(torch.squeeze, dim=0),
            lambda x: (x, None)

        ])
    elif degradation == 'gaussian_noise_035':
        return Compose([
            partial(add_gaussian_noise, std=0.35),
            partial(torch.clip, min=0, max=1),
            partial(torch.squeeze, dim=0),
            lambda x: (x, None)

        ])
    elif degradation == 'colorization_gaussian_noise_025':
        return Compose([
            lambda x: torch.mean(x, dim=0, keepdim=True),
            partial(add_gaussian_noise, std=0.25),
            partial(torch.clip, min=0, max=1),
            lambda x: (x, None)
        ])
    elif degradation == 'random_inpainting_gaussian_noise_01':
        def inpainting_dps(x):
            total = x.shape[1] ** 2
            # random pixel sampling
            l, h = [0.9, 0.9]
            prob = np.random.uniform(l, h)
            mask_vec = torch.ones([1, x.shape[1] * x.shape[1]])
            samples = np.random.choice(x.shape[1] * x.shape[1], int(total * prob), replace=False)
            mask_vec[:, samples] = 0
            mask_b = mask_vec.view(1, x.shape[1], x.shape[1])
            mask_b = mask_b.repeat(3, 1, 1)
            mask = torch.ones_like(x, device=x.device)
            mask[:, ...] = mask_b
            return add_gaussian_noise(x * mask, 0.1).clip(0, 1), None

        return inpainting_dps
    elif degradation == 'difface':
        def deg(x):
            blur_kernel_size = 41
            kernel_list = ['iso', 'aniso']
            kernel_prob = [0.5, 0.5]
            blur_sigma = [0.1, 15]
            downsample_range = [0.8, 32]
            noise_range = [0, 20]
            jpeg_range = [30, 100]
            gt_gray = True
            gray_prob = 0.01
            x = x.permute(1, 2, 0).numpy()[..., ::-1].astype(np.float32)
            # random horizontal flip
            img_gt = augment(x.copy(), hflip=True, rotation=False)
            h, w, _ = img_gt.shape

            # ------------------------ generate lq image ------------------------ #
            # blur
            kernel = degradations.random_mixed_kernels(
                kernel_list,
                kernel_prob,
                blur_kernel_size,
                blur_sigma,
                blur_sigma, [-math.pi, math.pi],
                noise_range=None)
            img_lq = cv2.filter2D(img_gt, -1, kernel)
            # downsample
            scale = np.random.uniform(downsample_range[0], downsample_range[1])
            img_lq = cv2.resize(img_lq, (int(w // scale), int(h // scale)), interpolation=cv2.INTER_LINEAR)
            # noise
            if noise_range is not None:
                img_lq = random_add_gaussian_noise(img_lq, noise_range)
            # jpeg compression
            if jpeg_range is not None:
                img_lq = random_add_jpg_compression(img_lq, jpeg_range)

            # resize to original size
            img_lq = cv2.resize(img_lq, (w, h), interpolation=cv2.INTER_LINEAR)

            # random color jitter (only for lq)
            # if self.color_jitter_prob is not None and (np.random.uniform() < self.color_jitter_prob):
            #     img_lq = self.color_jitter(img_lq, self.color_jitter_shift)
            # random to gray (only for lq)
            if np.random.uniform() < gray_prob:
                img_lq = cv2.cvtColor(img_lq, cv2.COLOR_BGR2GRAY)
                img_lq = np.tile(img_lq[:, :, None], [1, 1, 3])
                if gt_gray:  # whether convert GT to gray images
                    img_gt = cv2.cvtColor(img_gt, cv2.COLOR_BGR2GRAY)
                    img_gt = np.tile(img_gt[:, :, None], [1, 1, 3])  # repeat the color channels

            # BGR to RGB, HWC to CHW, numpy to tensor
            img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=True, float32=True)

            # random color jitter (pytorch version) (only for lq)
            # if self.color_jitter_pt_prob is not None and (np.random.uniform() < self.color_jitter_pt_prob):
            #     brightness = self.opt.get('brightness', (0.5, 1.5))
            #     contrast = self.opt.get('contrast', (0.5, 1.5))
            #     saturation = self.opt.get('saturation', (0, 1.5))
            #     hue = self.opt.get('hue', (-0.1, 0.1))
            #     img_lq = self.color_jitter_pt(img_lq, brightness, contrast, saturation, hue)

            # round and clip
            img_lq = torch.clamp((img_lq * 255.0).round(), 0, 255) / 255.

            return img_lq, img_gt.clip(0, 1)

        return deg
    else:
        raise NotImplementedError()


def down_scale(x, scale_factor, mode):
    with torch.no_grad():
        return interpolate(x.unsqueeze(0),
                           scale_factor=scale_factor,
                           mode=mode,
                           antialias=True,
                           align_corners=False).clip(0, 1)


def add_gaussian_noise(x, std):
    with torch.no_grad():
        x = x + torch.randn_like(x) * std
        return x
