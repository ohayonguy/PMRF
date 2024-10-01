# Codes adopted from VQFR: https://github.com/TencentARC/VQFR

import glob
import math
import os

import cv2
import numpy as np
import torch
from torch.nn import DataParallel
from torch.nn import functional as F
from torchvision.transforms.functional import normalize

from arcface.config.config import Config
from arcface.models.resnet import resnet_face18


def img2tensor(imgs, bgr2rgb=True, float32=True):
    """Numpy array to tensor.

    Args:
        imgs (list[ndarray] | ndarray): Input images.
        bgr2rgb (bool): Whether to change bgr to rgb.
        float32 (bool): Whether to change to float32.

    Returns:
        list[tensor] | tensor: Tensor images. If returned results only have
            one element, just return tensor.
    """

    def _totensor(img, bgr2rgb, float32):
        if img.shape[2] == 3 and bgr2rgb:
            if img.dtype == 'float64':
                img = img.astype('float32')
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img.transpose(2, 0, 1))
        if float32:
            img = img.float()
        return img

    if isinstance(imgs, list):
        return [_totensor(img, bgr2rgb, float32) for img in imgs]
    else:
        return _totensor(imgs, bgr2rgb, float32)


def load_image(img_path):
    image = cv2.imread(img_path, 0)  # only on gray images
    # resise
    image = cv2.resize(image, (128, 128), interpolation=cv2.INTER_LINEAR)
    if image is None:
        return None
    # image = np.dstack((image, np.fliplr(image)))
    # image = image.transpose((2, 0, 1))
    image = image[np.newaxis, :, :]
    image = image.astype(np.float32, copy=False)
    image -= 127.5
    image /= 127.5
    image = torch.from_numpy(image)
    return image


def load_image_torch(img_path):
    image = cv2.imread(img_path) / 255.
    image = image.astype(np.float32)
    image = img2tensor(image, bgr2rgb=True, float32=True)
    normalize(image, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5], inplace=True)
    image.unsqueeze_(0)
    image = (0.2989 * image[:, 0, :, :] + 0.5870 * image[:, 1, :, :] + 0.1140 * image[:, 2, :, :])
    image = image.unsqueeze(1)
    image = F.interpolate(image, (128, 128), mode='bilinear', align_corners=False)
    return image


def cosin_metric(x1, x2):
    return np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))


def calculate_cos_dist(gt_folder, restored_folder):
    test_model_path = './metrics_ckpt/resnet18_110.pth'

    restored_list = sorted(glob.glob(os.path.join(restored_folder, '*')))
    gt_list = sorted(glob.glob(os.path.join(gt_folder, '*')))

    opt = Config()
    if opt.backbone == 'resnet18':
        model = resnet_face18(opt.use_se)
    else:
        raise NotImplementedError
    # elif opt.backbone == 'resnet34':
    #    model = resnet34()
    # elif opt.backbone == 'resnet50':
    #    model = resnet50()

    model = DataParallel(model)
    model.load_state_dict(torch.load(test_model_path))
    model.to(torch.device('cuda'))
    model.eval()
    dist_list = []
    identical_count = 0
    for idx, (restored_path, gt_path) in enumerate(zip(restored_list, gt_list)):
        basename, ext = os.path.splitext(os.path.basename(gt_path))
        img = load_image(gt_path)
        img2 = load_image(restored_path)
        # img = load_image_torch(img_path)
        # img2 = load_image_torch(img_path2)
        data = torch.stack([img, img2], dim=0)
        data = data.to(torch.device('cuda'))
        output = model(data)
        output = output.data.cpu().numpy()
        dist = cosin_metric(output[0], output[1])
        dist = np.arccos(dist) / math.pi * 180
        print(f'{idx} - {dist} o : {basename}')
        if dist < 1:
            print(f'{basename} is almost identical to original.')
            identical_count += 1
        else:
            dist_list.append(dist)
    return sum(dist_list) / len(dist_list), identical_count
    # print(f'Result dist: {sum(dist_list) / len(dist_list):.6f}')
    # print(f'identical count: {identical_count}')

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('-restored_folder', type=str, help='Path to the folder.', required=True)
#     parser.add_argument('-gt_folder', type=str, help='Path to the folder.', required=True)
#     args = parser.parse_args()
#     calculate_cos_dist(args)
