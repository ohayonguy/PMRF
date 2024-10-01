import os

import piq
import torch
from torch.utils.data import DataLoader
from torch_fidelity import calculate_metrics
from tqdm import tqdm
from torch_fidelity.datasets import ImagesPathDataset
from torch_fidelity.registry import register_dataset

import argparse

torch.set_grad_enabled(False)

register_dataset('ffhq256',
                 lambda root, download: ImagesPathDataset([os.path.join(root, 'ffhq256', file_name) for file_name in
                                                           os.listdir(os.path.join(root, 'ffhq256'))]))


def compute_metrics_given_folder(xhat_dir, gt_dir, parent_ffhq_256_path):
    results = {}
    lpips_met = piq.LPIPS(reduction='mean').cuda()
    rec_files = sorted([os.path.join(xhat_dir, file) for file in os.listdir(xhat_dir)])
    xhat_ds = ImagesPathDataset(rec_files)
    gt_files = sorted([os.path.join(gt_dir, file) for file in os.listdir(gt_dir)])
    assert len(gt_files) == len(rec_files), f"{len(gt_files)}, {len(rec_files)}"
    for i in range(len(gt_files)):
        assert os.path.basename(gt_files[i]) == os.path.basename(
            rec_files[i]), f"{os.path.basename(gt_files[i])}, {os.path.basename(rec_files[i])}"
    gt_ds = ImagesPathDataset(gt_files)
    gt_dl = DataLoader(gt_ds, batch_size=128, shuffle=False, drop_last=False, num_workers=10)
    rec_dl = DataLoader(xhat_ds, batch_size=128, shuffle=False, drop_last=False, num_workers=10)

    mse = 0
    lpips = 0
    psnr = 0
    ssim = 0
    for gt, rec in tqdm(zip(gt_dl, rec_dl)):
        gt = gt.cuda().float()
        rec = rec.cuda().float()
        mse += ((gt - rec) ** 2).mean() * gt.shape[0]
        lpips += lpips_met(gt / 255., rec / 255.) * gt.shape[0]
        psnr += piq.psnr(gt / 255., rec / 255., data_range=1., reduction='sum')
        ssim += piq.ssim(gt / 255., rec / 255., data_range=1., reduction='sum')
    mse /= len(gt_ds)
    lpips /= len(gt_ds)
    ssim /= len(gt_ds)
    psnr /= len(gt_ds)
    results['psnr'] = psnr.item()
    results['mse'] = mse.item()
    results['lpips'] = lpips.item()
    results['ssim'] = ssim.item()

    fidelity_results = calculate_metrics(
        batch_size=512,
        input1='ffhq256',
        input2=xhat_ds,
        datasets_root=parent_ffhq_256_path,
        datasets_download=False,
        cuda=True,
        isc=True,
        fid=True,
        kid=True,
        prc=True,
        verbose=True,
        kid_subset_size=min(1000, len(xhat_ds)),
        cache=True
    )

    results = {**results, **fidelity_results}

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--parent_ffhq_256_path', type=str, required=True,
                        help='Path to parent folder of the ffhq 256 image data set (for computing FID, KID and Precision).'
                             'Make sure that the ffhq folder name is `ffhq256`, and provide here its parent directory.'
                             'For example, if ffhq256 sits in /path/to/parent/ffhq256, then you need to provide the path'
                             '/path/to/parent')
    parser.add_argument('--rec_path', type=str, required=True,
                        help='Path to a folder that contains reconstructions.')
    parser.add_argument('--gt_path', type=str, required=False,
                        help='Path to a folder that contains the ground-truth images.'
                             'The images must have the same file names as rec_path.'
                             'These are used to compute PSNR, SSIM, etc.')
    args = parser.parse_args()
    results = compute_metrics_given_folder(args.rec_path, args.gt_path, args.parent_ffhq_256_path)
    print(results)
