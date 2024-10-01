import os

import torch
import argparse
from basicsr.metrics.psnr_ssim import calculate_psnr_pt, calculate_ssim_pt
from torch.utils.data import DataLoader
from torch_fidelity import calculate_metrics
from calculate_lpips import calculate_lpips
from torch_fidelity.datasets import ImagesPathDataset
from torch_fidelity.registry import register_dataset
from calculate_landmark_distance import calculate_landmark_distance
from calculate_niqe import calculate_niqe
from calculate_cos_dist import calculate_cos_dist
from tqdm import tqdm

torch.set_grad_enabled(False)

register_dataset('ffhq512',
                 lambda root, download: ImagesPathDataset([os.path.join(root, 'ffhq512', file_name) for file_name in
                                                           os.listdir(os.path.join(root, 'ffhq512'))]))


def compute_metrics_given_folder(xhat_dir, gt_dir, mmse_dir, parent_ffhq_512_path):
    results = {}
    rec_files = sorted([os.path.join(xhat_dir, file) for file in os.listdir(xhat_dir)])
    xhat_ds = ImagesPathDataset(rec_files)
    results['niqe'] = calculate_niqe(xhat_dir)
    if gt_dir is not None:
        gt_files = sorted([os.path.join(gt_dir, file) for file in os.listdir(gt_dir)])
        assert len(gt_files) == len(rec_files)
        for i in range(len(gt_files)):
            assert os.path.basename(gt_files[i]) == os.path.basename(
                rec_files[i]), f"{os.path.basename(gt_files[i])}, {os.path.basename(rec_files[i])}"
        gt_ds = ImagesPathDataset(gt_files)
        gt_dl = DataLoader(gt_ds, batch_size=64, shuffle=False, drop_last=False, num_workers=10)
        rec_dl = DataLoader(xhat_ds, batch_size=64, shuffle=False, drop_last=False, num_workers=10)
        results['landmark_distance'] = calculate_landmark_distance(gt_dir, xhat_dir)
        results['arcface_cos_dist'], results['identity_count'] = calculate_cos_dist(gt_dir, xhat_dir)
        results['lpips'] = calculate_lpips(gt_dir, xhat_dir)
        mse = 0
        psnr = 0
        ssim = 0
        for gt, rec in tqdm(zip(gt_dl, rec_dl)):
            gt = gt.cuda().float()
            rec = rec.cuda().float()
            mse += ((gt - rec) ** 2).mean() * gt.shape[0]
            psnr += calculate_psnr_pt(gt / 255., rec / 255., crop_border=0).sum()
            ssim += calculate_ssim_pt(gt / 255., rec / 255., crop_border=0).sum()
        mse /= len(gt_ds)
        ssim /= len(gt_ds)
        psnr /= len(gt_ds)
        results['mse'] = mse.item()
        results['ssim'] = ssim.item()
        results['psnr'] = psnr.item()
    if mmse_dir is not None:
        mmse_files = sorted([os.path.join(mmse_dir, file) for file in os.listdir(mmse_dir)])
        assert len(mmse_files) == len(rec_files)
        for i in range(len(mmse_files)):
            assert os.path.basename(mmse_files[i]) == os.path.basename(
                rec_files[i]), f"{os.path.basename(mmse_files[i])}, {os.path.basename(rec_files[i])}"
        mmse_ds = ImagesPathDataset(mmse_files)
        mmse_dl = DataLoader(mmse_ds, batch_size=64, shuffle=False, drop_last=False, num_workers=10)
        rec_dl = DataLoader(xhat_ds, batch_size=64, shuffle=False, drop_last=False, num_workers=10)
        ind_psnr = 0
        ind_mse = 0
        for mmse, rec in tqdm(zip(mmse_dl, rec_dl)):
            mmse = mmse.cuda().float()
            rec = rec.cuda().float()
            ind_mse += ((mmse - rec) ** 2).mean() * mmse.shape[0]
            ind_psnr += calculate_psnr_pt(mmse / 255., rec / 255., crop_border=0).sum()
        ind_psnr /= len(mmse_ds)
        ind_mse /= len(mmse_ds)
        results['ind_psnr'] = ind_psnr.item()
        results['ind_mse'] = ind_mse.item()

    if 'frechet_inception_distance' not in results:
        fidelity_results = calculate_metrics(
            batch_size=256,
            input1='ffhq512',
            input2=xhat_ds,
            datasets_root=parent_ffhq_512_path,
            datasets_download=False,
            cuda=True,
            isc=True,
            fid=True,
            kid=True,
            kid_subset_size=min(1000, len(xhat_ds)),
            prc=True,
            verbose=True,
            cache=True
        )
        results = {**results, **fidelity_results}

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--parent_ffhq_512_path', type=str, required=True,
                        help='Path to parent folder of the ffhq 512 image data set (for computing FID, KID and Precision).'
                             'Make sure that the ffhq folder name is `ffhq512`, and provide here its parent directory.'
                             'For example, if ffhq512 sits in /path/to/parent/ffhq512, then you need to provide the path'
                             '/path/to/parent')
    parser.add_argument('--rec_path', type=str, required=True,
                        help='Path to a folder that contains reconstructions.')
    parser.add_argument('--gt_path', type=str, required=False,
                        help='Path to a folder that contains the ground-truth images.'
                             'The images must have the same file names as rec_path.'
                             'These are used to compute PSNR, SSIM, etc.')
    parser.add_argument('--mmse_rec_path', type=str, required=False,
                        help='Path to a folder where there is posterior mean predictions (MMSE estimator outputs).'
                             'The images must have the same file names as rec_path.'
                             'These are used to compute IndRMSE.')
    args = parser.parse_args()
    results = compute_metrics_given_folder(args.rec_path, args.gt_path, args.mmse_rec_path, args.parent_ffhq_512_path)
    print(results)
