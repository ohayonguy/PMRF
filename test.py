import argparse
import os

import torch
import torchvision.transforms as tvt
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader

from torch_datasets.image_folder_dataset import ImageFolderDataset
from lightning_models.mmse_rectified_flow import MMSERectifiedFlow
from utils.create_degradation import create_degradation

torch.set_float32_matmul_precision('high')


method_hyperparams_to_name = {
    'stage=flow;conditional=True;mmse_model_used=True': "posterior_conditioned_on_mmse",
    'stage=flow;conditional=True;mmse_model_used=False': "posterior_conditioned_on_y",
    'stage=flow;conditional=False;mmse_model_used=True': "pmrf",
    'stage=mmse;conditional=False;mmse_model_used=False': "mmse",
    'stage=naive_flow;conditional=False;mmse_model_used=False': "naive_flow",
}
def main(args):
    assert args.batch_size % args.num_gpus == 0
    test_transform = tvt.Compose([
        tvt.Resize(args.img_size),
        tvt.ToTensor(),
    ])
    degradation = create_degradation(args.degradation)
    test_data = ImageFolderDataset(root=args.test_data_root,
                                   degradation=degradation,
                                   transform=test_transform)

    test_dataloader = DataLoader(test_data,
                                 batch_size=args.batch_size,
                                 shuffle=False,
                                 drop_last=False,
                                 pin_memory=True,
                                 num_workers=args.num_workers)

    trainer = Trainer(accelerator='gpu',
                      strategy='ddp',
                      devices=args.num_gpus,
                      precision=args.precision)

    ckpt = torch.load(args.ckpt_path, map_location="cpu")
    mmse_model_arch = ckpt['hyper_parameters']['mmse_model_arch'] if 'mmse_model_arch' in ckpt['hyper_parameters'] else None
    model = MMSERectifiedFlow.load_from_checkpoint(args.ckpt_path,
                                                   # Need to provide mmse_model_arch to
                                                   # make sure the model initializes it.
                                                   mmse_model_arch=mmse_model_arch,
                                                   mmse_model_ckpt_path=None,  # Will ignore the original path of the
                                                   # MMSE model used for training,
                                                   # and instead load it from the model checkpoint.
                                                   map_location='cpu').cuda()
    method_hyperparams = (f"stage={model.hparams.stage};"
                          f"conditional={model.hparams.conditional};"
                          f"mmse_model_used={mmse_model_arch is not None}")
    model.test_results_path = os.path.join(args.results_path,
                                           args.degradation,
                                           method_hyperparams_to_name[method_hyperparams])
    model.num_test_flow_steps = tuple(args.num_flow_steps)
    os.makedirs(model.test_results_path, exist_ok=True)
    torch.compile(model, mode='max-autotune')
    model.freeze()
    trainer.test(model=model, dataloaders=test_dataloader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--precision', type=str, required=False, default='32', choices=['bf16-mixed', '32'],
                        help='The precision used for testing.')
    parser.add_argument('--degradation', type=str, required=True,
                        choices=['sr_bicubic_x8_gaussian_noise_005',
                                 'gaussian_noise_035',
                                 'colorization_gaussian_noise_025',
                                 'random_inpainting_gaussian_noise_01',
                                 'difface'],
                        help='The degradation type.')
    parser.add_argument('--test_data_root', type=str, required=True,
                        help='Path to test data. Should be high-quality images, which will be degraded according to'
                             '--degradation.')
    parser.add_argument('--num_gpus', type=int, required=False, default=4,
                        help='Number of gpus to use.')
    parser.add_argument('--batch_size', type=int, required=False, default=32,
                        help='Batch size to use for testing.')
    parser.add_argument('--num_workers', type=int, required=False, default=10,
                        help='Number of workers on all GPUs.')
    parser.add_argument('--ckpt_path', type=str, required=True,
                        help='The checkpoint path of the model to test.')
    parser.add_argument('--results_path', type=str, required=True,
                        help='Folder path where the reconstructed images will be saved.')
    parser.add_argument('--img_size', type=int, required=False, default=256,
                        help='Resize the images to a specific size.')
    parser.add_argument('--num_flow_steps', type=int, nargs = '+', required=True,
                        help='Number of flow steps to test. You may provide a list of values.')

    main(parser.parse_args())
