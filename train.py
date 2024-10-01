import argparse

import torch
import torchvision.transforms as tvt
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

from torch_datasets.image_folder_dataset import ImageFolderDataset
from lightning_models.mmse_rectified_flow import MMSERectifiedFlow
from utils.create_degradation import create_degradation

torch.set_float32_matmul_precision('high')


def create_dataset(args):
    degradation = create_degradation(args.degradation)
    train_transform = tvt.Compose([
        tvt.Resize(args.img_size),
        tvt.ToTensor(),
    ])
    val_transform = tvt.Compose([
        tvt.Resize(args.img_size),
        tvt.ToTensor(),
    ])
    train_data = ImageFolderDataset(root=args.train_data_root,
                                    degradation=degradation,
                                    transform=train_transform,
                                    source_samples_root=args.source_samples_train_data_root)
    val_data = ImageFolderDataset(root=args.val_data_root,
                                  degradation=degradation,
                                  transform=val_transform)

    return train_data, val_data


def main(args):
    assert args.train_batch_size % args.num_gpus == 0
    logger = WandbLogger(project=args.wandb_project_name,
                         group=args.wandb_group,
                         id=args.wandb_id)
    logger.log_hyperparams(vars(args))
    train_data, val_data = create_dataset(args)

    train_dataloader = DataLoader(train_data,
                                  batch_size=args.train_batch_size // args.num_gpus,
                                  shuffle=True,
                                  drop_last=True,
                                  num_workers=args.num_workers // args.num_gpus,
                                  pin_memory=True,
                                  persistent_workers=True)
    val_dataloader = DataLoader(val_data,
                                batch_size=args.val_batch_size,
                                shuffle=False,
                                drop_last=False,
                                pin_memory=True,
                                num_workers=args.num_workers // args.num_gpus)
    ckpt_callback = ModelCheckpoint(save_last=True)
    lr_monitor_callback = LearningRateMonitor()
    trainer = Trainer(logger=logger,
                      max_epochs=args.max_epochs,
                      accelerator='gpu',
                      strategy='ddp',
                      devices=args.num_gpus,
                      callbacks=[ckpt_callback, lr_monitor_callback],
                      precision=args.precision,
                      check_val_every_n_epoch=args.check_val_every_n_epoch)
    with trainer.init_module():
        model = MMSERectifiedFlow(stage=args.stage,
                                  arch=args.arch,
                                  conditional=args.conditional,
                                  mmse_model_ckpt_path=args.mmse_model_ckpt_path,
                                  lr=args.lr,
                                  weight_decay=args.weight_decay,
                                  betas=args.betas,
                                  mmse_noise_std=args.source_noise_std,
                                  mmse_model_arch=args.mmse_model_arch,
                                  num_flow_steps=args.num_flow_steps,
                                  ema_decay=args.ema_decay,
                                  eps=args.eps,
                                  t_schedule=args.t_schedule,
                                  colorization='colorization' in args.degradation)
    torch.compile(model, mode='max-autotune')
    trainer.fit(model=model,
                train_dataloaders=train_dataloader,
                val_dataloaders=val_dataloader,
                ckpt_path=args.resume_from_ckpt)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--precision', type=str, required=False, choices=['bf16-mixed', '32'],
                        help='The precision used for training.')
    parser.add_argument('--stage', type=str, required=True, choices=['mmse', 'flow', 'naive_flow'],
                        help='The stage of the model.')
    parser.add_argument('--conditional', action='store_true',
                        help='If set, the flow model is conditioned on either y or the posterior mean predictor. '
                             'Applies only to the stage "flow".')
    parser.add_argument('--degradation', type=str, required=True,
                        choices=['sr_bicubic_x8_gaussian_noise_005',
                                 'gaussian_noise_035',
                                 'colorization_gaussian_noise_025',
                                 'random_inpainting_gaussian_noise_01',
                                 'difface'],
                        help='The degradation type.')
    parser.add_argument('--train_data_root', type=str, required=True,
                        help='Path to training data.')
    parser.add_argument('--source_samples_train_data_root', type=str, required=False, default=None,
                        help='Path to source samples corresponding to the high-quality images in the training data'
                             ' (useful for reflow).')
    parser.add_argument('--val_data_root', type=str, required=True,
                        help='Path to validation data.')
    parser.add_argument('--arch', type=str, required=True,
                        choices=['hdit_XL2',
                                 'hdit_ImageNet256Sp4',
                                 'swinir_M',
                                 'swinir_L'],
                        help='Architecture name and size.')
    parser.add_argument('--mmse_model_ckpt_path', type=str, required=False, default=None,
                        help='Checkpoint path to a pre-trained MMSE model.'
                             'Relevant only for the stage "flow". If --conditional is set, the outputs of this model'
                             ' will be the input condition of the flow. Otherwise, if --conditional is not set,'
                             'PMRF will be trained.')
    parser.add_argument('--mmse_model_arch', type=str, required=False, default=None,
                        help='The architecture of the pre-trained MMSE model. Only relevant for the stage "flow".')
    parser.add_argument('--source_noise_std', type=float, required=False, default=0.0,
                        help='Noise std to add to the samples from the source distribution (sigma_s in the paper).'
                             'Applies only to PMRF and naive flow.')
    parser.add_argument('--num_flow_steps', type=int, required=False, default=50,
                        help='Number of flow steps for evaluation.')
    parser.add_argument('--num_gpus', type=int, required=False, default=4,
                        help='Number of gpus to use.')
    parser.add_argument('--check_val_every_n_epoch', type=int, required=False, default=1,
                        help='Check validation every n epochs.')
    parser.add_argument('--train_batch_size', type=int, required=False, default=256,
                        help='Training batch size (on DDP, will be the total batch size on all GPUs).')
    parser.add_argument('--val_batch_size', type=int, required=False, default=32,
                        help='Validation batch size (on DDP, will be the batch size on each GPU).')
    parser.add_argument('--num_workers', type=int, required=False, default=10,
                        help='Number of workers on all GPUs.')
    parser.add_argument('--img_size', type=int, required=False, default=512,
                        help='Resize training and validation images to a specific size.')
    parser.add_argument('--max_epochs', type=int, required=False, default=1000,
                        help='Number of training epochs.')
    parser.add_argument('--ema_decay', type=float, required=False, default=0.9999,
                        help='Exponential moving average decay.')
    parser.add_argument('--eps', type=float, required=False, default=0,
                        help='Starting time of the flow.')
    parser.add_argument('--t_schedule', type=str, required=False, default='stratified_uniform',
                        choices=['uniform', 'logit-normal', 'stratified_uniform'],
                        help='Flow time scheduler (sampler) for training. We found stratified_uniform to work best.')
    parser.add_argument('--weight_decay', type=float, required=False, default=1e-2,
                        help='Optimizer weight decay.')
    parser.add_argument('--lr', type=float, required=False, default=5e-4,
                        help='Optimizer learning rate.')
    parser.add_argument('--betas', type=tuple, required=False, default=(0.9, 0.95),
                        help='Betas for the AdamW optimizer.')
    parser.add_argument('--wandb_project_name', type=str, required=True, default='Rectified Restoration Flow',
                        help='Project name for weights and biases logger.')
    parser.add_argument('--wandb_group', type=str, required=False, default=None,
                        help='Group of wandb experiment.')
    parser.add_argument('--wandb_id', type=str, required=False, default=None,
                        help='Specify an id if you resume training from a checkpoint.')
    parser.add_argument('--resume_from_ckpt', type=str, required=False, default=None,
                        help='Resume lightning training from this checkpoint.')

    main(parser.parse_args())
