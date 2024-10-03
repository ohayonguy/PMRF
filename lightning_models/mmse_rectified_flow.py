import os
from contextlib import contextmanager, nullcontext

import torch
import wandb
from pytorch_lightning import LightningModule
from torch.nn.functional import mse_loss
from torch.nn.functional import sigmoid
from torch.optim import AdamW
from torch_ema import ExponentialMovingAverage as EMA
from torchmetrics.image import FrechetInceptionDistance, InceptionScore
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import save_image

from utils.create_arch import create_arch
from utils.img_utils import create_grid
from huggingface_hub import PyTorchModelHubMixin



class MMSERectifiedFlow(LightningModule,
                        PyTorchModelHubMixin,
                        pipeline_tag="image-to-image",
                        license="mit",
                        ):
    def __init__(self,
                 stage,
                 arch,
                 conditional=False,
                 mmse_model_ckpt_path=None,
                 mmse_model_arch=None,
                 lr=5e-4,
                 weight_decay=1e-3,
                 betas=(0.9, 0.95),
                 mmse_noise_std=0.1,
                 num_flow_steps=50,
                 ema_decay=0.9999,
                 eps=0.0,
                 t_schedule='stratified_uniform',
                 *args,
                 **kwargs
                 ):
        super().__init__()
        self.save_hyperparameters(logger=False)

        if stage == 'flow':
            if conditional:
                condition_channels = 3
            else:
                condition_channels = 0
            if mmse_model_arch is None and 'colorization' in kwargs and kwargs['colorization']:
                condition_channels //= 3
            self.model = create_arch(arch, condition_channels)
            self.mmse_model = create_arch(mmse_model_arch, 0) if mmse_model_arch is not None else None
            if mmse_model_ckpt_path is not None:
                ckpt = torch.load(mmse_model_ckpt_path, map_location="cpu")
                if mmse_model_arch is None:
                    mmse_model_arch = ckpt['hyper_parameters']['arch']
                self.mmse_model = create_arch(mmse_model_arch, 0)
                if 'ema' in ckpt:
                    # ema_decay doesn't affect anything here, because we are doing load_state_dict
                    mmse_ema = EMA(self.mmse_model.parameters(), decay=ema_decay)
                    mmse_ema.load_state_dict(ckpt['ema'])
                    mmse_ema.copy_to()
                elif 'params_ema' in ckpt:
                    self.mmse_model.load_state_dict(ckpt['params_ema'])
                else:
                    state_dict = ckpt['state_dict']
                    state_dict = {layer_name.replace('model.', ''): weights for layer_name, weights in
                                  state_dict.items()}
                    state_dict = {layer_name.replace('module.', ''): weights for layer_name, weights in
                                  state_dict.items()}
                    self.mmse_model.load_state_dict(state_dict)
                for param in self.mmse_model.parameters():
                    param.requires_grad = False
                self.mmse_model.eval()
        else:
            assert stage == 'mmse' or stage == 'naive_flow'
            assert not conditional
            self.model = create_arch(arch, 0)
            self.mmse_model = None
        if 'flow' in stage:
            self.fid = FrechetInceptionDistance(reset_real_features=True, normalize=True)
            self.inception_score = InceptionScore(normalize=True)

        self.ema = EMA(self.model.parameters(), decay=ema_decay) if self.ema_wanted else None
        self.test_results_path = None

    @property
    def ema_wanted(self):
        return self.hparams.ema_decay != -1

    def on_save_checkpoint(self, checkpoint: dict) -> None:
        if self.ema_wanted:
            checkpoint['ema'] = self.ema.state_dict()
        return super().on_save_checkpoint(checkpoint)

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        if self.ema_wanted:
            self.ema.load_state_dict(checkpoint['ema'])
        return super().on_load_checkpoint(checkpoint)

    def on_before_zero_grad(self, optimizer) -> None:
        if self.ema_wanted:
            self.ema.update(self.model.parameters())
        return super().on_before_zero_grad(optimizer)

    def to(self, *args, **kwargs):
        if self.ema_wanted:
            self.ema.to(*args, **kwargs)
        return super().to(*args, **kwargs)

    # This will use the contextmanager of ema, to copy the EMA weights to the flow model during validation, and then restore them for training.
    @contextmanager
    def maybe_ema(self):
        ema = self.ema
        ctx = nullcontext if ema is None else ema.average_parameters
        yield ctx

    def forward_mmse(self, y):
        return self.model(y).clip(0, 1)

    def forward_flow(self, x_t, t, y=None):
        if self.hparams.conditional:
            if self.mmse_model is not None:
                with torch.no_grad():
                    self.mmse_model.eval()
                    condition = self.mmse_model(y).clip(0, 1)
            else:
                condition = y
            x_t = torch.cat((x_t, condition), dim=1)
        return self.model(x_t, t)

    def forward(self, x_t, t, y):
        if 'flow' in self.hparams.stage:
            return self.forward_flow(x_t, t, y)
        else:
            return self.forward_mmse(y)

    @torch.no_grad()
    def create_source_distribution_samples(self, x, y, non_noisy_z0):
        with torch.no_grad():
            if self.hparams.conditional:
                source_dist_samples = torch.randn_like(x)
            else:
                if self.hparams.stage == 'flow':
                    if non_noisy_z0 is None:
                        self.mmse_model.eval()
                        non_noisy_z0 = self.mmse_model(y).clip(0, 1)
                    source_dist_samples = non_noisy_z0 + torch.randn_like(non_noisy_z0) * self.hparams.mmse_noise_std
                else:
                    assert self.hparams.stage == 'naive_flow'
                    if non_noisy_z0 is not None:
                        source_dist_samples = non_noisy_z0
                    else:
                        source_dist_samples = y
                    if source_dist_samples.shape[1] != x.shape[1]:
                        assert source_dist_samples.shape[1] == 1  # Colorization
                        source_dist_samples = source_dist_samples.expand(-1, x.shape[1], -1, -1)
                    if self.hparams.mmse_noise_std is not None:
                        source_dist_samples = source_dist_samples + torch.randn_like(source_dist_samples) * self.hparams.mmse_noise_std
        return source_dist_samples

    @staticmethod
    def stratified_uniform(bs, group=0, groups=1, dtype=None, device=None):
        if groups <= 0:
            raise ValueError(f"groups must be positive, got {groups}")
        if group < 0 or group >= groups:
            raise ValueError(f"group must be in [0, {groups})")
        n = bs * groups
        offsets = torch.arange(group, n, groups, dtype=dtype, device=device)
        u = torch.rand(bs, dtype=dtype, device=device)
        return ((offsets + u) / n).view(bs, 1, 1, 1)

    def generate_random_t(self, bs, dtype=None):
        if self.hparams.t_schedule == 'logit-normal':
            return sigmoid(torch.randn(bs, 1, 1, 1, device=self.device)) * (1.0 - self.hparams.eps) + self.hparams.eps
        elif self.hparams.t_schedule == 'uniform':
            return torch.rand(bs, 1, 1, 1, device=self.device) * (1.0 - self.hparams.eps) + self.hparams.eps
        elif self.hparams.t_schedule == 'stratified_uniform':
            return self.stratified_uniform(bs, self.trainer.global_rank, self.trainer.world_size, dtype=dtype,
                                           device=self.device) * (1.0 - self.hparams.eps) + self.hparams.eps
        else:
            raise NotImplementedError()

    def training_step(self, batch, batch_idx):
        x = batch['x']
        y = batch['y']
        non_noisy_z0 = batch['non_noisy_z0'] if 'non_noisy_z0' in batch else None
        if 'flow' in self.hparams.stage:
            with torch.no_grad():
                t = self.generate_random_t(x.shape[0], dtype=x.dtype)
                source_dist_samples = self.create_source_distribution_samples(x, y, non_noisy_z0)
                x_t = t * x + (1.0 - t) * source_dist_samples
            v_t = self(x_t, t.squeeze(), y)
            loss = mse_loss(v_t, x - source_dist_samples)
        else:
            xhat = self(x_t=None, t=None, y=y)
            loss = mse_loss(xhat, x)
        self.log("train/loss", loss)
        return loss

    @torch.no_grad()
    def generate_reconstructions(self, x, y, non_noisy_z0, num_flow_steps, result_device):
        with self.maybe_ema():
            if 'flow' in self.hparams.stage:
                source_dist_samples = self.create_source_distribution_samples(x, y, non_noisy_z0)

                dt = (1.0 / num_flow_steps) * (1.0 - self.hparams.eps)
                x_t_next = source_dist_samples.clone()
                x_t_seq = [x_t_next]
                t_one = torch.ones(x.shape[0], device=self.device)
                for i in range(num_flow_steps):
                    num_t = (i / num_flow_steps) * (1.0 - self.hparams.eps) + self.hparams.eps
                    v_t_next = self(x_t=x_t_next, t=t_one * num_t, y=y).to(x_t_next.dtype)
                    x_t_next = x_t_next.clone() + v_t_next * dt
                    x_t_seq.append(x_t_next.to(result_device))

                xhat = x_t_seq[-1].clip(0, 1).to(torch.float32)
                source_dist_samples = source_dist_samples.to(result_device)
            else:
                xhat = self(x_t=None, t=None, y=y).to(torch.float32)
                x_t_seq = None
                source_dist_samples = None
            return xhat.to(result_device), x_t_seq, source_dist_samples

    def validation_step(self, batch, batch_idx):
        x = batch['x']
        y = batch['y']
        non_noisy_z0 = batch['non_noisy_z0'] if 'non_noisy_z0' in batch else None
        xhat, x_t_seq, source_dist_samples = self.generate_reconstructions(x, y, non_noisy_z0, self.hparams.num_flow_steps,
                                                                           self.device)
        x = x.to(torch.float32)
        y = y.to(torch.float32)
        self.log_dict({"val_metrics/mse": ((x - xhat) ** 2).mean()}, on_step=False, on_epoch=True, sync_dist=True,
                      batch_size=x.shape[0])

        if 'flow' in self.hparams.stage:
            self.fid.update(x, real=True)
            self.fid.update(xhat, real=False)
            self.inception_score.update(xhat)

        if batch_idx == 0:
            wandb_logger = self.logger.experiment
            wandb_logger.log({'val_images/x': [wandb.Image(to_pil_image(create_grid(x)))],
                              'val_images/y': [wandb.Image(to_pil_image(create_grid(y.clip(0, 1))))],
                              'val_images/xhat': [wandb.Image(to_pil_image(create_grid(xhat)))], })
            if 'flow' in self.hparams.stage:
                wandb_logger.log({'val_images/x_t_seq': [wandb.Image(to_pil_image(create_grid(
                    torch.cat([elem[0].unsqueeze(0).to(torch.float32) for elem in x_t_seq], dim=0).clip(0, 1),
                    num_images=len(x_t_seq))))], 'val_images/source_distribution_samples': [
                    wandb.Image(to_pil_image(create_grid(source_dist_samples.clip(0, 1).to(torch.float32))))]})
                if self.mmse_model is not None:
                    xhat_mmse = self.mmse_model(y).clip(0, 1)
                    wandb_logger.log({'val_images/xhat_mmse': [
                        wandb.Image(to_pil_image(create_grid(xhat_mmse.to(torch.float32))))]})

    def on_validation_epoch_end(self):
        if 'flow' in self.hparams.stage:
            inception_score_mean, inception_score_std = self.inception_score.compute()
            self.log_dict(
                {'val_metrics/fid': self.fid.compute(),
                 'val_metrics/inception_score_mean': inception_score_mean,
                 'val_metrics/inception_score_std': inception_score_std},
                on_epoch=True, on_step=False, sync_dist=True,
                batch_size=1)
            self.fid.reset()
            self.inception_score.reset()

    def test_step(self, batch, batch_idx):
        assert self.test_results_path is not None, "Please set test_results_path before testing."
        assert os.path.isdir(self.test_results_path), 'Please make sure the test_result_path dir exists.'

        def save_image_batch(images, folder, image_file_names):
            os.makedirs(folder, exist_ok=True)
            for i, img in enumerate(images):
                save_image(images[i].clip(0, 1), os.path.join(folder, image_file_names[i]))

        os.makedirs(self.test_results_path, exist_ok=True)
        x = batch['x']
        y = batch['y']
        non_noisy_z0 = batch['non_noisy_z0'] if 'non_noisy_z0' in batch else None
        y_path = os.path.join(self.test_results_path, 'y')
        save_image_batch(y, y_path, batch['img_file_name'])

        if 'flow' in self.hparams.stage:
            source_dist_samples_to_save = None

            for num_flow_steps in self.num_test_flow_steps:
                xhat, x_t_seq, source_dist_samples = self.generate_reconstructions(x, y, non_noisy_z0, num_flow_steps,
                                                                                   torch.device("cpu"))
                xhat_path = os.path.join(self.test_results_path, f"num_flow_steps={num_flow_steps}", 'xhat')
                save_image_batch(xhat, xhat_path, batch['img_file_name'])
                if source_dist_samples_to_save is None:
                    source_dist_samples_to_save = source_dist_samples

            source_distribution_samples_path = os.path.join(self.test_results_path, 'source_distribution_samples')
            save_image_batch(source_dist_samples_to_save, source_distribution_samples_path, batch['img_file_name'])
            if self.mmse_model is not None:
                mmse_estimates = self.mmse_model(y).clip(0, 1)
                mmse_samples_path = os.path.join(self.test_results_path, 'mmse_samples')
                save_image_batch(mmse_estimates, mmse_samples_path, batch['img_file_name'])


        else:
            xhat, _, _ = self.generate_reconstructions(x, y, non_noisy_z0, None, torch.device('cpu'))
            xhat_path = os.path.join(self.test_results_path, 'xhat')
            save_image_batch(xhat, xhat_path, batch['img_file_name'])

    def configure_optimizers(self):
        # Add here a learning rate scheduler if you wish to do so.
        optimizer = AdamW(self.model.parameters(),
                          betas=self.hparams.betas,
                          eps=1e-8,
                          lr=self.hparams.lr,
                          weight_decay=self.hparams.weight_decay)
        return optimizer
