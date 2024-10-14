from lightning_models.mmse_rectified_flow import MMSERectifiedFlow
import torch
from torchvision.transforms.functional import to_tensor
from torchvision.utils import save_image
import os
import cv2
from torch.utils.data import DataLoader
from torch_datasets.image_folder_dataset import ImageFolderDataset
from tqdm import tqdm
import numpy as np
import argparse
from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils import img2tensor, tensor2img
from facexlib.utils.face_restoration_helper import FaceRestoreHelper
from realesrgan.utils import RealESRGANer


torch.set_float32_matmul_precision('high')
torch.set_grad_enabled(False)



realesrgan_folder = "checkpoints"
os.makedirs(realesrgan_folder, exist_ok=True)
realesr_model_path = f"{realesrgan_folder}/RealESRGAN_x4plus.pth"
if not os.path.exists(realesr_model_path):
    os.system(
        f"wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth -O {realesrgan_folder}/RealESRGAN_x4plus.pth"
    )


def set_realesrgan():
    use_half = False
    if torch.cuda.is_available():  # set False in CPU/MPS mode
        no_half_gpu_list = ["1650", "1660"]  # set False for GPUs that don't support f16
        if not True in [
            gpu in torch.cuda.get_device_name(0) for gpu in no_half_gpu_list
        ]:
            use_half = True

    model = RRDBNet(
        num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2,
    )
    upsampler = RealESRGANer(
        scale=2,
        model_path="https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/RealESRGAN_x2plus.pth",
        model=model,
        tile=400,
        tile_pad=40,
        pre_pad=0,
        half=use_half,
    )
    return upsampler


upsampler = set_realesrgan()
def resize(img, size):
    # From https://github.com/sczhou/CodeFormer/blob/master/facelib/utils/face_restoration_helper.py
    h, w = img.shape[0:2]
    scale = size / min(h, w)
    h, w = int(h * scale), int(w * scale)
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    return cv2.resize(img, (w, h), interpolation=interp)

def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    def identity(x):
        return x, None
    ds = ImageFolderDataset(args.lq_data_path, degradation=identity, transform=to_tensor)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=0)
    output_path = os.path.join(args.output_dir, 'restored_images')
    os.makedirs(output_path, exist_ok=True)

    if args.ckpt_path_is_huggingface:
        model = MMSERectifiedFlow.from_pretrained(args.ckpt_path).cuda()
    else:
        ckpt = torch.load(args.ckpt_path, map_location="cpu")
        mmse_model_arch = ckpt['hyper_parameters']['mmse_model_arch']
        model = MMSERectifiedFlow.load_from_checkpoint(args.ckpt_path,
                                                       # Need to provide mmse_model_arch to
                                                       # make sure the model initializes it.
                                                       mmse_model_arch=mmse_model_arch,
                                                       mmse_model_ckpt_path=None,  # Will ignore the original path of the
                                                       # MMSE model used for training,
                                                       # and instead load it from the model checkpoint.
                                                       map_location='cpu').cuda()
        if model.ema_wanted:
            model.ema.load_state_dict(ckpt['ema'])
            model.ema.copy_to()
    if model.mmse_model is not None:
        output_path_mmse = os.path.join(args.output_dir, 'restored_images_posterior_mean')
        os.makedirs(output_path_mmse, exist_ok=True)


    torch.compile(model, mode='max-autotune')
    print("Compiled model")

    model.freeze()

    for batch in tqdm(dl):
        y = batch['y'].cuda()
        dummy_x = batch['x'].cuda()
        estimate = model.generate_reconstructions(dummy_x, y, None, args.num_flow_steps, torch.device("cpu"))[0]
        for i in tqdm(range(y.shape[0])):
            save_image(estimate[i], os.path.join(output_path, os.path.basename(batch['img_file_name'][i])))
        if model.mmse_model is not None:
            mmse_estimate = model.mmse_model(y)
            for i in tqdm(range(y.shape[0])):
                save_image(mmse_estimate[i],
                           os.path.join(output_path_mmse, os.path.basename(batch['img_file_name'][i])))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--ckpt_path', type=str, required=False,
                        default='./checkpoints/blind_face_restoration_pmrf.ckpt',
                        help='Path to the model checkpoint.')
    parser.add_argument('--ckpt_path_is_huggingface', action='store_true', required=False, default=False,
                        help='Whether the ckpt path is a huggingface model or a path to a local file.')
    parser.add_argument('--lq_data_path', type=str, required=True,
                        help='Path to a folder that contains low quality images.')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Path to a folder where the reconstructed images will be saved.')
    parser.add_argument('--num_flow_steps', type=int, required=False, default=25,
                        help='Number of flow steps to use for inference.')
    parser.add_argument('--batch_size', type=int, required=False, default=64,
                        help='Batch size for inference.')
    parser.add_argument('--seed', type=int, required=False, default=0,
                        help='The input random seed.')

    main(parser.parse_args())
