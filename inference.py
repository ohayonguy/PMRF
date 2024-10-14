from lightning_models.mmse_rectified_flow import MMSERectifiedFlow
import torch
from torchvision.transforms.functional import to_tensor
from torchvision.utils import save_image
import os
from torch.utils.data import DataLoader
from torch_datasets.image_folder_dataset import ImageFolderDataset
from tqdm import tqdm
import argparse

torch.set_float32_matmul_precision('high')
torch.set_grad_enabled(False)


def main(args):
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

    main(parser.parse_args())
