import os

import torch
from PIL import Image
from torch.utils.data import Dataset


class ImageFolderDataset(Dataset):
    def __init__(self, root, degradation, transform=None, source_samples_root=None):
        self.root_dir = root
        self.img_file_names = list(sorted(os.listdir(root)))
        self.source_samples_root = source_samples_root
        if source_samples_root is not None:
            self.source_samples_img_file_names = list(sorted(os.listdir(source_samples_root)))
            assert len(self.source_samples_img_file_names) == len(self.img_file_names)
            for i, source_img_name in enumerate(self.source_samples_img_file_names):
                assert source_img_name == self.img_file_names[i]

        self.transform = transform
        self.degradation = degradation

    def __len__(self):
        return len(self.img_file_names)

    def __getitem__(self, idx):
        # There may be some torch operations here that calculate gradients and slow down data loading.
        # We don't want that to happen.
        with torch.no_grad():
            img_path = os.path.join(self.root_dir, self.img_file_names[idx])
            x = Image.open(img_path).convert('RGB')
            if self.transform:
                x = self.transform(x)
            y, maybe_x = self.degradation(x)
            if maybe_x is not None:
                x = maybe_x
            result = {'x': x, 'img_file_name': self.img_file_names[idx], 'y': y}
            if self.source_samples_root is not None:
                source_img_path = os.path.join(self.source_samples_root, self.source_samples_img_file_names[idx])
                source_img = Image.open(source_img_path).convert('RGB')
                if self.transform:
                    source_img = self.transform(source_img)
                result['non_noisy_z0'] = source_img
        return result
