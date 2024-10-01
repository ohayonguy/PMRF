from torchvision.utils import make_grid


def create_grid(img, normalize=False, num_images=5):
    return make_grid(img[:num_images], padding=0, normalize=normalize, nrow=16)
