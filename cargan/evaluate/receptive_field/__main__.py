import argparse
import torch

import cargan


def main(gpu=None):
    device = torch.device('cpu' if gpu is None else f'cuda:{gpu}')
    netG = cargan.GENERATOR().to(device)

    # Create dummy input and store gradient
    length = 524288 if cargan.CUMSUM else 2048
    x = torch.randn(1, cargan.NUM_FEATURES, length).to(device)
    x.requires_grad_(True)
    x.retain_grad()

    # Make a forward pass
    out = netG(x)

    # Create gradient variable
    print("Grad shape : ", out.shape)
    grad = torch.zeros_like(out)
    grad[:, :, grad.shape[-1] // 2] = 1

    # Make a backward pass
    out.backward(grad)

    # Check non-zero values
    gradmap = x.grad.squeeze(0)
    gradmap = (gradmap != 0).sum(0)  # sum across features
    rf = (gradmap != 0).sum()

    print(f"Receptive field : {rf}")


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, help='The index of the gpu to use')
    return parser.parse_args()


main(**vars(parse_args()))
