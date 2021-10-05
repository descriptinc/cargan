dependencies = [
    'librosa',
    'numpy',
    'torch',
    'torchaudio',
    'torchcrepe',
    'tqdm']
import torch

import cargan


def cargan(pretrained=False):
    """PyTorch Hub publishing script"""
    model = cargan.model.gantts.Generator()

    if pretrained:
        state_dict = torch.load(cargan.DEFAULT_CHECKPOINT, map_location='cpu')
        model.load_state_dict(state_dict)

    return model
