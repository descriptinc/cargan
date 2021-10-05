import json

import numpy as np
import torch
import torchaudio

import cargan


###############################################################################
# Loading utilities
###############################################################################


def audio(file):
    """Load audio and maybe resample"""
    # Load
    audio, sample_rate = torchaudio.load(file)

    # Maybe resample
    if sample_rate != cargan.SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(
            sample_rate,
            cargan.SAMPLE_RATE)
        audio = resampler(audio)

    return audio


def model(checkpoint, gpu=None):
    """Load model from checkpoint"""
    device = torch.device('cpu' if gpu is None else f'cuda:{gpu}')
    checkpoint = torch.load(checkpoint, map_location=device)
    generator = cargan.GENERATOR()
    generator.load_state_dict(checkpoint)
    generator.eval()
    return generator.to(device)

    

def partitions(datasets):
    """Load partition file for datasets"""
    offset = 0
    all_train, all_valid, all_test = [], [], []
    for dataset in datasets:

        # Load partition
        with open(cargan.PARTITION_DIR / f'{dataset}.json') as file:
            partition = json.load(file)
        
        # Add offset into concatenated dataset
        train = np.array(partition['train']) + offset
        valid = np.array(partition['valid']) + offset
        test = np.array(partition['test']) + offset
        
        # Increment offset by total length
        offset += len(train) + len(valid) + len(test)

        # For vctk, replace test set with (seen) validation set
        if dataset == 'vctk':
            test = valid.copy()
    
        # Update indices
        all_train.extend(list(train))
        all_valid.extend(list(valid))
        all_test.extend(list(test))

    return {'train': all_train, 'valid': all_valid, 'test': all_test}
    