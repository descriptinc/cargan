import argparse
import json
import random

import cargan


###############################################################################
# Partition
###############################################################################


def dataset(name):
    """Partition datasets and save partitions to disk"""
    # Handle vctk
    if name == 'vctk':
        return vctk()
    
    # Get the data indices
    length = len(cargan.data.Dataset(name))
    indices = list(range(length))

    # Deterministically shuffle indices
    random.seed(cargan.RANDOM_SEED)
    if name != 'musdb':
        random.shuffle(indices)
    
    # Daps is eval-only
    if name == 'daps':
        return {'train': [], 'valid': [], 'test': indices}

    # Get split locations
    left = int(cargan.SPLIT_SIZE_TRAIN * length)
    right = left + int(cargan.SPLIT_SIZE_VALID * length)

    # Shuffle musdb test set
    test = indices[right:]

    # Split into partitions
    return {
        'train': indices[:left],
        'valid': indices[left:right],
        'test': test}


def vctk():
    """Partition the vctk dataset"""
    # Get list of speakers
    dataset = cargan.data.Dataset('vctk')
    speakers = dataset.speakers()

    # Shuffle speakers
    random.seed(cargan.RANDOM_SEED)
    random.shuffle(speakers)
    
    # Select test speakers
    test_speakers = speakers[:8]
    
    # Get test partition indices
    test_indices = [
        i for i in range(len(dataset))
        if dataset.speaker(i) in test_speakers]
    
    # Shuffle so adjacent indices aren't always same speaker
    random.shuffle(test_indices)
    
    # Get residual indices
    indices = list(range(len(dataset)))
    indices = [i for i in indices if i not in test_indices]
    random.shuffle(indices)

    # Split into train/valid
    split = int(.95 * len(indices))
    train_indices = indices[:split]
    valid_indices = indices[split:]

    return {
        'train': train_indices,
        'valid': valid_indices,
        'test': test_indices}
    

###############################################################################
# Entry point
###############################################################################


def main(datasets, overwrite):
    """Partition datasets and save to disk"""
    for name in datasets:

        # Check if partition already exists
        file = cargan.PARTITION_DIR / f'{name}.json'
        if file.exists():
            if not overwrite:
                print(f'Not overwriting existing partition {file}')
                continue

        # Save to disk
        with open(file, 'w') as file:
            json.dump(dataset(name), file, ensure_ascii=False, indent=4)


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description='Partition datasets')
    parser.add_argument(
        '--datasets',
        nargs='+',
        help='The datasets to partition')
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Whether to overwrite existing partitions')
    return parser.parse_args()


if __name__ == '__main__':
    main(**vars(parse_args()))
    