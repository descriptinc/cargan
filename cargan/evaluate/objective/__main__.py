import argparse
from pathlib import Path

import cargan


def main(name, datasets, checkpoint, num=256, gpu=None):
    """Evaluate datasets"""
    # Evaluate cumsum
    if 'cumsum' in datasets:
        cargan.evaluate.objective.cumsum(name, checkpoint, num, gpu)
        datasets.remove('cumsum')

    # Evaluate pitch metrics
    if datasets:
        cargan.evaluate.objective.pitch(name, datasets, checkpoint, num, gpu)


###############################################################################
# Entry point
###############################################################################


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description='Perform objective evaluation')
    parser.add_argument(
        '--name',
        required=True,
        help='A name to give this evaluation run')
    parser.add_argument(
        '--datasets',
        nargs='+',
        required=True,
        help='The datasets to evaluate')
    parser.add_argument(
        '--checkpoint',
        type=Path,
        required=True,
        help='The model checkpoint to use for inference')
    parser.add_argument(
        '--num',
        type=int,
        default=256,
        help='The number of samples to evaluate')
    parser.add_argument(
        '--gpu',
        type=int,
        help='The index of the gpu to use')
    return parser.parse_args()


main(**vars(parse_args()))
