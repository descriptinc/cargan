import argparse

import cargan


###############################################################################
# Preprocessing entry point
###############################################################################


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--datasets',
        nargs='+',
        required=True,
        help='The datasets to preprocess')
    parser.add_argument(
        '--gpu',
        type=int,
        help='The index of the gpu to use')
    return parser.parse_args()


cargan.preprocess.datasets(**vars(parse_args()))
