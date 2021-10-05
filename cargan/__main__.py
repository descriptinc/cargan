import argparse
from pathlib import Path

import cargan


###############################################################################
# Model inference
###############################################################################


def main(audio_files, feature_files, output_files, checkpoint, gpu=None):
    """Perform vocoding"""
    if audio_files is not None:

        # Vocode audio
        cargan.from_audio_files_to_files(
            audio_files,
            output_files,
            checkpoint,
            gpu)
    
    else:

        # Vocode features
        cargan.from_feature_files_to_files(
            feature_files,
            output_files,
            checkpoint,
            gpu)


###############################################################################
# Entry point
###############################################################################


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description='Perform inference on audio or features and save to disk')
    parser.add_argument(
        '--audio_files',
        type=Path,
        nargs='+',
        help='The audio files to vocode')
    parser.add_argument(
        '--feature_files',
        type=Path,
        nargs='+',
        help='The pytorch features files to vocode')
    parser.add_argument(
        '--output_files',
        type=Path,
        nargs='+',
        help='The files to write the vocoded audio')
    parser.add_argument(
        '--checkpoint',
        type=Path,
        required=True,
        default=cargan.ASSETS_DIR / 'checkpoints' / 'cargan.pt',
        help='The generator checkpoint file')
    parser.add_argument(
        '--gpu',
        type=int,
        help='The index of the gpu to use')
    
    # Parse
    args = parser.parse_args()

    # Error check
    if ((args.audio_files is None and args.feature_files is None) or
        (args.audio_files is not None and args.feature_files is not None)):
        raise ValueError(
            'Only one of "audio_files" or "feature_files" should be given')
    
    return args


main(**vars(parse_args()))
