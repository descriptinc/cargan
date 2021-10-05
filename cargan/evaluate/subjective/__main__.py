import argparse
import time
from pathlib import Path

import torchaudio
import tqdm

import cargan


def main(name, datasets, checkpoint, num, gpu=None):
    """Generate files for subjective evaluation"""
    # Evaluate on each dataset
    for dataset in datasets:

        # Setup output directories
        directory = cargan.EVAL_DIR / 'subjective' / dataset
        original_dir = directory / 'original'
        generated_dir = directory / name
        original_dir.mkdir(exist_ok=True, parents=True)
        generated_dir.mkdir(exist_ok=True, parents=True)

        # Setup data loader
        loader = cargan.data.loaders(datasets)[2]
        iterator = tqdm.tqdm(
            loader,
            dynamic_ncols=True,
            desc='Generating',
            total=num)
        samples = 0
        seconds = 0.
        i = 0
        loaded = False
        for features, audio, _, _ in iterator:

            # Stop once we've generated enough          
            if i >= num:
                break
            
            # VCTK has a lot of short samples. Running subjective tests on very
            # short samples is less conclusive. So we impose a minimum length.
            if dataset == 'vctk' and audio.numel() < 3 * cargan.SAMPLE_RATE:
                continue
            i += 1

            # Save original audio
            sample_name = f'sample-{i:04d}.wav'
            torchaudio.save(
                original_dir / sample_name,
                audio.squeeze(0),
                cargan.SAMPLE_RATE)
            
            # Dummy call to place the model on GPU outside of timer
            if not loaded:
                cargan.from_features(features, checkpoint, gpu)
                loaded = True
            
            # Vocode
            start = time.time()
            vocoded = cargan.from_features(features, checkpoint, gpu)
            seconds += time.time() - start
            samples += vocoded.shape[1]

            # Save vocoded audio
            torchaudio.save(
                generated_dir / sample_name,
                vocoded.cpu(),
                cargan.SAMPLE_RATE)
    
    # Report generation time
    print(f'Generated {samples / 1e6:.2f} million samples in {seconds:.3f} seconds')
    print(f'Inference speed is {samples / cargan.SAMPLE_RATE / seconds:.2f}x faster than real time')


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description='Generate files for subjective evaluation')
    parser.add_argument(
        '--name',
        required=True,
        help='The name to give to the experiment condition')
    parser.add_argument(
        '--datasets',
        required=True,
        nargs='+',
        help='The dataset to generate from')
    parser.add_argument(
        '--checkpoint',
        type=Path,
        required=True,
        help='The checkpoint file')
    parser.add_argument(
        '--num',
        type=int,
        default=100,
        help='The number of samples to generate')
    parser.add_argument(
        '--gpu',
        type=int,
        help='The index of the gpu to run inference on')
    return parser.parse_args()


main(**vars(parse_args()))
