import argparse
import urllib
import shutil
import ssl
import tarfile
import zipfile
from pathlib import Path

import torch
import torchaudio
import tqdm

import cargan


###############################################################################
# Download datasets
###############################################################################


def datasets_from_cloud(datasets):
    """Download datasets from cloud storage"""
    # Download and format vctk dataset
    if 'vctk' in datasets:
        vctk()
    
    # Download and format musdb dataset
    if 'musdb' in datasets:
        musdb()

    # Download and format daps dataset
    if 'daps' in datasets:
        daps()

    # Create cumsum dataset from vctk
    if 'cumsum' in datasets:
        cumsum()


def cumsum():
    """Create cumsum dataset from vctk"""
    # Make sure vctk exists
    input_directory = cargan.CACHE_DIR / 'vctk'
    if not input_directory.exists():
        raise ValueError('Cannot create cumsum dataset before vctk')

    # Get audio files
    audio_files = list(input_directory.rglob('*.wav'))

    # Seed random generator
    torch.manual_seed(cargan.RANDOM_SEED)

    # Write cumsums to cache
    output_directory = cargan.CACHE_DIR / 'cumsum'
    output_directory.mkdir(exist_ok=True, parents=True)
    with cargan.data.chdir(output_directory):

        # Iterate over files
        iterator = tqdm.tqdm(
            enumerate(audio_files),
            desc='Formatting cumsum',
            dynamic_ncols=True,
            total=len(audio_files))
        for i, audio_file in iterator:

            # Load audio
            audio = cargan.load.audio(audio_file).squeeze()

            # Create cumsum input and output
            cumsum_input = torch.rand_like(audio)
            cumsum_output = torch.cumsum(cumsum_input, dim=0)

            # Normalize to [0, 1]
            maximum = cumsum_output[-1].clone()
            cumsum_input /= maximum
            cumsum_output /= maximum

            # Save to cache
            torch.save(cumsum_input, f'input-{i:06d}.pt')
            torch.save(cumsum_output, f'output-{i:06d}.pt')


def daps():
    """Download daps dataset"""
    # Download
    url = 'https://zenodo.org/record/4783456/files/daps-segmented.tar.gz?download=1'
    file = cargan.DATA_DIR / 'daps.tar.gz'
    download_file(url, file)

    # Unzip
    input_directory = cargan.DATA_DIR / 'daps'
    input_directory.mkdir(exist_ok=True, parents=True)
    with tarfile.open(file, 'r:gz') as tfile:
        tfile.extractall(input_directory)

    # Get audio files
    audio_files = list(input_directory.rglob('*.wav'))

    # Write audio to cache
    output_directory = cargan.CACHE_DIR / 'daps'
    output_directory.mkdir(exist_ok=True, parents=True)
    with cargan.data.chdir(output_directory):

        # Iterate over files
        iterator = tqdm.tqdm(
            enumerate(audio_files),
            desc='Formatting daps',
            dynamic_ncols=True,
            total=len(audio_files))
        for i, audio_file in iterator:

            # Convert to 22.05k
            audio = cargan.load.audio(audio_file)

            # If audio is too quiet, increase the volume
            maximum = torch.abs(audio).max()
            if maximum < .35:
                audio *= .35 / maximum
            
            # Save to disk
            torchaudio.save(f'{i:06d}.wav', audio, cargan.SAMPLE_RATE)


def musdb():
    """Download musdb dataset"""
    # Download
    url = 'https://zenodo.org/record/3338373/files/musdb18hq.zip?download=1'
    file = cargan.DATA_DIR / 'musdb.zip'
    download_file(url, file)

    # Unzip
    input_directory = cargan.DATA_DIR / 'musdb'
    input_directory.mkdir(exist_ok=True, parents=True)
    with zipfile.ZipFile(file, 'r') as zfile:
        zfile.extractall(input_directory)

    # Get audio files
    audio_files = [
        f for f in input_directory.rglob('*.wav') if f.stem != 'mixture']

    # Write audio to cache
    output_directory = cargan.CACHE_DIR / 'musdb'
    output_directory.mkdir(exist_ok=True, parents=True)
    with cargan.data.chdir(output_directory):

        # Iterate over files creating chunks of audio
        i = 0
        iterator = tqdm.tqdm(
            audio_files,
            desc='Formatting musdb',
            dynamic_ncols=True,
            total=len(audio_files))
        for audio_file in iterator:

            # Convert to 22.05k mono
            audio = cargan.load.audio(audio_file)
            audio = audio.mean(dim=0, keepdim=True)

            # If audio is too quiet, increase the volume
            maximum = torch.abs(audio).max()
            if maximum < .35:
                audio *= .35 / maximum

            # Handle long audio
            max_len = cargan.MAX_LENGTH
            if audio.numel() < max_len:
                if not (audio == 0).all():
                    chunks = [audio]
            else:
                j = 0
                chunks = []
                while (j + 1) * max_len < audio.numel():
                    chunk = audio[:, j * max_len:(j + 1) * max_len]
                    if not (chunk == 0).all():
                        chunks.append(chunk)
                    j += 1
            
            # Save chunks
            for chunk in chunks:
                torchaudio.save(f'{i:08d}.wav', chunk, cargan.SAMPLE_RATE)
                i += 1


def vctk():
    """Download vctk dataset"""
    # Download
    url = 'https://datashare.ed.ac.uk/download/DS_10283_3443.zip'
    file = cargan.DATA_DIR / 'vctk.zip'
    download_file(url, file)
        
    # Unzip
    directory = cargan.DATA_DIR / 'vctk'
    with zipfile.ZipFile(file, 'r') as zfile:
        zfile.extractall(directory)
    file = next((directory).glob('*.zip'))
    with zipfile.ZipFile(file) as zfile:
        zfile.extractall(directory)

    # Audio location
    audio_directory = directory / 'wav48_silence_trimmed'

    # Get source files
    audio_files = list(audio_directory.rglob('*.flac'))

    # Write audio to cache
    output_directory = cargan.CACHE_DIR / 'vctk'
    output_directory.mkdir(exist_ok=True, parents=True)
    with cargan.data.chdir(output_directory):

        # Iterate over files
        iterator = tqdm.tqdm(
            audio_files,
            desc='Formatting vctk',
            dynamic_ncols=True,
            total=len(audio_files))
        for audio_file in iterator:

            # Organize by speaker
            speaker_dir = Path(audio_file.stem.split('_')[0])
            speaker_dir.mkdir(exist_ok=True, parents=True)

            # Convert to 22.05k wav
            audio = cargan.load.audio(audio_file)

            # If audio is too quiet, increase the volume
            maximum = torch.abs(audio).max()
            if maximum < .35:
                audio *= .35 / maximum
            
            # Save to disk
            output_file = f'{len(list(speaker_dir.glob("*"))):06d}.wav'
            torchaudio.save(speaker_dir / output_file, audio, cargan.SAMPLE_RATE)
            
            
###############################################################################
# Utilities
###############################################################################


def download_file(url, file):
    """Download file from url"""
    with urllib.request.urlopen(url, context=ssl.SSLContext()) as response, \
         open(file, 'wb') as output:
        shutil.copyfileobj(response, output)


def get_dataset_length(audio_files):
    """Get the size of the dataset given the file paths"""
    length = 0
    for file in audio_files:
        info = torchaudio.info(file)
        samples = info.num_frames * cargan.SAMPLE_RATE / info.sample_rate
        if samples < cargan.MAX_LENGTH:
            length += 1
        else:
            length += samples // cargan.MAX_LENGTH
    return length


###############################################################################
# Entry point
###############################################################################


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description='Download datasets')
    parser.add_argument(
        '--datasets',
        nargs='+',
        required=True,
        help='The datasets to download')
    return parser.parse_args()


if __name__ == '__main__':
    datasets_from_cloud(**vars(parse_args()))
