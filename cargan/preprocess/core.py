import torch
import tqdm

import cargan


###############################################################################
# Preprocessing
###############################################################################


def datasets(datasets, gpu=None):
    """Preprocess datasets"""
    for dataset in datasets:
        
        # Input/output directories
        directory = cargan.CACHE_DIR / dataset

        # Get audio files
        audio_files = list(directory.rglob('*.wav'))

        # Open cache dir
        with cargan.data.chdir(directory):

            # Iterate over files
            iterator = tqdm.tqdm(
                audio_files,
                desc=f'Preprocessing {dataset}',
                dynamic_ncols=True,
                total=len(audio_files))
            for audio_file in iterator:

                # Load audio
                audio = cargan.load.audio(audio_file)

                # Compute features
                mels, pitch, periodicity = from_audio(audio, gpu=gpu)

                # Save to disk
                torch.save(
                    mels,
                    audio_file.parent / f'{audio_file.stem}-mels.pt')
                torch.save(
                    pitch,
                    audio_file.parent / f'{audio_file.stem}-pitch.pt')
                torch.save(
                    periodicity,
                    audio_file.parent / f'{audio_file.stem}-periodicity.pt')


def from_audio(audio, sample_rate=cargan.SAMPLE_RATE, gpu=None):
    """Compute input features from audio"""
    # Maybe increase volume
    maximum = torch.abs(audio).max()
    if maximum < .35:
        audio *= .35 / maximum
    
    # Compute mels
    mels = cargan.preprocess.mels.from_audio(audio, sample_rate)
    
    # Compute pitch and periodicity
    pitch, periodicity = cargan.preprocess.pitch.from_audio(
        audio, sample_rate, gpu)
    pitch = torch.log2(pitch)

    return mels, pitch[None], periodicity[None]
