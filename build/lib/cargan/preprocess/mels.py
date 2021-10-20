import librosa
import numpy as np
import torch
import torchaudio

import cargan


###############################################################################
# Interface
###############################################################################


def from_audio(audio, sample_rate=cargan.SAMPLE_RATE):
    """Compute mels from audio"""
    # Mayble resample
    if sample_rate != cargan.SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(
            sample_rate,
            cargan.SAMPLE_RATE)
        audio = resampler(audio)
    
    # Cache function for computing mels
    if not hasattr(from_audio, 'mels'):
        from_audio.mels = MelSpectrogram()
    
    # Compute mels
    return from_audio.mels(audio)


###############################################################################
# Mel spectrogram
###############################################################################


class MelSpectrogram(torch.nn.Module):

    def __init__(self):
        super().__init__()
        window = torch.hann_window(cargan.NUM_FFT, dtype=torch.float)
        mel_basis = librosa.filters.mel(
            cargan.SAMPLE_RATE,
            cargan.NUM_FFT,
            cargan.NUM_MELS
        ).astype(np.float32)
        mel_basis = torch.from_numpy(mel_basis)
        self.register_buffer("mel_basis", mel_basis)
        self.register_buffer("window", window)

    @property
    def device(self):
        return self.mel_basis.device

    def log_mel_spectrogram(self, audio):
        # Compute STFT
        stft = torch.stft(
            audio,
            n_fft=cargan.NUM_FFT,
            window=self.window,
            center=False,
            return_complex=False)
        
        # Compute magnitude spectrogram
        spectrogram = torch.sqrt(stft.pow(2).sum(-1) + 1e-9)

        # Compute melspectrogram
        melspectrogram = torch.matmul(self.mel_basis, spectrogram)

        # Compute logmelspectrogram
        return torch.log10(torch.clamp(melspectrogram, min=1e-5))

    def forward(self, audio):
        # Ensure correct shape
        if audio.dim() == 2:
            audio = audio[:, None, :]
        elif audio.dim() == 1:
            audio = audio[None, None, :]

        # Pad audio
        p = (cargan.NUM_FFT - cargan.HOPSIZE) // 2
        audio = torch.nn.functional.pad(audio, (p, p), "reflect").squeeze(1)

        # Compute logmelspectrogram
        return self.log_mel_spectrogram(audio)
