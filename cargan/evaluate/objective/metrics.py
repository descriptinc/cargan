import numpy as np
import torch
import torchaudio
import torchcrepe

import cargan


###############################################################################
# Pitch metrics
###############################################################################


class Pitch:

    def __init__(self):
        self.threshold = torchcrepe.threshold.Hysteresis()
        self.reset()
    
    def __call__(self):
        pitch_rmse = torch.sqrt(self.pitch_total / self.voiced)
        periodicity_rmse = torch.sqrt(self.periodicity_total / self.count)
        precision = \
            self.true_positives / (self.true_positives + self.false_positives)
        recall = \
            self.true_positives / (self.true_positives + self.false_negatives)
        f1 = 2 * precision * recall / (precision + recall)
        return {
            'pitch': pitch_rmse.item(),
            'periodicity': periodicity_rmse.item(),
            'f1': f1.item(),
            'precision': precision.item(),
            'recall': recall.item()}

    def reset(self):
        self.count = 0
        self.voiced = 0
        self.pitch_total = 0.
        self.periodicity_total = 0.
        self.true_positives = 0
        self.false_positives = 0
        self.false_negatives = 0
    
    def update(self, true_pitch, true_periodicity, pred_pitch, pred_periodicity):
        # Threshold
        true_threshold = self.threshold(true_pitch, true_periodicity)
        pred_threshold = self.threshold(pred_pitch, pred_periodicity)
        true_voiced = ~torch.isnan(true_threshold)
        pred_voiced = ~torch.isnan(pred_threshold)

        # Update periodicity rmse
        self.count += true_pitch.shape[1]
        self.periodicity_total += (true_periodicity - pred_periodicity).pow(2).sum()

        # Update pitch rmse
        voiced = true_voiced & pred_voiced
        self.voiced += voiced.sum()
        difference_cents = 1200 * (torch.log2(true_pitch[voiced]) - 
                                   torch.log2(pred_pitch[voiced]))
        self.pitch_total += difference_cents.pow(2).sum()
        
        # Update voiced/unvoiced precision and recall
        self.true_positives += (true_voiced & pred_voiced).sum()
        self.false_positives += (~true_voiced & pred_voiced).sum()
        self.false_negatives += (true_voiced & ~pred_voiced).sum()


###############################################################################
# Phase metrics
###############################################################################


class Phase:

    def __init__(self):
        self.reset()
        
    def __call__(self):
        return self.total / self.count
    
    def reset(self):
        self.count = 0
        self.total = 0.
    
    def update(self, x, y):
        x_m, x_p = self.magphase(x)
        _, y_p = self.magphase(y)

        # Take circular difference
        diff = x_p - y_p
        diff[diff < -np.pi] += 2 * np.pi
        diff[diff > np.pi] -= 2 * np.pi

        # Scale true magnitude to weights in [0, 1]
        x_min, x_max = x_m.min(), x_m.max()
        weights = (x_m - x_min) / (x_max - x_min)

        # Update running values
        self.count += x_m.shape[-1]
        self.total += (weights * diff ** 2).sum()
        
    @staticmethod
    def magphase(x):
        window = torch.hann_window(
            cargan.NUM_FFT,
            dtype=torch.float,
            device=x.device)
        stft = torch.stft(
            x.squeeze(1),
            n_fft=cargan.NUM_FFT,
            win_length=cargan.NUM_FFT,
            window=window,
            center=False,
            return_complex=False)
        return torchaudio.functional.magphase(stft)


###############################################################################
# Waveform metrics
###############################################################################


class RMSE:

    def __init__(self):
        self.reset()
    
    def __call__(self):
        return torch.sqrt(self.total / self.count).item()

    def reset(self):
        self.count = 0
        self.total = 0.

    def update(self, x, y):
        self.count += x.numel()
        self.total += ((x - y) ** 2).sum()

    
class L1:

    def __init__(self):
        self.reset()
    
    def __call__(self):
        return (self.total / self.count).item()

    def reset(self):
        self.count = 0
        self.total = 0.

    def update(self, x, y):
        self.count += x.numel()
        self.total += torch.abs(x - y).sum()
