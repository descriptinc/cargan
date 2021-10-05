import torch

import cargan


###############################################################################
# Collate function
###############################################################################


def cargan_collate(batch, partition='train'):
    """Collate batch for training cargan"""
    features, audio, pitch, periodicity = zip(*batch)

    # Crop and pad features
    features, pitch, periodicity, start_idxs = prepare_features(
        features,
        pitch,
        periodicity,
        partition)

    # Crop audio
    audio = prepare_audio(audio, partition, start_idxs)

    # Maybe remove some features
    if cargan.AUTOREGRESSIVE and partition in ['train', 'valid']:
        if cargan.CUMSUM:
            start = cargan.AR_INPUT_SIZE
        else:
            start = cargan.AR_INPUT_SIZE // cargan.HOPSIZE
        features = features[:, :, start:]

        if not cargan.CUMSUM:
            pitch = pitch[:, :, start:]
            periodicity = periodicity[:, :, start:]

    return features, audio, pitch, periodicity


###############################################################################
# Collate utilities
###############################################################################


def prepare_audio(audio_list, partition, start_idxs):
    """Pad a batch of audio"""
    # Prepare audio for training or validation
    if partition in ['train', 'valid']:

        # Convert indices from frames to samples
        if not cargan.CUMSUM:
            start_idxs *= cargan.HOPSIZE

        # Crop audio
        return torch.stack([
            x[:, start_idx:start_idx + cargan.TRAIN_AUDIO_LENGTH]
            for x, start_idx in zip(audio_list, start_idxs)])
    
    # Prepare audio for testing
    elif partition == 'test':
        assert len(audio_list) == 1
        return audio_list[0][None]
    
    raise ValueError('Partition mush be one of ["train", "valid", "test"]')


def prepare_features(features_list, pitch_list, period_list, partition):
    """Pad a batch of input features"""
    if cargan.CUMSUM:
        length = cargan.TRAIN_AUDIO_LENGTH
    else:
        length = cargan.TRAIN_FEATURE_LENGTH
    
    # Prepare features for training
    if partition == 'train':
        features = []
        pitches = []
        periods = []
        start_idxs = []
        for feature, pitch, period in zip(features_list, pitch_list, period_list):

            # Get start index
            if not cargan.CUMSUM or cargan.AUTOREGRESSIVE:
                start_idx = torch.randint(
                    0,
                    1 + max(0, feature.shape[-1] - length),
                    (1,)).item()
            else:
                start_idx = 0

            # Save slice and start point
            if cargan.CUMSUM:
                features.append(feature[start_idx:start_idx + length])
            else:
                features.append(feature[:, :, start_idx:start_idx + length])
                pitches.append(pitch[:, :, start_idx:start_idx + length])
                periods.append(period[:, :, start_idx:start_idx + length])
            start_idxs.append(start_idx)

        # Convert to arrays
        if cargan.CUMSUM:
            features = torch.stack(features)[:, None]
        else:
            features = torch.cat(features)
            pitches = torch.cat(pitches)
            periods = torch.cat(periods)
        start_idxs = torch.tensor(start_idxs, dtype=torch.int)

    # Prepare features for validation (start_idxs is all zeros)
    elif partition == 'valid':
        if cargan.CUMSUM:
            features = torch.stack(
                [feat[:length] for feat in features_list])[:, None]
            pitches, periods = [None] * len(features), [None] * len(features)
        else:
            features = torch.cat(
                [feat[:, :, :length] for feat in features_list])
            pitches = torch.cat(
                [pitch[:, :, :length] for pitch in pitch_list])
            periods = torch.cat(
                [period[:, :, :length] for period in period_list])
        start_idxs = torch.zeros(len(features), dtype=torch.int)
    
    # Prepare features for testing
    elif partition == 'test':
        assert len(features_list) == 1
        if cargan.CUMSUM:
            features = features_list[0][None, None]
            pitches, periods = [None], [None]
        else:
            features = features_list[0]
            pitches = pitch_list[0]
            periods = period_list[0]
        start_idxs = torch.zeros(1, dtype=torch.int)
    
    # Raise on bad partition
    else:
        raise ValueError(
            f'Partition must be one of ["train", "valid", "test"]')

    return features, pitches, periods, start_idxs
