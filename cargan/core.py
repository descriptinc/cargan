import torch

import cargan


###############################################################################
# Model inference
###############################################################################


def from_audio(
    audio,
    sample_rate,
    checkpoint=cargan.DEFAULT_CHECKPOINT,
    gpu=None):
    """Perform vocoding from audio
    
    Arguments
        audio : torch.Tensor(shape=(1, samples))
            The audio to vocode
        sample_rate : int
            The audio sample rate
        gpu : int or None
            The index of the gpu to use
    
    Returns
        vocoded : torch.Tensor(shape=(1, samples))
            The vocoded audio
    """
    # Compute features
    features = cargan.preprocess.mels.from_audio(audio, sample_rate)

    # Vocode
    return from_features(features, checkpoint, gpu)


def from_audio_file_to_file(
    audio_file,
    output_file,
    checkpoint=cargan.DEFAULT_CHECKPOINT,
    gpu=None):
    """Perform vocoding from audio file and save to file
    
    Arguments
        audio_file : Path
            The audio file to vocode
        output_file : Path
            The location to save the vocoded audio
        checkpoint : Path
            The generator checkpoint
        gpu : int or None
            The index of the gpu to use
    """
    audio = cargan.load.audio(audio_file)
    vocoded = from_audio(audio, cargan.SAMPLE_RATE, checkpoint, gpu)
    torchaudio.save(output_file, vocoded.cpu(), cargan.SAMPLE_RATE)


def from_audio_files_to_files(
    audio_files,
    output_files,
    checkpoint=cargan.DEFAULT_CHECKPOINT,
    gpu=None):
    """Perform vocoding from audio files and save to files

    Arguments
        audio_files : list(Path)
            The audio files to vocode
        output_files : list(Path)
            The locations to save the vocoded audio
        checkpoint : Path
            The generator checkpoint
        gpu : int or None
            The index of the gpu to use
    """
    for audio_file, output_file in zip(audio_files, output_files):
        from_audio_file_to_file(audio_file, output_file, checkpoint, gpu)


def from_features(
    features,
    checkpoint=cargan.DEFAULT_CHECKPOINT,
    gpu=None):
    """Perform vocoding from features
    
    Arguments
        features : torch.Tensor(shape=(1, cargan.NUM_FEATURES, frames)
            The features to vocode
        gpu : int or None
            The index of the gpu to use
    
    Returns
        vocoded : torch.Tensor(shape=(1, cargan.HOPSIZE * frames))
            The vocoded audio
    """
    # Cache model
    if (not hasattr(from_features, 'model') or
        from_features.checkpoint != checkpoint or
        from_features.gpu != gpu):
        from_features.model = cargan.load.model(checkpoint, gpu)
        from_features.checkpoint = checkpoint
        from_features.gpu = gpu
    
    # Place features on device
    device = torch.device('cpu' if gpu is None else f'cuda:{gpu}')
    features = features.to(device)

    # Vocode
    with torch.no_grad():
        if cargan.AUTOREGRESSIVE:
            vocoded = ar_loop(from_features.model, features)
        else:
            vocoded = from_features.model(features)
        
    return vocoded.squeeze(0)


def from_feature_file_to_file(
    feature_file,
    output_file,
    checkpoint=cargan.DEFAULT_CHECKPOINT,
    gpu=None):
    """Perform vocoding from feature file and save to disk
    
    Arguments
        feature_file : Path
            The feature file to vocode
        output_file : Path
            The location to save the vocoded audio
        checkpoint : Path
            The generator checkpoint
        gpu : int or None
            The index of the gpu to use
    """
    features = torch.load(feature_file, map_location='cpu')
    vocoded = from_features(features, checkpoint, gpu)
    torchaudio.save(output_file, vocoded, cargan.SAMPLE_RATE)


def from_feature_files_to_files(
    feature_files,
    output_files,
    checkpoint=cargan.DEFAULT_CHECKPOINT,
    gpu=None):
    """Perform vocoding from feature files and save to disk
    
    Arguments
        feature_files : list(Path)
            The feature files to vocode
        output_files : list(Path)
            The locations to save the vocoded audio
        checkpoint : Path
            The generator checkpoint
        gpu : int or None
            The index of the gpu to use
    """
    for feature_file, output_file in zip(feature_files, output_files):
        from_feature_file_to_file(feature_file, output_file, checkpoint, gpu)


###############################################################################
# Autoregressive inference loop
###############################################################################


def ar_loop(model, features):
    """Perform autoregressive inference"""
    if cargan.CUMSUM:
        # Save output size
        output_length = features.shape[2]

        # Get feature chunk size
        feat_chunk = cargan.CHUNK_SIZE

    else:
        # Save output size
        output_length = features.shape[2] * cargan.HOPSIZE

        # Get feature chunk size
        feat_chunk = cargan.CHUNK_SIZE // cargan.HOPSIZE

    # Pad features to be a multiple of the chunk size
    padding = (feat_chunk - (features.shape[2] % feat_chunk)) % feat_chunk
    features = torch.nn.functional.pad(features, (0, padding))
    
    # Start with all zeros as conditioning
    prev_samples = torch.zeros(
        (1, 1, cargan.AR_INPUT_SIZE),
        dtype=features.dtype,
        device=features.device)

    # Get output signal length
    if cargan.CUMSUM:
        signal_length = features.shape[2]
        feat_hop = cargan.AR_HOPSIZE
    else:
        signal_length = features.shape[2] * cargan.HOPSIZE
        feat_hop = cargan.AR_HOPSIZE // cargan.HOPSIZE
    
    # Autoregressive loop
    signals = torch.zeros(
        signal_length,
        dtype=features.dtype,
        device=features.device)
    with torch.no_grad():
        for i in range(0, features.shape[2] - feat_chunk + 1, feat_hop):
            signal = model(features[:, :, i:i + feat_chunk], prev_samples)

            # Place newly generated chunk
            start = i if cargan.CUMSUM else i * cargan.HOPSIZE
            signals[start:start + cargan.CHUNK_SIZE] += signal.squeeze()

            # Update AR context
            if cargan.AR_INPUT_SIZE <= cargan.CHUNK_SIZE:
                prev_samples = signal[:, :, -cargan.AR_INPUT_SIZE:]
            else:
                prev_samples[:, :, :-cargan.CHUNK_SIZE] = \
                    prev_samples[:, :, cargan.CHUNK_SIZE:].clone()
                prev_samples[:, :, -cargan.CHUNK_SIZE:] = signal
        
        # Concatenate and remove padding
        return signals[None, None, :output_length]
