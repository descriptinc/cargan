import cargan


def test_core(audio):
    """Test inference"""
    # Preprocess
    features = cargan.preprocess.mels.from_audio(audio)

    # Vocode
    vocoded_from_features = cargan.from_features(features)
    vocoded_from_audio = cargan.from_audio(audio, cargan.SAMPLE_RATE)

    # Should be deterministic
    assert (vocoded_from_audio == vocoded_from_features).all()

    # Should be correct length
    assert vocoded_from_audio.shape[-1] == features.shape[-1] * cargan.HOPSIZE
    