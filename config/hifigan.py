import functools
from pathlib import Path

import torch

import cargan


###############################################################################
# Directories
###############################################################################

# Location of preprocessed features
CACHE_DIR = Path(__file__).parent.parent / 'data' / 'cache'

# Location of datasets on disk
DATA_DIR = Path(__file__).parent.parent / 'data' / 'datasets'

# Location to save assets to be bundled with pip release
ASSETS_DIR = Path(__file__).parent / 'assets'

# Location to save dataset partitions
PARTITION_DIR = ASSETS_DIR / 'partitions'

# Location to save logs, checkpoints, and configurations
RUNS_DIR = Path(__file__).parent.parent / 'runs'

# Location to save evaluation artifacts
EVAL_DIR = Path(__file__).parent.parent / 'eval'

# Default checkpoint for inference
DEFAULT_CHECKPOINT = ASSETS_DIR / 'checkpoints' / 'cargan.pt'


###############################################################################
# Features
###############################################################################


# Whether to condition the generator on pitch features
PITCH_FEATURE = False

# Whether to condition the discriminator on pitch and periodicity
PITCH_COND_DISCRIM = False

# Whether to condition the generator on periodicity features
PERIODICITY_FEATURE = False


###############################################################################
# Autoregression
###############################################################################


# Whether to use autoregressive GAN
AUTOREGRESSIVE = False

# Whether to pass autoregressive conditioning to the discriminator
AR_DISCRIM = True

# Autoregressive model hidden size
AR_HIDDEN_SIZE = 256

# Autoregressive hopsize
AR_HOPSIZE = 2048

# The number of samples of autoregressive conditioning
AR_INPUT_SIZE = 512

# The number of autoregressive samples to pass to the discriminator
AR_INPUT_SIZE_DISCRIM = 512

# The size of the autoregressive embedding
AR_OUTPUT_SIZE = 128


###############################################################################
# Cumsum experiment
###############################################################################


# Whether to use synthetic cumsum data without upsampling
CUMSUM = False


###############################################################################
# DSP parameters
###############################################################################


# Minimum and maximum frequency
FMIN = 50
FMAX = 550

# Audio hopsize
HOPSIZE = 256

# Maximum sample value of 16-bit audio
MAX_SAMPLE_VALUE = 32768

# Number of melspectrogram channels
NUM_MELS = 80

# Number of spectrogram channels
NUM_FFT = 1024

# Audio sample rate
SAMPLE_RATE = 22050


###############################################################################
# Training parameters
###############################################################################


# Batch size (per gpu)
BATCH_SIZE = 64

# Training chunk size
CHUNK_SIZE = 8192

# Discriminator architecture
DISCRIMINATOR = cargan.model.hifigan.Discriminator

# Generator architecture
GENERATOR = cargan.model.hifigan.Generator

# Maximum length of a training example
MAX_LENGTH = 10 * SAMPLE_RATE

# Maximum number of training steps
MAX_STEPS = 500000

# Number of input channels
if CUMSUM:
    NUM_FEATURES = 1 + AUTOREGRESSIVE * AR_OUTPUT_SIZE
else:
    NUM_FEATURES = (
        NUM_MELS + 
        PITCH_FEATURE + 
        PERIODICITY_FEATURE + 
        AUTOREGRESSIVE * AR_OUTPUT_SIZE)

# Number of input channels to the discriminator
NUM_DISCRIM_FEATURES = 1 + 2 * PITCH_COND_DISCRIM

# Number of data loading worker threads
NUM_WORKERS = 2

# The optimizer to use for training
OPTIMIZER = functools.partial(torch.optim.AdamW, lr=2e-4, betas=(.8, .99))

# Number of pitch bins for pitch classification discriminator
PITCH_BINS = 256

# Seed for all random number generators
RANDOM_SEED = 0

# Size of each partition. Must add to 1.
SPLIT_SIZE_TEST = .1
SPLIT_SIZE_TRAIN = .8
SPLIT_SIZE_VALID = .1

# Number of samples in a training example
TRAIN_AUDIO_LENGTH = CHUNK_SIZE + AR_INPUT_SIZE * AUTOREGRESSIVE

# Number of frames in a training example
TRAIN_FEATURE_LENGTH = TRAIN_AUDIO_LENGTH // HOPSIZE


###############################################################################
# Training parameters (loss)
###############################################################################


# Loss function to use for adversarial loss
# Options: 'hinge', 'mse', None
LOSS_ADVERSARIAL = 'mse'

# Whether to use CREPE perceptual loss
LOSS_CREPE = False

# CREPE perceptual loss weight
LOSS_CREPE_WEIGHT = 1.

# Whether to use feature matching loss
LOSS_FEAT_MATCH = True

# Feature matching loss weight
LOSS_FEAT_MATCH_WEIGHT = 2.

# Whether to use mel error loss
LOSS_MEL_ERROR = True

# Mel error loss weight
LOSS_MEL_ERROR_WEIGHT = 45.

# Whether to use pitch classification discriminator
LOSS_PITCH_DISCRIMINATOR = False

# Pitch discriminator loss weight
LOSS_PITCH_DISCRIMINATOR_WEIGHT = 1

# Whether to use L2 waveform loss
LOSS_WAVEFORM_ERROR = False

# Waveform error loss weight
LOSS_WAVEFORM_ERROR_WEIGHT = 1


###############################################################################
# Training parameters (logging)
###############################################################################


# Number of steps between logging
INTERVAL_LOG = 50

# Number of steps between logging phase error
INTERVAL_PHASE = 500

# Number of steps between pitch evaluation
INTERVAL_PITCH = 3500

# Number of steps between generating samples
INTERVAL_SAMPLE = 1000

# Number of steps between saving
INTERVAL_SAVE = 25000

# Number of steps between validation
INTERVAL_VALID = 500

# Number of steps between logging waveform MSE
INTERVAL_WAVEFORM = 500

# Number of samples to use for pitch evaluation
NUM_PITCH_SAMPLES = 64

# Number of samples to put on tensorboard
NUM_TEST_SAMPLES = 10
