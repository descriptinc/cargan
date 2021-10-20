# Chunked Autoregressive GAN (CARGAN)
[![PyPI](https://img.shields.io/pypi/v/cargan.svg)](https://pypi.python.org/pypi/cargan)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Downloads](https://pepy.tech/badge/cargan)](https://pepy.tech/project/cargan)

Official implementation of the paper _Chunked Autoregressive GAN for Conditional Waveform Synthesis_ [[paper]](https://www.maxrmorrison.com/pdfs/morrison2022chunked.pdf) [[companion website]](https://www.maxrmorrison.com/sites/cargan/)


## Table of contents

- [Installation](#installation)
- [Configuration](#configuration)
- [Inference](#inference)
    * [CLI](#cli)
    * [API](#api)
        * [`cargan.from_audio`](#carganfrom_audio)
        * [`cargan.from_audio_file_to_file`](#carganfrom_audio_file_to_file)
        * [`cargan.from_audio_files_to_files`](#carganfrom_audio_files_to_files)
        * [`cargan.from_features`](#carganfrom_features)
        * [`cargan.from_feature_file_to_file`](#carganfrom_feature_file_to_file)
        * [`cargan.from_feature_files_to_files`](#carganfrom_feature_files_to_files)
- [Reproducing results](#reproducing-results)
    * [Download](#download)
    * [Partition](#partition)
    * [Preprocess](#preprocess)
    * [Train](#train)
    * [Evaluate](#evaluate)
        * [Objective](#objective)
        * [Subjective](#subjective)
        * [Receptive field](#receptive-field)
- [Running tests](#running-tests)
- [Citation](#citation)


## Installation

`pip install cargan`


## Configuration

All configuration is performed in `cargan/constants.py`. The default configuration is
CARGAN. Additional configuration files for experiments described in our paper
can be found in `config/`.


## Inference

### CLI

Infer from an audio files on disk. `audio_files` and `output_files` can be
lists of files to perform batch inference.

```
python -m cargan \
    --audio_files <audio_files> \
    --output_files <output_files> \
    --checkpoint <checkpoint> \
    --gpu <gpu>
```

Infer from files of features on disk. `feature_files` and `output_files` can
be lists of files to perform batch inference.

```
python -m cargan \
    --feature_files <feature_files> \
    --output_files <output_files> \
    --checkpoint <checkpoint> \
    --gpu <gpu>
```


### API

#### `cargan.from_audio`

```
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
```

#### `cargan.from_audio_file_to_file`

```
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
```


#### `cargan.from_audio_files_to_files`

```
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
```


#### `cargan.from_features`

```
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
```


#### `cargan.from_feature_file_to_file`

```
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
```


#### `cargan.from_feature_files_to_files`

```
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
```


## Reproducing results

For the following subsections, the arguments are as follows
- `checkpoint` - Path to an existing checkpoint on disk
- `datasets` - A list of datasets to use. Supported datasets are
  `vctk`, `daps`, `cumsum`, and `musdb`.
- `gpu` - The index of the gpu to use
- `gpus` - A list of indices of gpus to use for distributed data parallelism
  (DDP)
- `name` - The name to give to an experiment or evaluation
- `num` - The number of samples to evaluate


### Download

Downloads, unzips, and formats datasets. Stores datasets in `data/datasets/`.
Stores formatted datasets in `data/cache/`.

```
python -m cargan.data.download --datasets <datasets>
```

`vctk` must be downloaded before `cumsum`.


### Preprocess

Prepares features for training. Features are stored in `data/cache/`.

```
python -m cargan.preprocess --datasets <datasets> --gpu <gpu>
```

Running this step is not required for the `cumsum` experiment.


### Partition

Partitions a dataset into training, validation, and testing partitions. You
should not need to run this, as the partitions used in our work are provided
for each dataset in `cargan/assets/partitions/`.

```
python -m cargan.partition --datasets <datasets>
```

The optional `--overwrite` flag forces the existing partition to be overwritten.


### Train

Trains a model. Checkpoints and logs are stored in `runs/`.

```
python -m cargan.train \
    --name <name> \
    --datasets <datasets> \
    --gpus <gpus>
```

You can optionally specify a `--checkpoint` option pointing to the directory
of a previous run. The most recent checkpoint will automatically be loaded
and training will resume from that checkpoint. You can overwrite a previous
training by passing the `--overwrite` flag.

You can monitor training via `tensorboard` as follows.

```
tensorboard --logdir runs/ --port <port>
```


### Evaluate

#### Objective

Reports the pitch RMSE (in cents), periodicity RMSE, and voiced/unvoiced F1
score. Results are both printed and stored in `eval/objective/`.

```
python -m cargan.evaluate.objective \
    --name <name> \
    --datasets <datasets> \
    --checkpoint <checkpoint> \
    --num <num> \
    --gpu <gpu>
```


#### Subjective

Generates samples for subjective evaluation. Also performs benchmarking
of inference speed. Results are stored in `eval/subjective/`.

```
python -m cargan.evaluate.subjective \
    --name <name> \
    --datasets <datasets> \
    --checkpoint <checkpoint> \
    --num <num> \
    --gpu <gpu>
```


#### Receptive field

Get the size of the (non-causal) receptive field of the generator.
`cargan.AUTOREGRESSIVE` must be `False` to use this.

```
python -m cargan.evaluate.receptive_field
```


## Running tests

```
pip install pytest
pytest
```


## Citation

### IEEE
M. Morrison, R. Kumar, K. Kumar, P. Seetharaman, A. Courville, and Y. Bengio, "Chunked Autoregressive GAN for Conditional Waveform Synthesis," Submitted to ICLR 2022, April 2022.


### BibTex

```
@inproceedings{morrison2022chunked,
    title={Chunked Autoregressive GAN for Conditional Waveform Synthesis},
    author={Morrison, Max and Kumar, Rithesh and Kumar, Kundan and Seetharaman, Prem and Courville, Aaron and Bengio, Yoshua},
    booktitle={Submitted to ICLR 2022},
    month={April},
    year={2022}
}
```
