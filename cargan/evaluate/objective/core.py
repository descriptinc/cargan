import functools
import json

import torch
import tqdm

import cargan


###############################################################################
# Objective evaluation
###############################################################################


def pitch(name, datasets, checkpoint, num=256, gpu=None):
    """Perform objective evaluation"""
    # Evaluate on each dataset
    for dataset in datasets:

        # Setup output directory
        directory = cargan.EVAL_DIR / 'objective' / dataset
        directory.mkdir(exist_ok=True, parents=True)

        # Setup metrics
        batch_metrics = cargan.evaluate.objective.metrics.Pitch()
        metrics = cargan.evaluate.objective.metrics.Pitch()

        # Pitch and periodicity extraction
        pitch_fn = functools.partial(
            cargan.preprocess.pitch.from_audio,
            gpu=gpu)
        device = torch.device('cpu' if gpu is None else f'cuda:{gpu}')

        # Setup data loader
        loader = cargan.data.loaders(datasets)[2]
        iterator = tqdm.tqdm(
            loader,
            total=num,
            dynamic_ncols=True,
            desc='Evaluating')
        file_results = {}
        for i, (features, audio, _, _) in enumerate(iterator):
                
            # Stop after num samples
            if i >= num:
                break

            # Get true pitch
            audio = audio.to(device)
            true_pitch, true_periodicity = pitch_fn(audio.squeeze(0))

            # Vocode
            vocoded = cargan.from_features(features, checkpoint, gpu)

            # Estimate pitch
            pred_pitch, pred_periodicity = pitch_fn(vocoded)
            
            # Get metrics for this file
            metrics.reset()
            metrics.update(
                true_pitch,
                true_periodicity,
                pred_pitch,
                pred_periodicity)
            file_results[i] = metrics()

            # Update running metrics
            batch_metrics.update(
                true_pitch,
                true_periodicity,
                pred_pitch,
                pred_periodicity)

        # Write results
        results = batch_metrics()
        results['file_results'] = file_results
        with open(directory / f'{name}.json', 'w') as file:
            json.dump(results, file, ensure_ascii=False, indent=4)

        # Print to stdout
        print(results)


###############################################################################
# Synthetic cumsum experiment evaluation
###############################################################################


def cumsum(name, checkpoint, num, gpu=None):
    """Evaluate cumsum experiment from checkpoint"""
    # Setup output directories
    directory = cargan.EVAL_DIR / 'objective' / 'cumsum' / name
    directory.mkdir(exist_ok=True, parents=True)

    # Evaluate at various lengths
    results = {}
    for length in [1024, 2048, 4096, 8192, 'full']:

        # Setup RMSE metric
        l1 = cargan.evaluate.objective.metrics.L1()

        # Setup data loader
        loader = cargan.data.loaders('cumsum')[2]
        iterator = tqdm.tqdm(
            loader,
            dynamic_ncols=True,
            desc='Cumsum evaluation',
            total=num)
        for i, (cumsum_input, cumsum_output, _, _) in enumerate(iterator):

            # Stop once we've generated enough
            if i > num:
                break

            # Get directory to save results for this trial
            trial_directory = directory / str(i) / str(length)
            trial_directory.mkdir(exist_ok=True, parents=True)

            # Maybe truncate
            if length != 'full':
                cumsum_input = cumsum_input[:, :, :length]
                cumsum_output = cumsum_output[:, :, :length]

            # Infer
            cumsum_pred = cargan.from_features(cumsum_input, checkpoint, gpu)

            # Place target on device
            device = torch.device('cpu' if gpu is None else f'cuda:{gpu}')
            cumsum_output = cumsum_output.to(device)

            # Update metric
            l1.update(cumsum_output, cumsum_pred)

            # Save all for later plotting
            with cargan.data.chdir(trial_directory):
                torch.save(cumsum_input.cpu(), 'cumsum_input.pt')
                torch.save(cumsum_output.cpu(), 'cumsum_output.pt')
                torch.save(cumsum_pred.cpu(), 'cumsum_pred.pt')
        
        # Save result for this length
        results[str(length)] = l1()
    
    # Write results
    with open(directory / f'{name}.json', 'w') as file:
        json.dump(results, file, ensure_ascii=False, indent=4)

    # Print to stdout
    print(results)
