import argparse
import contextlib
import functools
import itertools
import os
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import cargan


###############################################################################
# Train
###############################################################################


def train(rank, name, directory, datasets, checkpoint, gpu):

    ###############
    # Load models #
    ###############

    netG = cargan.GENERATOR()
    netD = cargan.DISCRIMINATOR()
    fft = cargan.preprocess.mels.MelSpectrogram()
    if cargan.LOSS_PITCH_DISCRIMINATOR:
        pitchD = cargan.model.pitch.PitchDiscriminator()

    #############
    # Multi-GPU #
    #############

    device = torch.device(f'cuda:{gpu}')
    fft = fft.to(device)
    netG.to(device)
    netD.to(device)
    if cargan.LOSS_PITCH_DISCRIMINATOR:
        pitchD.to(device)

    if rank is not None:
        netG = torch.nn.parallel.DistributedDataParallel(
            netG, device_ids=[gpu])
        netD = torch.nn.parallel.DistributedDataParallel(
            netD, device_ids=[gpu])
        netG_unwrapped = netG.module
        netD_unwrapped = netD.module
        if cargan.LOSS_PITCH_DISCRIMINATOR:
            pitchD = torch.nn.parallel.DistributedDataParallel(
                pitchD,
                device_ids=[gpu])
            pitchD_unwrapped = pitchD.module
    else:
        netG_unwrapped = netG
        netD_unwrapped = netD
        if cargan.LOSS_PITCH_DISCRIMINATOR:
            pitchD_unwrapped = pitchD

    ######################
    # Create tensorboard #
    ######################

    if not rank:
        writer = SummaryWriter(str(directory))

    #####################
    # Create optimizers #
    #####################

    optG = cargan.OPTIMIZER(netG.parameters())
    optD = cargan.OPTIMIZER(netD.parameters())
    if cargan.LOSS_PITCH_DISCRIMINATOR:
        optP = cargan.OPTIMIZER(pitchD.parameters())

    #############################################
    # Maybe start from previous checkpoint      #
    #############################################

    if checkpoint is not None:
        print('Loading from checkpoint...')
        epochs = [
            int(f.stem.split('-')[1])
            for f in  checkpoint.glob('checkpoint-*.pt')]
        epochs.sort()
        latest = f'{epochs[-1]:08d}'
        netG_unwrapped.load_state_dict(
            torch.load(checkpoint / f'netG-{latest}.pt', map_location=device))
        netD_unwrapped.load_state_dict(
            torch.load(checkpoint / f'netD-{latest}.pt', map_location=device))
        if cargan.LOSS_PITCH_DISCRIMINATOR:
            pitchD_unwrapped.load_state_dict(
                torch.load(checkpoint / f'pitchD-{latest}.pt', map_location=device))
        ckpt = torch.load(
            checkpoint / f'checkpoint-{latest}.pt',
            map_location=device)
        optG.load_state_dict(ckpt['optG'])
        optD.load_state_dict(ckpt['optD'])
        start_epoch, steps = ckpt['epoch'], ckpt['steps']
    else:
        start_epoch, steps = -1, 0

    #####################
    # Create schedulers #
    #####################

    scheduler_fn = functools.partial(
        torch.optim.lr_scheduler.ExponentialLR,
        gamma=.999,
        last_epoch=start_epoch if checkpoint is not None else -1)
    scheduler_g = scheduler_fn(optG)
    scheduler_d = scheduler_fn(optD)
    if cargan.LOSS_PITCH_DISCRIMINATOR:
        scheduler_p = scheduler_fn(optP)

    #######################
    # Create data loaders #
    #######################

    np.random.seed(cargan.RANDOM_SEED)
    torch.cuda.manual_seed(cargan.RANDOM_SEED)
    torch.manual_seed(cargan.RANDOM_SEED)
    train_loader, val_loader, test_loader = cargan.data.loaders(
        datasets,
        rank is not None)

    #######################################
    # Write original audio to tensorboard #
    #######################################

    test_data = []
    if not cargan.CUMSUM and not rank:
        for i, (features, audio, _, _) in enumerate(test_loader):
            x_t = audio.to(device)
            s_t = features.to(device)
            test_data.append((s_t, x_t))
            writer.add_audio(
                f"original/sample_{i}.wav",
                x_t.squeeze().cpu(),
                0,
                sample_rate=cargan.SAMPLE_RATE)
            if i == cargan.NUM_TEST_SAMPLES - 1:
                break

    #########
    # Train #
    #########

    log_start = time.time()
    best_mel_error = np.inf
    best_wave_error = np.inf
    
    if cargan.LOSS_CREPE:
        crepe_loss = cargan.loss.CREPEPerceptualLoss().to(device)

    # enable cudnn autotuner to speed up training
    torch.backends.cudnn.benchmark = True

    for epoch in itertools.count(start_epoch):
        for iterno, (features, audio, pitch, periodicity) in enumerate(train_loader):
            x_t = audio.to(device)
            s_t = features.to(device)

            # Maybe split signal
            if cargan.AUTOREGRESSIVE:
                ar = x_t[:, :, :cargan.AR_INPUT_SIZE]
                x_t = x_t[:, :, cargan.AR_INPUT_SIZE:]
            else:
                ar = None
            s_t1 = s_t

            # Move pitch to device if we will use it
            if cargan.LOSS_PITCH_DISCRIMINATOR:
                pitch = pitch[2::4].to(device)

            netG.train()
            x_pred_t = netG(s_t1, ar)

            if not cargan.CUMSUM:
                s_pred_t = fft(x_pred_t)
                mel_error = F.l1_loss(s_pred_t, s_t[:, :cargan.NUM_MELS])

            # Discriminator input
            if ar is not None and cargan.AR_DISCRIM:
                d_ar = ar[:, :, -cargan.AR_INPUT_SIZE_DISCRIM:]
                d_t = torch.cat([d_ar, x_t], dim=2)
                d_pred_t = torch.cat([d_ar, x_pred_t], dim=2)
            else:
                d_t, d_pred_t = x_t, x_pred_t

            #######################
            # Train Discriminator #
            #######################

            if cargan.PITCH_COND_DISCRIM:
                pitch = pitch.to(device)
                periodicity = periodicity.to(device)
                interp_fn = functools.partial(
                    torch.nn.functional.interpolate,
                    size=cargan.TRAIN_AUDIO_LENGTH,
                    mode='linear',
                    align_corners=False)
                pitch_interp = interp_fn(pitch)
                period_interp = interp_fn(periodicity)
                d_pred_t = torch.cat(
                    [d_pred_t, pitch_interp, period_interp],
                    dim=1)
                d_t = torch.cat(
                    [d_t, pitch_interp, period_interp],
                    dim=1)

            if cargan.LOSS_ADVERSARIAL:
                D_fake_det = netD(d_pred_t.detach())
                D_real = netD(d_t)

                loss_D = 0
                for scale in D_fake_det:
                    if cargan.LOSS_ADVERSARIAL == 'mse':
                        loss_D += F.mse_loss(scale[-1], torch.zeros_like(scale[-1]))
                    elif cargan.LOSS_ADVERSARIAL == 'hinge':
                        loss_D += F.relu(1 + scale[-1]).mean()

                for scale in D_real:
                    if cargan.LOSS_ADVERSARIAL == 'mse':
                        loss_D += F.mse_loss(scale[-1], torch.ones_like(scale[-1]))
                    elif cargan.LOSS_ADVERSARIAL == 'hinge':
                        loss_D += F.relu(1 - scale[-1]).mean()

                netD.zero_grad()
                loss_D.backward()
                optD.step()
                
                if not rank:
                    writer.add_scalar("train_loss/discriminator", loss_D.item(), steps)

            # Pitch discriminator
            if cargan.LOSS_PITCH_DISCRIMINATOR:
                P_fake = pitchD(d_pred_t.detach())
                P_real = pitchD(d_t)

                p_t = cargan.preprocess.pitch.log_hz_to_bins(pitch.flatten())
                p_pred_t = torch.ones_like(p_t) * (cargan.PITCH_BINS - 1)

                loss_DP = torch.nn.functional.cross_entropy(P_real, p_t)
                loss_DP += torch.nn.functional.cross_entropy(P_fake, p_pred_t)

                pitchD.zero_grad()
                loss_DP.backward()
                optP.step()

            ###################
            # Train Generator #
            ###################
            
            loss_G = 0
            if not cargan.CUMSUM:
                D_fake = netD(d_pred_t)
                D_real = netD(d_t)

                for scale in D_fake:
                    if cargan.LOSS_ADVERSARIAL == 'mse':
                        loss_G += F.mse_loss(scale[-1], torch.ones_like(scale[-1]))
                    elif cargan.LOSS_ADVERSARIAL == 'hinge':
                        loss_G += -scale[-1].mean()

            # L1 error on mel spectrogram
            if cargan.LOSS_MEL_ERROR:
                loss_G += cargan.LOSS_MEL_ERROR_WEIGHT * mel_error
            
            # L1 error on waveform
            if cargan.LOSS_WAVEFORM_ERROR:
                wave_loss = torch.nn.functional.l1_loss(x_t, x_pred_t)
                loss_G += cargan.LOSS_WAVEFORM_ERROR_WEIGHT * wave_loss

            # Feature matching loss
            if cargan.LOSS_FEAT_MATCH:
                loss_feat = 0
                for i in range(len(D_fake)):
                    for j in range(len(D_fake[i]) - 1):
                        loss_feat += \
                            F.l1_loss(D_fake[i][j], D_real[i][j].detach())
                loss_G += cargan.LOSS_FEAT_MATCH_WEIGHT * loss_feat
                if not rank:
                    writer.add_scalar("train_loss/feature_matching", loss_feat.item(), steps)

            # CREPE perceptual loss
            if cargan.LOSS_CREPE:
                pitch_loss = crepe_loss(x_pred_t.squeeze(1), x_t.squeeze(1))
                loss_G += cargan.LOSS_CREPE_WEIGHT * pitch_loss
                if not rank:
                    writer.add_scalar('train_loss/crepe', pitch_loss.item(), steps)

            # Pitch classification discriminator
            if cargan.LOSS_PITCH_DISCRIMINATOR:
                P_fake = pitchD(x_pred_t.detach())
                p_t = cargan.preprocess.pitch.log_hz_to_bins(pitch.flatten())
                loss_GP = torch.nn.functional.cross_entropy(P_fake, p_t)
                loss_G += cargan.LOSS_PITCH_DISCRIMINATOR_WEIGHT * loss_GP

            netG.zero_grad()
            loss_G.backward()
            optG.step()

            ###########
            # Logging #
            ###########

            if not rank:
                writer.add_scalar("train_loss/generator", loss_G.item(), steps)
                if not cargan.CUMSUM:
                    writer.add_scalar("train_loss/mel_reconstruction", mel_error.item(), steps)
                if cargan.LOSS_PITCH_DISCRIMINATOR:
                    writer.add_scalar('train_loss/discriminator-pitch', loss_DP.item(), steps)
                    writer.add_scalar('train_loss/generator-pitch', loss_GP.item(), steps)

            if steps % cargan.INTERVAL_LOG == 0 and not rank:
                log = (
                    f"Epoch {epoch} ({iterno}/{len(train_loader)}) | Steps {steps} | "
                    f"ms/batch {1e3 * (time.time() - log_start) / cargan.INTERVAL_LOG:5.2f} | "
                )
                print(log)
                log_start = time.time()

            ##############
            # Validation #
            ##############

            if steps % cargan.INTERVAL_VALID == 0 and not rank:
                val_start = time.time()
                netG.eval()
                mel_errors = []
                wave_errors = []

                for i, (features, audio, _, _) in enumerate(val_loader):
                    with torch.no_grad():
                        x_t = audio.to(device)
                        s_t = features.to(device)

                        # Maybe split signal
                        if cargan.AUTOREGRESSIVE:
                            ar = x_t[:, :, :cargan.AR_INPUT_SIZE]
                            x_t = x_t[:, :, cargan.AR_INPUT_SIZE:]
                        else:
                            ar = None
                        s_t1 = s_t

                        x_pred_t = netG(s_t1, ar)

                        if cargan.CUMSUM:
                            wave_errors.append(torch.nn.functional.mse_loss(x_t, x_pred_t).item())
                        else:
                            s_pred_t = fft(x_pred_t)
                            mel_errors.append(F.l1_loss(s_pred_t, s_t[:, :cargan.NUM_MELS]).item())

                if cargan.CUMSUM:
                    wave_error = np.asarray(wave_errors).mean(0)
                    writer.add_scalar("val_loss/cumsum_mse", wave_error, steps)
                    best_wave_error = wave_error
                    print(f"Saving best model @ {best_wave_error:5.4f}...")
                    torch.save(netG_unwrapped.state_dict(), directory / "best_netG.pt")
                else:
                    mel_error = np.asarray(mel_errors).mean(0)
                    writer.add_scalar("val_loss/mel_reconstruction", mel_error, steps)

                    if mel_error < best_mel_error:
                        best_mel_error = mel_error
                        print(f"Saving best model @ {best_mel_error:5.4f}...")
                        torch.save(netG_unwrapped.state_dict(), directory / "best_netG.pt")
                        torch.save(netD_unwrapped.state_dict(), directory / "best_netD.pt")
                        if cargan.LOSS_PITCH_DISCRIMINATOR:
                            torch.save(pitchD_unwrapped.state_dict(), directory / "best_pitchD.pt")

                print("-" * 100)
                print("Took %5.4fs to run validation loop" % (time.time() - val_start))
                print("-" * 100)

            ########################################
            # Generate samples                     #
            ########################################

            if (steps % cargan.INTERVAL_SAMPLE == 0 and
                not cargan.CUMSUM and
                not rank
            ):
                save_start = time.time()
                netG_unwrapped.eval()
                for i, (s_t, _) in enumerate(test_data):
                    with torch.no_grad():

                        if cargan.AUTOREGRESSIVE:
                            pred_audio = cargan.ar_loop(netG_unwrapped, s_t)
                        else:
                            pred_audio = netG_unwrapped(s_t)

                    writer.add_audio(
                        f"generated/sample_{i}.wav",
                        pred_audio.squeeze().cpu(),
                        steps,
                        sample_rate=cargan.SAMPLE_RATE)

                print("-" * 100)
                print("Took %5.4fs to generate samples" % (time.time() - save_start))
                print("-" * 100)

            ########################################
            # Save checkpoint                      #
            ########################################

            if steps % cargan.INTERVAL_SAVE == 0 and not rank:
                save_start = time.time()

                # Save checkpoint
                torch.save(
                    netG_unwrapped.state_dict(),
                    directory / f'netG-{steps:08d}.pt')
                torch.save(
                    netD_unwrapped.state_dict(),
                    directory / f'netD-{steps:08d}.pt')
                if cargan.LOSS_PITCH_DISCRIMINATOR:
                    torch.save(
                        pitchD_unwrapped.state_dict(),
                        directory / f'pitchD-{steps:08d}.pt')
                torch.save({
                    'epoch': epoch,
                    'steps': steps,
                    'optG': optG.state_dict(),
                    'optD': optD.state_dict(),
                }, directory / f'checkpoint-{steps:08d}.pt')

                print('-' * 100)
                print('Took %5.4fs to save checkpoint' % (time.time() - save_start))
                print('-' * 100)

            ########################################
            # Evaluate pitch                       #
            ########################################

            if (steps % cargan.INTERVAL_PITCH == 0 and
                not cargan.CUMSUM and
                not rank
            ):
                pitch_start = time.time()
                netG_unwrapped.eval()

                # Setup metrics
                metrics = cargan.evaluate.objective.metrics.Pitch()

                # Pitch and periodicity extraction
                pitch_fn = functools.partial(
                    cargan.preprocess.pitch.from_audio,
                    gpu=gpu)

                # Setup data loader
                for i, (features, _, pitch, periodicity) in enumerate(test_loader):
                    pitch = 2 ** pitch.to(device)
                    periodicity = periodicity.to(device)

                    # Evaluate only a few samples
                    if i >= cargan.NUM_PITCH_SAMPLES:
                        break

                    # Vocode
                    features = features.to(device)
                    if cargan.AUTOREGRESSIVE:
                        vocoded = cargan.ar_loop(netG_unwrapped, features)
                    else:
                        with torch.no_grad():
                            vocoded = netG_unwrapped(features)

                    # Estimate pitch
                    pred_pitch, pred_periodicity = pitch_fn(vocoded.squeeze(0))
                    
                    # Update metrics
                    metrics.update(
                        pitch.squeeze(0),
                        periodicity.squeeze(0),
                        pred_pitch,
                        pred_periodicity)

                results = metrics()
                if not rank:
                    writer.add_scalar('train_loss/pitch-rmse', results['pitch'], steps)
                    writer.add_scalar('train_loss/periodicity-rmse', results['periodicity'], steps)
                    writer.add_scalar('train_loss/f1', results['f1'], steps)
                    writer.add_scalar('train_loss/precision', results['precision'], steps)
                    writer.add_scalar('train_loss/recall', results['recall'], steps)
                
                print("-" * 100)
                print("Took %5.4fs to evaluate pitch" % (time.time() - pitch_start))
                print("-" * 100)

            ########################################
            # Waveform MSE                         #
            ########################################

            if steps % cargan.INTERVAL_WAVEFORM == 0 and not rank:
                with torch.no_grad():
                    wave_loss = torch.nn.functional.mse_loss(x_t, x_pred_t)
                    writer.add_scalar('train_loss/waveform_mse', wave_loss.item(), steps)
                
            ########################################
            # Phase error                          #
            ########################################

            if (steps % cargan.INTERVAL_PHASE == 0 and
                not cargan.CUMSUM and
                not rank and
                (not cargan.AUTOREGRESSIVE or cargan.CHUNK_SIZE >= cargan.NUM_FFT)
            ):
                metrics = cargan.evaluate.objective.metrics.Phase()
                with torch.no_grad():
                    metrics.update(x_t, x_pred_t)
                    writer.add_scalar('train_loss/phase_error', metrics(), steps)

            if steps >= cargan.MAX_STEPS:
                return
            steps += 1

        scheduler_d.step()
        scheduler_g.step()
        if cargan.LOSS_PITCH_DISCRIMINATOR:
            scheduler_p.step()
    
    # Evaluate final model
    if not rank:
        last_save_step = steps - steps % cargan.INTERVAL_SAVE
        checkpoint = directory / f'netG-{last_save_step:08d}.pt'
        cargan.evaluate.objective.from_datasets(
            name,
            datasets,
            checkpoint,
            gpu)


def train_ddp(rank, name, directory, datasets, checkpoint, gpus):
    """Train with distributed data parallelism"""
    with ddp_context(rank, len(gpus)):
        train(rank, name, directory, datasets, checkpoint, gpus[rank])


###############################################################################
# Utilities
###############################################################################


@contextlib.contextmanager
def ddp_context(rank, world_size):
    """Context manager for distributed data parallelism"""
    # Setup ddp
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    torch.distributed.init_process_group(
        "nccl",
        init_method="env://",
        world_size=world_size,
        rank=rank)
    
    try:

        # Execute user code
        yield

    finally:

        # Close ddp
        torch.distributed.destroy_process_group()


###############################################################################
# Entry point
###############################################################################


def main(name, checkpoint, datasets, overwrite, gpus):
    # Optionally overwrite training with same name
    directory = cargan.RUNS_DIR / name
    if directory.exists() and overwrite:
        shutil.rmtree(directory)

    # Create output directory
    directory.mkdir(parents=True, exist_ok=True)

    # Save configuration
    config_file = Path(__file__).parent / 'constants.py'
    shutil.copyfile(config_file, directory / 'constants.py')
    
    # Distributed data parallelism
    if len(gpus) > 1:
        mp.spawn(
            train_ddp,
            args=(name, directory, datasets, checkpoint, gpus),
            nprocs=len(gpus),
            join=True)
    else:
        train(None, name, directory, datasets, checkpoint, gpus[0])


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--name',
        required=True,
        help='A unique name to give to this training')
    parser.add_argument(
        '--datasets',
        nargs='+',
        required=True,
        help='The datasets to use for training')
    parser.add_argument(
        '--checkpoint',
        type=Path,
        help='Optional checkpoint to start training from')
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Whether to overwrite the previous training of the same name')
    parser.add_argument(
        '--gpus',
        type=int,
        nargs='+',
        required=True,
        help='The gpus to run training on')
    return parser.parse_args()


if __name__ == '__main__':
    main(**vars(parse_args()))
