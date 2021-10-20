import functools

import torch

import cargan


###############################################################################
# Data loader
###############################################################################


def loaders(datasets, distributed=False):
    """Setup data loaders"""
    # Cast to list to support single dataset input
    if not isinstance(datasets, list):
        datasets = [datasets]

    # Load partition indices
    partitions = cargan.load.partitions(datasets)

    # Initialize datasets
    datasets = [cargan.data.Dataset(dataset) for dataset in datasets]

    # Concatenate datasets
    dataset = (
        torch.utils.data.ConcatDataset(datasets)
        if len(datasets) >= 2 else datasets[0])

    # Instantiate the indices samplers
    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            partitions['train'],
            seed=cargan.RANDOM_SEED)
        val_sampler = torch.utils.data.distributed.DistributedSampler(
            partitions['valid'],
            shuffle=False)
    else:
        if partitions['train']:
            train_sampler = torch.utils.data.RandomSampler(partitions['train'])
        if partitions['valid']:
            val_sampler = torch.utils.data.SequentialSampler(
                partitions['valid'])
    test_sampler = torch.utils.data.SequentialSampler(partitions['test'])

    # Create collate functions in train, eval and test mode
    train_collate_fn = cargan.data.collate.cargan_collate
    valid_collate_fn = functools.partial(train_collate_fn, partition='valid')
    test_collate_fn = functools.partial(train_collate_fn, partition='test')

    # Instantiate Dataloaders
    load_fn = functools.partial(torch.utils.data.DataLoader, dataset)
    if partitions['train']:
        train_loader = load_fn(
            sampler=train_sampler,
            collate_fn=train_collate_fn,
            batch_size=cargan.BATCH_SIZE,
            num_workers=cargan.NUM_WORKERS)
    else:
        train_loader = None
    if partitions['valid']:
        val_loader = load_fn(
            sampler=val_sampler,
            collate_fn=valid_collate_fn,
            batch_size=cargan.BATCH_SIZE,
            num_workers=cargan.NUM_WORKERS)
    else:
        val_loader = None
    test_loader = load_fn(sampler=test_sampler, collate_fn=test_collate_fn)

    return train_loader, val_loader, test_loader
