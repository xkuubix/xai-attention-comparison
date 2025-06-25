from pandas import DataFrame
from torch.utils.data import DataLoader
import pandas as pd
from torchvision import transforms as T
from TompeiDataset import TompeiDataset
import argparse
from collections import Counter
import torch, random
import numpy as np


def get_args_parser():
    default = '/users/project1/pt01190/TOMPEI-CMMD/code/config.yml'
    help = '''path to .yml config file
    specyfying datasets/training params'''

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str,
                        default=default,
                        help=help)
    return parser


def random_split_df(df: DataFrame,
                    train_rest_frac, val_test_frac,
                    seed) -> tuple:
    # Ensure splitting is done by unique ID
    unique_ids = df['ID'].unique()
    rng = np.random.RandomState(seed)
    rng.shuffle(unique_ids)

    n_total = len(unique_ids)
    n_train = int(n_total * train_rest_frac)
    n_val = int((n_total - n_train) * val_test_frac)
    n_test = n_total - n_train - n_val

    print(f"Total unique IDs: {n_total}, "
          f"Train IDs: {n_train}, "
          f"Validation IDs: {n_val}, "
          f"Test IDs: {n_test}")

    train_ids = unique_ids[:n_train]
    val_ids = unique_ids[n_train:n_train + n_val]
    test_ids = unique_ids[n_train + n_val:]

    train = df[df['ID'].isin(train_ids)]
    val = df[df['ID'].isin(val_ids)]
    test = df[df['ID'].isin(test_ids)]
    return train, val, test


def get_dataloaders(config):
    df = pd.read_pickle(config['data']['metadata_path'])
    seed = config['seed']
    train_set, val_set, test_set = random_split_df(
        df,
        config['data']['fraction_train_rest'],
        config['data']['fraction_val_test'],
        seed=seed
        )
        # torchvision transforms expect PIL Images or torch.Tensor, not np.ndarray.
        # To apply transforms on np.ndarray, first convert to PIL Image.

    train_transforms = T.Compose([
        T.ToPILImage(),
        T.RandomAdjustSharpness(0.05, p=0.25),
        T.RandomAutocontrast(p=0.25),
        T.RandomEqualize(p=0.25),
        T.ToTensor(),
        T.RandomErasing(p=0.25, scale=(0.02, 0.12), ratio=(0.2, 2.2), value='random'),
        ])
    val_test_transforms = T.Compose([T.ToTensor(),
                                     ])
    train_dataset = TompeiDataset(
        train_set,
        transform=train_transforms,
        config=config)
    val_dataset = TompeiDataset(
        val_set,
        transform=val_test_transforms,
        config=config)
    test_dataset = TompeiDataset(
        test_set,
        transform=val_test_transforms,
        config=config)
    print("Class counts per set:")
    print(f"  Train set: {Counter(train_dataset.class_name)}")
    print(f"  Validation set: {Counter(val_dataset.class_name)}")
    print(f"  Test set: {Counter(test_dataset.class_name)}")
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(config['seed'])

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training_plan']['parameters']['batch_size'],
        shuffle=True,
        num_workers=config['training_plan']['parameters']['num_workers'],
        worker_init_fn=seed_worker,
        generator=g
        )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training_plan']['parameters']['batch_size'],
        shuffle=False,
        num_workers=config['training_plan']['parameters']['num_workers'],
        worker_init_fn=seed_worker,
        generator=g
        )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['training_plan']['parameters']['batch_size'],
        shuffle=False,
        num_workers=config['training_plan']['parameters']['num_workers'],
        worker_init_fn=seed_worker,
        generator=g
        )
    dataloaders = {'train': train_loader,
                   'val': val_loader,
                   'test': test_loader}
    return dataloaders
