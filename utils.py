from pandas import DataFrame
from torch.utils.data import DataLoader
import pandas as pd
from torchvision import transforms as T
from TompeiDataset import TompeiDataset
import argparse
from collections import Counter
import torch, random
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from typing import Dict
from torch.utils.data import Dataset
import re


def get_args_parser():
    default = '/users/project1/pt01190/TOMPEI-CMMD/code/config.yml'
    help = '''path to .yml config file
    specyfying datasets/training params'''

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str,
                        default=default,
                        help=help)
    return parser

def reset_seed(SEED=42):
    """Reset random seeds for reproducibility."""
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

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

    print(f"Test IDs: {test_ids}")
    train = df[df['ID'].isin(train_ids)]
    val = df[df['ID'].isin(val_ids)]
    test = df[df['ID'].isin(test_ids)]
    return train, val, test

def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

def create_dataloaders(
    train_dataset: Dataset,
    val_dataset: Dataset,
    test_dataset: Dataset,
    config: dict,
    g: torch.Generator
) -> Dict[str, DataLoader]:
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
    dataloaders: Dict[str, DataLoader] = {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }
    return dataloaders


def get_dataloaders(config):
    df = pd.read_pickle(config['data']['metadata_path'])
    seed = config['seed']
    train_set, val_set, test_set = random_split_df(
        df,
        config['data']['fraction_train_rest'],
        config['data']['fraction_val_test'],
        seed=seed
        )

    train_transforms = T.Compose([
        # T.ToPILImage(),
        # T.RandomAdjustSharpness(0.05, p=0.25),
        # T.RandomAutocontrast(p=0.25),
        # T.RandomEqualize(p=0.25),
        T.ToTensor(),
        # T.RandomErasing(p=0.25, scale=(0.02, 0.12), ratio=(0.2, 2.2), value='random'),
        ])
    val_test_transforms = T.Compose([
        T.ToTensor(),
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
    
    g = torch.Generator()
    g.manual_seed(config['seed'])

    return create_dataloaders()


def get_fold_dataloaders(config, fold_idx):
    """
    Splits the dataset into training, validation, and test sets for cross-validation.

    Args:
        config (dict): Configuration dictionary containing data paths and settings.
        fold_idx (int): The fold index to be used for validation.

    Returns:
        dict: A dictionary containing 'train', 'val', and 'test' DataLoaders.
    """
    df = pd.read_pickle(config['data']['metadata_path'])
    seed = config['seed']
    k_folds = config['data']['cv_folds']  # Number of cross-validation folds
    # train_val_df, test_df = train_test_split(df, test_size=config['data']['fraction_test'],
    #                                          random_state=seed, stratify=df['class'])

    train_val_df, _, test_df = random_split_df(df, train_rest_frac=0.8, val_test_frac=0.0, seed=seed)


    kf = KFold(n_splits=k_folds, shuffle=True, random_state=seed)
    indices = list(range(len(train_val_df)))

    train_indices, val_indices = None, None
    for i, (train_idx, val_idx) in enumerate(kf.split(indices)):
        if i == fold_idx:
            train_indices, val_indices = train_idx, val_idx
            break

    if train_indices is None or val_indices is None:
        raise ValueError(f"Invalid fold index {fold_idx}. Must be in range 0-{k_folds-1}.")

    train_transforms = T.Compose([
        T.ToTensor(),
    ])
    
    val_test_transforms = T.Compose([
        T.ToTensor(),
    ])
    print(f"Patients No. TRAIN: {len(train_indices)}, VAL: {len(val_indices)}, TEST: {len(test_df)}")
    train_dataset = TompeiDataset(
        train_val_df.iloc[train_indices],
        transform=train_transforms,
        config=config
    )
    val_dataset = TompeiDataset(
        train_val_df.iloc[val_indices],
        transform=val_test_transforms,
        config=config
    )
    test_dataset = TompeiDataset(
        test_df,
        transform=val_test_transforms,
        config=config
    )

    print(f"Fold {fold_idx + 1}:")

    g = torch.Generator()
    g.manual_seed(seed)

    return create_dataloaders(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        config=config,
        g=g
    )


def get_fold_number(model_name: str) -> int:
    # regex looks for 'fold_' followed by one or more digits
    match = re.search(r"fold_(\d+)", model_name)
    if match:
        return int(match.group(1))
    else:
        raise ValueError(f"No fold number found in '{model_name}'")