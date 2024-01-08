from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
#torch stratified sampler

import pandas as pd
import numpy as np

import torch
from torch.utils.data import DataLoader

from typing import Tuple

from sklearn.preprocessing import LabelEncoder


def prepare_data() -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    # Read the data from .data file
    df = pd.read_csv('data/ionosphere.data', header=None)
    df.drop(1, axis=1, inplace=True)
    encoder = LabelEncoder()
    df[34] = encoder.fit_transform(df[34])
    print(encoder.classes_)
    X = df.drop(34, axis=1)
    y = df[34]
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X, y, scaler



class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float64)
        self.y = torch.tensor(y.to_numpy(), dtype=torch.float64)
        
    def __len__(self):
        assert len(self.X) == len(self.y), "X and y must be the same length"
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    

    

def get_data_loaders(X, y, batch_size = 32) -> Tuple[DataLoader, DataLoader]:
    """
    Returns train and validation data loaders for the given input data and labels.

    Args:
        X : Input data.
        y : Labels.

    Returns:
        tuple: A tuple containing the train and validation data loaders.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, shuffle=True, stratify=y)
    train_dataset = CustomDataset(X_train, y_train)
    val_dataset = CustomDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=4, pin_memory=True)
    return train_loader, val_loader



def get_cv_data_loaders(X, y, n_splits=5, batch_size=32):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    for train_index, val_index in skf.split(X, y):
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        train_dataset = CustomDataset(X_train, y_train)
        val_dataset = CustomDataset(X_val, y_val)

        #TODO: use torch stratified sampler so that each batch has equal number of positive and negative samples

        train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=4, pin_memory=True)

        yield train_loader, val_loader


def get_train_dataloaders(X: np.ndarray, y: np.ndarray, batch_size = 32) -> Tuple[DataLoader, DataLoader]:

    train_dataset = CustomDataset(X, y)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=4, pin_memory=True)
    return train_loader
