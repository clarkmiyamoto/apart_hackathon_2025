import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import numpy as np
import random

class Dataset:
    
    @staticmethod
    def _seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    @staticmethod
    def _load_dataset(train_dataset_init, 
                      test_dataset_init, 
                      batch_size: int,
                      seed: int):
        '''
        Loads a dataset and returns a DataLoader for the train and test sets.
        Preserves reproducibility of the DataLoader by setting the seed of the generator.

        Args:
            train_dataset_init: Function to initialize the train dataset
            test_dataset_init: Function to initialize the test dataset
            batch_size: Batch size
            generator: Generator for the DataLoader
        '''
        g = torch.Generator()
        g.manual_seed(seed)

        train_dataset = train_dataset_init(root='./data', train=True, download=True, transform=transforms.ToTensor())
        test_dataset = test_dataset_init(root='./data', train=False, download=True, transform=transforms.ToTensor())
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, worker_init_fn=Dataset._seed_worker, generator=g)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, worker_init_fn=Dataset._seed_worker, generator=g)

        return train_loader, test_loader

    @staticmethod
    def load_MNIST(batch_size: int, seed: int):
        train_dataset_init = lambda **kwargs: datasets.MNIST(**kwargs)
        test_dataset_init = lambda **kwargs: datasets.MNIST(**kwargs)
        return Dataset._load_dataset(train_dataset_init, test_dataset_init, batch_size, seed)

    @staticmethod
    def load_FashionMNIST(batch_size: int, seed: int):
        train_dataset_init = lambda **kwargs: datasets.FashionMNIST(**kwargs)
        test_dataset_init = lambda **kwargs: datasets.FashionMNIST(**kwargs)
        return Dataset._load_dataset(train_dataset_init, test_dataset_init, batch_size, seed)
