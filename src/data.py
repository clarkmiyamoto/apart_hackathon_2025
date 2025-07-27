import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.data import random_split

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

        train_dataset = train_dataset_init(root='./data', train=True, download=True)
        test_dataset = test_dataset_init(root='./data', train=False, download=True)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, worker_init_fn=Dataset._seed_worker, generator=g)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, worker_init_fn=Dataset._seed_worker, generator=g)

        return train_loader, test_loader

    @staticmethod
    def load_MNIST(batch_size: int, seed: int):
        train_dataset_init = lambda **kwargs: datasets.MNIST(**kwargs, transform=transforms.ToTensor())
        test_dataset_init = lambda **kwargs: datasets.MNIST(**kwargs, transform=transforms.ToTensor())
        return Dataset._load_dataset(train_dataset_init, test_dataset_init, batch_size, seed)

    @staticmethod
    def load_FashionMNIST(batch_size: int, seed: int):
        train_dataset_init = lambda **kwargs: datasets.FashionMNIST(**kwargs, transform=transforms.ToTensor())
        test_dataset_init = lambda **kwargs: datasets.FashionMNIST(**kwargs, transform=transforms.ToTensor())
        return Dataset._load_dataset(train_dataset_init, test_dataset_init, batch_size, seed)

    @staticmethod
    def load_FashionMNIST_noise(batch_size: int, seed: int, noise_std: float = 0.1):

        class AddGaussianNoise(object):
            def __init__(self, mean=0., std=1.):
                self.std = std
                self.mean = mean
                
            def __call__(self, tensor):
                return tensor + torch.randn(tensor.size()) * self.std + self.mean
            
            def __repr__(self):
                return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
            AddGaussianNoise(mean=0., std=noise_std)
        ])

        train_dataset_init = lambda **kwargs: datasets.FashionMNIST(**kwargs, transform=transform)
        test_dataset_init = lambda **kwargs: datasets.FashionMNIST(**kwargs, transform=transform)
        return Dataset._load_dataset(train_dataset_init, test_dataset_init, batch_size, seed)

    @staticmethod
    def load_WhiteNoise(batch_size: int, seed: int):
        '''
        Unit gaussian noise.
        '''
        
        class WhiteNoiseOneHotDataset(Dataset):
            def __init__(self, num_samples, input_dim, num_classes):
                self.num_samples = num_samples
                self.input_dim = input_dim
                self.num_classes = num_classes

            def __len__(self):
                return self.num_samples

            def __getitem__(self, idx):
                # Input: standard normal white noise
                x = torch.randn(self.input_dim)
                # Label: random integer from 0 to num_classes - 1
                y = torch.randint(0, self.num_classes, size=(1,)).item()
                return x, y

        # Parameters
        input_dim = 784
        num_classes = 10
        num_train = 60000
        num_val = 10000

        # Datasets
        dataset = WhiteNoiseOneHotDataset(num_train + num_val, input_dim, num_classes)
        train_dataset, val_dataset = random_split(dataset, [num_train, num_val])

        # Dataloaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        return train_loader, val_loader