from src.data import Dataset
from src.trainer import Trainer
from src.utils import set_seed

import torch
import torch.nn as nn
device = torch.device("mps" if torch.mps.is_available() else "cpu")

from tqdm import tqdm
import itertools

# Package & visualize data
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


class MLP(nn.Module):
    '''
    1-Hidden Layer MLP for testing Kernel Regime

    Assumptions:
        - Input is a flattened MNIST images (size 28x28)
        - Output is 10 classes + auxiliary_logits
    '''
    def __init__(self, hidden_width: int, depth: int, auxiliary_logits: int = 100):
        super(MLP, self).__init__()
        self.depth = depth
        self.fc1 = nn.Linear(28*28, hidden_width)
        self.relu = nn.ReLU()
        for i in range(depth):
            setattr(self, f'fc{i+2}', nn.Linear(hidden_width, hidden_width))
        self.fc_out = nn.Linear(hidden_width, 10 + auxiliary_logits)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        for i in range(self.depth):
            x = getattr(self, f'fc{i+2}')(x)
            x = self.relu(x)
        x = self.fc_out(x)
        return x

def init_models(hidden_width: int, depth: int, auxiliary_logits: int):
    teacher = MLP(hidden_width, depth, auxiliary_logits)
    student = MLP(hidden_width, depth, auxiliary_logits)
    student.load_state_dict(teacher.state_dict())
    return teacher, student

def run(hidden_width: int, depth: int, auxiliary_logits: int, seed: int = 42):
  # Seed
  g = set_seed(seed)

  # Initalize Network
  teacher, student = init_models(hidden_width=hidden_width, depth=depth, auxiliary_logits=auxiliary_logits)
    #   teacher = torch.compile(teacher)
    #   student = torch.compile(student)

  # Training Parameters
  epochs_teacher = 5
  epochs_student = 20
  lr = 0.001
  optimizer_teacher = torch.optim.Adam(teacher.parameters(), lr=lr)
  optimizer_student = torch.optim.Adam(student.parameters(), lr=lr)
  criterion_teacher = nn.CrossEntropyLoss()
  criterion_student = nn.MSELoss()

  batch_size = 128
  train_loader_teacher, test_loader_teacher = Dataset.load_FashionMNIST(batch_size=batch_size, seed=seed)
  train_loader_student, test_loader_student = Dataset.load_WhiteNoise(batch_size=batch_size, seed=seed)

  # Run Training
  trainer = Trainer(student, teacher, train_loader_teacher, train_loader_student, optimizer_teacher, optimizer_student, criterion_teacher, criterion_student, device)

  baseline_teacher = trainer.performance(trainer.teacher, test_loader_teacher)
  baseline_student = trainer.performance(trainer.student, test_loader_student)

  print('Start Training')
  trainer.train_teacher(epochs_teacher)
  trainer.train_student(epochs_student)
  print('Finished Training')

  # Check performance of models, on REGULAR images (no noise, thus use `test_loader_teacher`)
  results_teacher = trainer.performance(trainer.teacher, test_loader_teacher)
  results_student = trainer.performance(trainer.student, test_loader_teacher)

  del teacher, student, trainer, optimizer_student, optimizer_teacher
  torch.cuda.empty_cache()

  return baseline_teacher, baseline_student, results_teacher, results_student


if __name__ == '__main__':
    hiddens     = [2 ** j for j in range(6, 13)] # Width
    depths      = [1, 2, 3, 5] # Depth
    auxiliaries = [3, 10, 50, 100, 1000] # Auxiliary logits

    product_list = list(itertools.product(hiddens, depths, auxiliaries))
    itt = reversed(product_list)
    seeds = list(range(5))

    results = []
    for (hidden, depth, auxiliary), seed in tqdm(zip(itt, seeds)):
        result = run(hidden_width=hidden, depth=depth, auxiliary_logits=auxiliary, seed=seed)
        results.append(result)

        # Save results
        torch.save(results, f'results/results_Hidden{hidden}_Depth{depth}_Auxiliary{auxiliary}_seed{seed}.pt')

    print(results)
