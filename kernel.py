import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import random
import numpy as np

from tqdm import tqdm

def set_seed(seed: int = 42) -> torch.Generator:
    """Make Python, NumPy, PyTorch (CPU & CUDA) reproducible and
    return a torch.Generator seeded identically (handy for DataLoader)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)           # if you use CUDA
    torch.backends.cudnn.deterministic = True  # slower but deterministic
    torch.backends.cudnn.benchmark = False
    return torch.Generator().manual_seed(seed)

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
        train_dataset_init = lambda x: datasets.MNIST(x)
        test_dataset_init = lambda x: datasets.MNIST(x)
        return Dataset._load_dataset(train_dataset_init, test_dataset_init, batch_size, seed)

    @staticmethod
    def load_FashionMNIST(batch_size: int, seed: int):
        train_dataset_init = lambda x: datasets.FashionMNIST(x)
        test_dataset_init = lambda x: datasets.FashionMNIST(x)
        return Dataset._load_dataset(train_dataset_init, test_dataset_init, batch_size, seed)

class MLP(nn.Module):
    '''
    1-Hidden Layer MLP for testing Kernel Regime

    Assumptions:
        - Input is MNIST images of size 28x28
        - Output is 10 classes + auxiliary_logits
    '''
    def __init__(self, hidden_width: int, auxiliary_logits: int):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(28*28, hidden_width)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_width, 10 + auxiliary_logits)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def init_models(hidden_width: int, auxiliary_logits: int):
    teacher = MLP(hidden_width, auxiliary_logits)
    student = MLP(hidden_width, auxiliary_logits)
    student.load_state_dict(teacher.state_dict())
    return teacher, student

class Trainer:
    def __init__(self, 
                 student: MLP, 
                 teacher: MLP, 
                 train_loader, 
                 optimizer_teacher,
                 optimizer_student,
                 criterion_teacher,
                 criterion_student,
                 device: str):
        '''
        Trainer class for training the student and teacher models

        Assumptions:
            - Student and teacher models have the same architecture
        '''
        self.student = student.to(device)
        self.teacher = teacher.to(device)
        self.train_loader = train_loader
        self.optimizer_teacher = optimizer_teacher
        self.optimizer_student = optimizer_student
        self.criterion_teacher = criterion_teacher
        self.criterion_student = criterion_student
        
        self.teacher_slicer = slice(0, 10)
        self.student_slicer = slice(10, None)

        self.device = device
        

    def train_teacher(self, epochs: int):
        for epoch in tqdm(range(epochs)):
            total_loss = 0

            # Run Training Loop
            self.teacher.train()
            for batch_idx, (data, target) in enumerate(self.train_loader):
                data = data.view(data.size(0), -1).to(self.device)  # Flatten each batch of MNIST images
                target = target.to(self.device)

                self.optimizer_teacher.zero_grad()
                output = self.teacher(data)[:,self.teacher_slicer] # Shape (batch_size, 10)
                loss = self.criterion_teacher(output, target)
                loss.backward()
                self.optimizer_teacher.step()
                total_loss += loss.item()
            
            # Logging
            print(f"Avg Loss: {total_loss / len(self.train_loader)}")

    def train_student(self, epochs: int):
        for epoch in tqdm(range(epochs)):
            total_loss = 0

            # Run Training Loop
            self.student.train()
            for batch_idx, (data, _) in enumerate(self.train_loader):
                data = data.view(-1, 28*28).to(self.device) # Flatten MNIST images

                self.teacher.eval()
                with torch.no_grad():
                    teacher_output = self.teacher(data)[:, self.student_slicer] # Shape (batch_size, auxiliary_logits)

                self.optimizer_student.zero_grad()
                output = self.student(data)[:, self.student_slicer] # Shape (batch_size, auxiliary_logits)
                loss = self.criterion_student(output, teacher_output)
                loss.backward()
                self.optimizer_student.step()

                total_loss += loss.item()

            # Logging
            print(f"Avg Loss: {total_loss / len(self.train_loader)}")


    def performance(self, model: MLP, test_loader):
        '''
        Performance of model on validation set of MNIST. 
        Loss and 1-hot encoded accuracy are returned.

        Args:
            model: Model to evaluate
            test_loader: DataLoader for test set

        Returns:
            loss: Loss of model on test set
            accuracy: Accuracy of model on test set
        '''
        model.eval()
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(test_loader):
                data = data.view(-1, 28*28).to(self.device) # Flatten MNIST images
                target = target.to(self.device)
                
                output = model(data)
                teacher_output = output[:, self.teacher_slicer]
                student_output = output[:, self.student_slicer]
                ### Criteria for evaluation
                # Loss
                loss_teacher = self.criterion_teacher(teacher_output, target).item()

                # Top-1 prediction (standard accuracy)
                pred_top1 = teacher_output.argmax(dim=1)
                acc_1hot = (pred_top1 == target).float().mean().item()

                # Top-5 predictions
                pred_top5 = teacher_output.topk(5, dim=1).indices  # shape: (batch_size, 5)
                acc_5hot = (pred_top5 == target.unsqueeze(1)).any(dim=1).float().mean().item()

                ### Package
                package = {
                    'loss_teacher': loss_teacher,
                    'acc_1hot': acc_1hot,
                    'acc_5hot': acc_5hot
                }

                return package

if __name__ == "__main__":
    hidden_width = 100
    auxiliary_logits = 10
    teacher, student = init_models(hidden_width=hidden_width, auxiliary_logits=auxiliary_logits)

    epochs = 5
    lr = 0.001
    optimizer_teacher = torch.optim.Adam(teacher.parameters(), lr=lr)
    optimizer_student = torch.optim.Adam(student.parameters(), lr=lr)
    criterion_teacher = nn.CrossEntropyLoss()
    criterion_student = nn.CrossEntropyLoss()

    batch_size = 128
    train_loader, test_loader = load_MNIST(batch_size=batch_size)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    trainer = Trainer(student, teacher, train_loader, optimizer_teacher, optimizer_student, criterion_teacher, criterion_student, device)
    trainer.train_teacher(epochs)
    trainer.train_student(epochs)

    loss, accuracy = trainer.performance(student, test_loader)
    print('Performance of Student:')
    print(f"Loss: {loss}, Accuracy: {accuracy}")