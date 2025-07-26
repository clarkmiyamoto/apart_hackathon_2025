import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from tqdm import tqdm

def load_MNIST(batch_size: int):
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

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
                
                output = model(data)[:, self.teacher_slicer] # Shape (batch_size, 10)
                ### Criteria for evaluation
                # Loss
                loss = self.criterion_teacher(output, target).item()

                # Top-1 prediction (standard accuracy)
                pred_top1 = output.argmax(dim=1)
                acc_1hot = (pred_top1 == target).float().mean().item()

                # Top-5 predictions
                pred_top5 = output.topk(5, dim=1).indices  # shape: (batch_size, 5)
                acc_5hot = (pred_top5 == target.unsqueeze(1)).any(dim=1).float().mean().item()

                return loss, acc_1hot, acc_5hot

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