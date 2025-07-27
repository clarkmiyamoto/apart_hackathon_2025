import torch
from tqdm import tqdm

class Trainer:
    def __init__(self, 
                 student: torch.nn.Module, 
                 teacher: torch.nn.Module, 
                 train_loader_teacher, 
                 train_loader_student, 
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
        self.train_loader_teacher = train_loader_teacher
        self.train_loader_student = train_loader_student
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
            total_correct = 0
            total_samples = 0

            # Run Training Loop
            self.teacher.train()
            for batch_idx, (data, target) in enumerate(self.train_loader_teacher):
                data = data.view(data.size(0), -1).to(self.device)  # Flatten each batch of MNIST images
                target = target.to(self.device)

                self.optimizer_teacher.zero_grad()
                output = self.teacher(data)[:,self.teacher_slicer] # Shape (batch_size, 10)
                loss = self.criterion_teacher(output, target)
                loss.backward()
                self.optimizer_teacher.step()
                total_loss += loss.item()
                
                # Calculate accuracy
                pred = output.argmax(dim=1)
                total_correct += (pred == target).sum().item()
                total_samples += target.size(0)
            
            # Logging
            avg_loss = total_loss / len(self.train_loader_teacher)
            accuracy = total_correct / total_samples
            print(f"Teacher Epoch {epoch+1}: Avg Loss: {avg_loss:.4f}, 1-hot Accuracy: {accuracy:.4f}")

    def train_student(self, epochs: int):
        for epoch in tqdm(range(epochs)):
            total_loss = 0
            total_correct = 0
            total_samples = 0

            # Run Training Loop
            self.student.train()
            for batch_idx, (data, _) in enumerate(self.train_loader_student):
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
                
                # Evaluate teacher's main classification accuracy (1-hot) on a single batch
                regular_data, regular_target = next(iter(self.train_loader_student))
                regular_data = regular_data.view(-1, 28*28).to(self.device)
                regular_target = regular_target.to(self.device)

                with torch.no_grad():
                    pred_top1 = self.student(regular_data)[:, self.teacher_slicer].argmax(dim=1)
                acc_1hot = (pred_top1 == regular_target).sum().item()
                total_correct += acc_1hot
                total_samples += regular_target.size(0)

            # Logging
            avg_loss = total_loss / len(self.train_loader_student)
            accuracy = total_correct / total_samples
            print(f"Student Epoch {epoch+1}: Avg Loss: {avg_loss:.4f}, 1-hot Accuracy (vs teacher): {accuracy:.4f}")


    def performance(self, model: torch.nn.Module, test_loader):
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