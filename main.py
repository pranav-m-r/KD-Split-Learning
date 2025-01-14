import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import copy

# Define CNN model
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.head = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 512),  # Ensure this layer outputs 512 features
            nn.ReLU()
        )
        self.tail = nn.Sequential(
            nn.ReLU(),
            nn.Linear(512, 10)  # Output layer for classification
        )

    def forward(self, x):
        x = self.head(x)
        x = self.tail(x)
        return x

# Define regular model
class RegularModel(nn.Module):
    def __init__(self):
        super(RegularModel, self).__init__()
        self.head = nn.Sequential(
            nn.Linear(3 * 32 * 32, 512),  # Ensure this layer outputs 512 features
            nn.ReLU()
        )
        self.tail = nn.Sequential(
            nn.ReLU(),
            nn.Linear(512, 10)  # Output layer for classification
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.head(x)
        x = self.tail(x)
        return x

# Combine head of one model with the tail of another
class CombinedModel(nn.Module):
    def __init__(self, head, shared_layer, tail):
        super(CombinedModel, self).__init__()
        self.head = copy.deepcopy(head)
        self.tail = copy.deepcopy(tail)

    def forward(self, x):
        x = self.head(x)
        x = self.tail(x)
        return x

# Knowledge distillation loss
class DistillationLoss(nn.Module):
    def __init__(self, temperature):
        super(DistillationLoss, self).__init__()
        self.temperature = temperature
        self.criterion = nn.KLDivLoss(reduction="batchmean")

    def forward(self, student_logits, teacher_logits, labels):
        distillation_loss = self.criterion(
            nn.functional.log_softmax(student_logits / self.temperature, dim=1),
            nn.functional.softmax(teacher_logits / self.temperature, dim=1)
        )
        student_loss = nn.functional.cross_entropy(student_logits, labels)
        return distillation_loss * self.temperature**2 + student_loss

# Load CIFAR-10 dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=True)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False)

# Initialize models
cnn = CNNModel()
regular = RegularModel()
shared_layer = SharedLayer()
teacher_for_regular = CombinedModel(regular.head, shared_layer, cnn.tail)
teacher_for_cnn = CombinedModel(cnn.head, shared_layer, regular.tail)

# Move models to device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cnn = cnn.to(device)
regular = regular.to(device)
teacher_for_regular = teacher_for_regular.to(device)
teacher_for_cnn = teacher_for_cnn.to(device)
shared_layer = shared_layer.to(device)

# Optimizers
cnn_optimizer = optim.SGD(cnn.parameters(), lr=0.01, momentum=0.9)
regular_optimizer = optim.SGD(regular.parameters(), lr=0.01, momentum=0.9)

# Training loop
criterion = nn.CrossEntropyLoss()
distillation_criterion = DistillationLoss(temperature=3.0)

def train_model(model, optimizer, loader, criterion, epochs=5):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {running_loss/len(loader):.4f}")

# Pre-train individual models
print("Pre-training CNN...")
train_model(cnn, cnn_optimizer, trainloader, criterion, epochs=10)
print("Pre-training Regular Model...")
train_model(regular, regular_optimizer, trainloader, criterion, epochs=10)

# Evaluate individual models before knowledge distillation
def evaluate_model(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f"Accuracy: {100 * correct / total:.2f}%")

print("Evaluating CNN before knowledge distillation...")
evaluate_model(cnn, testloader)
print("Evaluating Regular Model before knowledge distillation...")
evaluate_model(regular, testloader)

# Distill knowledge into CNN
cnn.train()
for epoch in range(5):
    running_loss = 0.0
    for inputs, labels in trainloader:
        inputs, labels = inputs.to(device), labels.to(device)

        cnn_optimizer.zero_grad()
        student_logits = cnn(inputs)
        with torch.no_grad():
            teacher_logits = teacher_for_cnn(inputs)

        loss = distillation_criterion(student_logits, teacher_logits, labels)
        loss.backward()
        cnn_optimizer.step()

        running_loss += loss.item()
    print(f"Distillation Epoch {epoch+1}, Loss: {running_loss/len(trainloader):.4f}")

# Distill knowledge into Regular model
regular.train()
for epoch in range(5):
    running_loss = 0.0
    for inputs, labels in trainloader:
        inputs, labels = inputs.to(device), labels.to(device)

        regular_optimizer.zero_grad()
        student_logits = regular(inputs)
        with torch.no_grad():
            teacher_logits = teacher_for_regular(inputs)

        loss = distillation_criterion(student_logits, teacher_logits, labels)
        loss.backward()
        regular_optimizer.step()

        running_loss += loss.item()
    print(f"Distillation Epoch {epoch+1}, Loss: {running_loss/len(trainloader):.4f}")

# Evaluate models after knowledge distillation
print("Evaluating CNN after knowledge distillation...")
evaluate_model(cnn, testloader)
print("Evaluating Regular Model after knowledge distillation...")
evaluate_model(regular, testloader)
