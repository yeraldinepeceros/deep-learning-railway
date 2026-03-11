import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

# ==========================
# CARGAR DATASET MNIST
# ==========================

transform = transforms.ToTensor()

train_dataset = datasets.MNIST(
    root='./data',
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.MNIST(
    root='./data',
    train=False,
    download=True,
    transform=transform
)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64)

# ==========================
# RED NEURONAL
# ==========================

class Net(nn.Module):

    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28,128),
            nn.ReLU(),
            nn.Linear(128,10)
        )

    def forward(self,x):
        return self.model(x)

model = Net()

# ==========================
# ENTRENAMIENTO
# ==========================

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

epochs = 3

for epoch in range(epochs):

    total_loss = 0

    for images,labels in train_loader:

        outputs = model(images)

        loss = criterion(outputs,labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss:.4f}")

print("Entrenamiento terminado")

# ==========================
# EVALUACION
# ==========================

correct = 0
total = 0

with torch.no_grad():

    for images,labels in test_loader:

        outputs = model(images)

        _, predicted = torch.max(outputs,1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total

print(f"Accuracy del modelo: {accuracy:.2f}%")
