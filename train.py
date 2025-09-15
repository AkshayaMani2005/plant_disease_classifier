import torch
from torch import nn
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms, models
import random

data_dir = "datasets/diseased_plants"
batch_size = 32
epochs = 10
learning_rate = 1e-3
total_images = 1000

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

full_dataset = datasets.ImageFolder(f"{data_dir}/train", transform=transform)

random.seed(42)
indices = list(range(len(full_dataset)))
random.shuffle(indices)
limited_indices = indices[:total_images]

limited_dataset = Subset(full_dataset, limited_indices)

train_size = int(0.8 * len(limited_dataset))
val_size = len(limited_dataset) - train_size
train_subset, val_subset = random_split(limited_dataset, [train_size, val_size])

train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_subset, batch_size=batch_size)

model = models.resnet18(pretrained=True)
num_classes = len(full_dataset.classes)
model.fc = nn.Linear(model.fc.in_features, num_classes)


device = torch.device("cpu")
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(epochs):
    model.train()
    total_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print(f"Validation Accuracy: {accuracy * 100:.2f}%")

# Save trained model
torch.save(model.state_dict(), "diseased_plant_classifier.pth")
print("✅ Model saved as diseased_plant_classifier.pth")