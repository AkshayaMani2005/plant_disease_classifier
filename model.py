import torch
from torchvision import models, transforms, datasets
from PIL import Image

data_dir = "datasets/diseased_plants/train"
model_path = "diseased_plant_classifier.pth"
image_path = "datasets/diseased_plants/train/Corn Healthy/Corn-Healthy-0a5cb475-a6e8-4233-b251-bc5165868730___R-S_HL-7978-copy_jpg.rf.8ca297737565a8d00da2f0455f1cebc7.jpg"         # Path to the image you want to predict

train_dataset = datasets.ImageFolder(data_dir)
class_names = train_dataset.classes  # List of class names

# ----------------------------
# Load the model
# ----------------------------
num_classes = len(class_names)
model = models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

# Load trained weights
model.load_state_dict(torch.load(model_path, map_location="cpu"))
model.eval()  # Evaluation mode

# ----------------------------
# Prepare the image
# ----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

image = Image.open(image_path).convert("RGB")
image = transform(image).unsqueeze(0)  # Add batch dimension

# ----------------------------
# Make prediction
# ----------------------------
with torch.no_grad():
    outputs = model(image)
    _, predicted_idx = torch.max(outputs, 1)

predicted_label = class_names[predicted_idx.item()]
print(f"Predicted label: {predicted_label}")