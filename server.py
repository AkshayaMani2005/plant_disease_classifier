import torch
from torchvision import models, transforms, datasets
from PIL import Image
from flask import Flask, request, jsonify
import io

train_dataset = datasets.ImageFolder("datasets/diseased_plants/train")
labels = train_dataset.classes

no_of_classes = len(labels)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

model = models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, no_of_classes)
model.load_state_dict(torch.load("diseased_plant_classifier.pth", map_location="cpu"))
model.eval()
print("Model loaded!")

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    
    file = request.files["image"]
    try:
        image = Image.open(io.BytesIO(file.read())).convert("RGB")
    except Exception as e:
        print(f"Error opening image: {e}")
        return jsonify({"error": "Invalid image"}), 400
    
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(image)
        _, predicted_idx = torch.max(outputs, 1)

    predicted_label = labels[predicted_idx.item()]
    print(f"Predicted label: {predicted_label}")

    return jsonify({"predicted_label": predicted_label})

if __name__ == "__main__":
    print("Starting Flask server on http://0.0.0.0:5000")
    app.run(host="0.0.0.0", port=5000, debug=True)