import torch
from torchvision import models, transforms
from PIL import Image
from flask import Flask, request, jsonify
import io

labels =  ['Cassava Bacterial Blight', 'Cassava Healthy', 'Cassava Mosaic Disease', 'Corn Cercospora Leaf Spot', 'Corn Common Rust', 'Corn Healthy', 'Corn Northern Leaf Blight', 'Mango Anthracnose', 'Mango Gall Midge', 'Mango Healthy', 'Mango Powdery Mildew', 'Orange Citrus Greening', 'Pepper Bacterial Spot', 'Pepper Healthy', 'Potato Early Blight', 'Potato Healthy', 'Potato Late Blight', 'Rice BrownSpot', 'Rice Healthy', 'Tomato Bacterial Spot', 'Tomato Early Blight', 'Tomato Healthy', 'Tomato Late Blight', 'Tomato Yellow Leaf Curl Virus']

no_of_classes = len(labels)

print("labels: ",labels)

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
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=10000)
