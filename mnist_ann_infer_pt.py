import torch
import torch.nn as nn
from PIL import Image
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt

# 1. Define the same model structure as during training
class SimpleANN(nn.Module):
    def __init__(self):
        super(SimpleANN, self).__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )
        
    def forward(self, x):
        return self.net(x)

# 2. Load the trained model weights
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
model = SimpleANN().to(device)
model.load_state_dict(torch.load("mnist_ann_model.pt", map_location=device))
model.eval()
print("✅ Model loaded.")

# 3. Preprocess an external image (grayscale, 28x28)
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Grayscale(),          # Convert to grayscale
        transforms.Resize((28, 28)),     # Resize to 28x28
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    image = Image.open(image_path)
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    return image_tensor.to(device)

# 4. Predict digit
def predict_image(image_path):
    image_tensor = preprocess_image(image_path)
    
    with torch.no_grad():
        outputs = model(image_tensor)
        predicted = torch.argmax(outputs, dim=1)
        confidence = torch.softmax(outputs, dim=1).max().item() * 100

    # Display the input image
    img = Image.open(image_path).convert("L")
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.title(f"Predicted: {predicted.item()} ({confidence:.2f}%)")
    plt.show()

# 5. Example usage
# Replace 'your_image.png' with your image path
predict_image("test0.jpg")
