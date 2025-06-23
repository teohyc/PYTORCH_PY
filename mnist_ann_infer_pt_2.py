import torch
import torch.nn as nn
from PIL import Image
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt

# 1. Define the improved ANN model structure
class ImprovedANN(nn.Module):
    def __init__(self):
        super(ImprovedANN, self).__init__()
        self.model = nn.Sequential(
            nn.Flatten(),                    # Converts (1, 28, 28) → (784,)
            nn.Linear(784, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 10)               # 10 classes (0–9)
        )

    def forward(self, x):
        return self.model(x)

# 2. Load the model and weights
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

model = ImprovedANN().to(device)
model.load_state_dict(torch.load("mnist_ann_model_2.pt", map_location=device))
model.eval()  # Set to evaluation mode (important for BatchNorm and Dropout)
print("✅ Improved model loaded.")

# 3. Preprocess the input image
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Grayscale(),              # Ensure grayscale input
        transforms.Resize((28, 28)),         # Match MNIST size
        transforms.ToTensor(),               # Convert to tensor
        transforms.Normalize((0.5,), (0.5,)) # Normalize to [-1, 1]
    ])
    
    image = Image.open(image_path)

    # Optional inversion if digit is black on white
    image_array = np.array(image.convert("L"))
    if np.mean(image_array) > 127:
        image_array = 255 - image_array
        image = Image.fromarray(image_array)

    image_tensor = transform(image).unsqueeze(0)  # Shape: (1, 1, 28, 28)
    return image_tensor.to(device)

# 4. Predict the digit and display the result
def predict_image(image_path):
    image_tensor = preprocess_image(image_path)

    with torch.no_grad():
        outputs = model(image_tensor)               # Get logits
        predicted = torch.argmax(outputs, dim=1)    # Class with highest score
        confidence = torch.softmax(outputs, dim=1).max().item() * 100

    # Display image with prediction result
    img = Image.open(image_path).convert("L")
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.title(f"Predicted: {predicted.item()} ({confidence:.2f}%)")
    plt.show()

# 5. Use the function
predict_image("test8.jpg")  # 🔁 Change to your image path
