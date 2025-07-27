import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torchinfo import summary

import torchvision
from torchvision import datasets, transforms

import matplotlib.pyplot as plt
import numpy as np
import random
import time

#set seed value
SEED_VALUE = 42
random.seed(SEED_VALUE)
np.random.seed(SEED_VALUE)
torch.manual_seed(SEED_VALUE)

#for gpu support
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED_VALUE)
    torch.cuda.manual_seed_all(SEED_VALUE)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark =True




#conver to tensors and normalize
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)) # mean and std
])

#download and load dataset
train_set = datasets.FashionMNIST(root = "F_MNIST_data", download = True, train = True, transform = transform) #train set
val_set = datasets.FashionMNIST(root = "F_MNIST_data", download = True, train = False, transform = transform) # test set

print("Tota; Train Images: ", len(train_set))
print("Total Val Images: ", len(val_set))

#set batch size and shuffle
train_loader = torch.utils.data.DataLoader(train_set, shuffle = True, batch_size = 64)
val_loader = torch.utils.data.DataLoader(val_set, shuffle = False, batch_size = 64)

#class idx mapping
class_mapping = {
    0: "T-shirt/top",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle boot"  }

#dataset visualisation
def visualize_images(trainloader, num_images=20):
    fig = plt.figure(figsize=(10, 10))

    #iterate over first batch
    images, labels = next(iter(trainloader))

    #to calculate number of rows and columns
    num_rows = 4
    num_cols = int(np.ceil(num_images / num_rows))

    for idx in range(min(num_images, len(images))):
        image, label = images[idx], labels[idx]

        ax = fig.add_subplot(num_rows, num_cols, idx+1, xticks=[], yticks=[])
        ax.imshow(np.squeeze(image), cmap="gray")
        ax.set_title(f"{label.item()}:{class_mapping[label.item()]}")

    fig.tight_layout()
    plt.show()
    
visualize_images(train_loader, num_images=16)

### MODEL ARCHITECTURE ###

class MLP(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.fc0 = nn.Linear(784, 512)
        self.bn0 = nn.BatchNorm1d(512)
        self.fc1 = nn.Linear(512, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.fc4 = nn.Linear(64, num_classes)

        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):
        #flatten input
        x = x.view(x.shape[0], -1) #(B,784) --> 28x28 = 784

        #first fully connected layer with relu, batch norm, dropout
        x = F.relu(self.bn0(self.fc0(x)))
        x = self.dropout(x)

        x = F.relu(self.bn1(self.fc1(x)))

        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)

        x = F.relu(self.bn3(self.fc3(x)))
        x = F.log_softmax(self.fc4(x), dim = 1)

        return x
    
#instatiate model
mlp_model = MLP(num_classes = 10)

#dummy input of (B,C,H,W) = (1,1,28,28)
print(summary(mlp_model, input_size = (1,1,28,28), row_settings = ["var_names"]))
