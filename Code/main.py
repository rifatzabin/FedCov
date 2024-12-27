# %%
import numpy as np
import pandas as pd
import os
import cv2
import shutil
import matplotlib.pyplot as plt
from PIL import Image
import random
import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch import Tensor
import torchvision.transforms as transforms
from torchvision.transforms import Resize, Compose, PILToTensor, InterpolationMode, ToPILImage
from torch.utils.data import Dataset, DataLoader, random_split, ConcatDataset
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from torch.nn import init

from torchvision import models
from torchsummary import summary
from torchviz import make_dot

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve


#%%

def set_seed(seed=42):
    """Set random seed for reproducibility across numpy, random, and torch.

    Args:
        seed (int): Seed value to set for reproduction. Defaults to 42.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(903)


# %%

class Classification_dataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = ['COVID', 'Non_COVID', 'Normal']
        self.data = []
        self.class_to_id = {clazz: i for i, clazz in enumerate(self.classes)}
        for clazz in self.classes:
            class_dir = os.path.join(root_dir, clazz)
            class_id = self.class_to_id[clazz]
            for img in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img)
                self.data.append((img_path, class_id))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, id):
        img_path, label = self.data[id]

        try:
            img = cv2.imread(img_path)
            if img is None:
                raise ValueError()
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except Exception as e:
            return None, None

        if self.transform:
            transform = self.transform(image=img)
            img = transform['image']

        return img, label


#%%

classification_root_dir = '/home/foysal/Documents/covid19_dataset'
classification_ds = Classification_dataset(classification_root_dir)



#%%
def visualize_classification(dataset, n=5):
    random_sample_ids = random.sample(range(len(dataset)), k=n)
    plt.figure(figsize=(16, 8))

    for i, idx in enumerate(random_sample_ids):
        image, label = dataset[idx]
        image = image.permute(1, 2, 0).cpu().numpy()
        height, width, _ = image.shape
        class_name = dataset.dataset.classes[label]

        plt.subplot(1, n, i + 1)
        plt.imshow(image)
        plt.title(f'Class: {class_name}\nSize: ({height}, {width})')
        plt.axis('off')

    plt.tight_layout()
    plt.show()


#%%

train_size = int(0.8 * len(classification_ds))
val_size = int(0.1 * len(classification_ds))
test_size = len(classification_ds) - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(classification_ds, [train_size, val_size, test_size])


#%%

img_size = 256

train_transform = A.Compose([
            A.LongestMaxSize(max_size=img_size),
            A.GaussianBlur(blur_limit=(3, 7), p=0.5),
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8)),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])

val_transform = A.Compose([
    A.LongestMaxSize(max_size=img_size),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])


#%%

train_dataset.dataset.transform = train_transform
val_dataset.dataset.transform = val_transform



#%%

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, drop_last=True)



#%%
len(train_dataset), len(val_dataset), len(test_dataset)
print(train_dataset[0][0].shape)
visualize_classification(train_dataset, n=3)


#%%

img_size = 256
num_classes = 3

# Loading VGG16 as a base model, and remove the fully connected layers
base_model = models.vgg16(pretrained=True)
base_features = nn.Sequential(*list(base_model.features))


#%%

# Unfreeze all layers
for param in base_model.parameters():
    param.requires_grad = True



#%%

# Build the model
model = nn.Sequential(
    base_features,
    nn.AdaptiveAvgPool2d((1, 1)),
    nn.Flatten(),
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(256, 256),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(256, num_classes),
    #nn.Softmax(dim=1)
)


#%%

# Summary the model
from torchsummary import summary

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

summary(model, input_size=(3, img_size, img_size), device=device.type)


#%%

# Initialize optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=0.00001)
loss_fn = nn.CrossEntropyLoss()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

epochs = 10
best_val_accuracy = 0
best_val_loss = None

train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []

#%%


for epoch in range(epochs):
    print(f'####################################  EPOCH {epoch + 1}/{epochs}  #####################################')

    # Train phase
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device).float(), labels.to(device)

        # inputs = inputs.float()

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        # Calculate the accuracy in each batch
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        total_loss += loss.item()

        if batch_idx % 100 == 99:
            print(
                f'Batch {batch_idx + 1}/{len(train_loader)}, Loss: {loss.item():.4f}, Accuracy: {correct / total:.4f}')

    # Save loss and accuracy for training phase
    epoch_loss = total_loss / len(train_loader)
    epoch_accuracy = correct / total
    train_losses.append(epoch_loss)
    train_accuracies.append(epoch_accuracy)
    print(f'Epoch {epoch + 1} completed. Average Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}')

    # Validation phase
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device).float(), labels.to(device)

            # inputs = inputs.float()
            outputs = model(inputs)

            loss = loss_fn(outputs, labels)
            val_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_loss /= len(val_loader)
    val_accuracy = correct / total
    val_losses.append(val_loss)
    val_accuracies.append(val_accuracy)
    print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}')

    # Save the best model
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        best_val_loss = val_loss
        model_path = '/home/foysal/Documents/covid19_dataset/vgg16_from_scratch.pth'
        torch.save(model.state_dict(), model_path)
        print(f'New best model saved with accuracy {best_val_accuracy:.4f} to {model_path}')

print(f'Best model with loss {best_val_loss:.4f} and accuracy {best_val_accuracy:.4f}')
