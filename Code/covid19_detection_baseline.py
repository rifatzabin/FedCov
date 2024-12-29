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
    """
    Set random seed for reproducibility across numpy, random, and torch.

    Args:
        seed (int): Seed value to set for reproduction. Defaults to 42.
    """
    random.seed(seed)  # Python's random module
    np.random.seed(seed)  # NumPy random generator
    torch.manual_seed(seed)  # PyTorch CPU generator
    torch.cuda.manual_seed_all(seed)  # PyTorch GPU generator
    torch.backends.cudnn.deterministic = True  # Ensure deterministic behavior in cuDNN
    torch.backends.cudnn.benchmark = False  # Disable cuDNN autotuner for reproducibility

set_seed(903)

# %%

class Classification_dataset(Dataset):
    """
    Custom dataset class for classification tasks.
    Each sample is an image and its corresponding label.
    """
    def __init__(self, root_dir, transform=None):
        """
        Initialize the dataset with the directory structure and optional transformations.

        Args:
            root_dir (str): Root directory containing class-wise subdirectories of images.
            transform (callable, optional): Transformations to apply to the images. Defaults to None.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.classes = ['COVID', 'Non_COVID', 'Normal']  # Define class names
        self.data = []  # Store (image_path, label) tuples
        self.class_to_id = {clazz: i for i, clazz in enumerate(self.classes)}  # Map class names to IDs

        # Iterate through each class and collect image paths with labels
        for clazz in self.classes:
            class_dir = os.path.join(root_dir, clazz)  # Class-specific directory
            class_id = self.class_to_id[clazz]  # Get corresponding class ID
            for img in os.listdir(class_dir):  # Iterate over images in the directory
                img_path = os.path.join(class_dir, img)  # Full path to image
                self.data.append((img_path, class_id))  # Add (path, label) tuple to dataset

    def __len__(self):
        """
        Return the total number of samples in the dataset.
        """
        return len(self.data)

    def __getitem__(self, id):
        """
        Retrieve an image and its label by index.

        Args:
            id (int): Index of the sample.

        Returns:
            tuple: Processed image tensor and its corresponding label.
        """
        img_path, label = self.data[id]  # Get image path and label

        try:
            img = cv2.imread(img_path)  # Read the image using OpenCV
            if img is None:
                raise ValueError()  # Handle cases where image cannot be read
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB format
        except Exception as e:
            return None, None  # Return None for invalid samples

        if self.transform:  # Apply transformations if provided
            transform = self.transform(image=img)
            img = transform['image']

        return img, label  # Return processed image and label

#%%

classification_root_dir = '/home/foysal/Documents/covid19_dataset'
classification_ds = Classification_dataset(classification_root_dir)

#%%
def visualize_classification(dataset, n=5):
    """
    Visualize a random sample of images from the dataset with their labels.

    Args:
        dataset (Dataset): Dataset from which to sample images.
        n (int): Number of images to visualize. Defaults to 5.
    """
    random_sample_ids = random.sample(range(len(dataset)), k=n)  # Randomly select sample indices
    plt.figure(figsize=(16, 8))

    for i, idx in enumerate(random_sample_ids):
        image, label = dataset[idx]  # Get image and label
        image = image.permute(1, 2, 0).cpu().numpy()  # Convert tensor to numpy array
        height, width, _ = image.shape  # Get image dimensions
        class_name = dataset.dataset.classes[label]  # Map label to class name

        plt.subplot(1, n, i + 1)  # Create subplot for visualization
        plt.imshow(image)  # Display the image
        plt.title(f'Class: {class_name}\nSize: ({height}, {width})')  # Add title with class and size
        plt.axis('off')  # Remove axis ticks

    plt.tight_layout()  # Adjust subplot spacing
    plt.show()  # Show the plot

#%%

# Split the dataset into training, validation, and test sets
train_size = int(0.8 * len(classification_ds))  # 80% for training
val_size = int(0.1 * len(classification_ds))  # 10% for validation
test_size = len(classification_ds) - train_size - val_size  # Remaining for testing
train_dataset, val_dataset, test_dataset = random_split(classification_ds, [train_size, val_size, test_size])

#%%

# Define image size for preprocessing
img_size = 256

# Define transformations for training and validation
train_transform = A.Compose([
            A.LongestMaxSize(max_size=img_size),  # Resize the image
            A.GaussianBlur(blur_limit=(3, 7), p=0.5),  # Apply Gaussian blur
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8)),  # Apply CLAHE for contrast enhancement
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize the image
            ToTensorV2(),  # Convert to PyTorch tensor
        ])

val_transform = A.Compose([
    A.LongestMaxSize(max_size=img_size),  # Resize the image
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize the image
    ToTensorV2(),  # Convert to PyTorch tensor
])

#%%

# Apply transformations to datasets
train_dataset.dataset.transform = train_transform
val_dataset.dataset.transform = val_transform

#%%

# Create DataLoaders for training, validation, and testing
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, drop_last=True)

#%%

# Print dataset sizes and visualize some training samples
len(train_dataset), len(val_dataset), len(test_dataset)
print(train_dataset[0][0].shape)
visualize_classification(train_dataset, n=3)

#%%

# Define model parameters
img_size = 256
num_classes = 3

# Load pre-trained VGG16 model and remove fully connected layers
base_model = models.vgg16(pretrained=True)
base_features = nn.Sequential(*list(base_model.features))

#%%

# Unfreeze all layers of the base model
for param in base_model.parameters():
    param.requires_grad = True

#%%

# Build the classification model
model = nn.Sequential(
    base_features,  # Feature extractor
    nn.AdaptiveAvgPool2d((1, 1)),  # Pooling layer
    nn.Flatten(),  # Flatten the feature maps
    nn.Linear(512, 256),  # Fully connected layer
    nn.ReLU(),  # Activation function
    nn.Dropout(0.5),  # Dropout for regularization
    nn.Linear(256, 256),  # Fully connected layer
    nn.ReLU(),  # Activation function
    nn.Dropout(0.5),  # Dropout for regularization
    nn.Linear(256, num_classes),  # Output layer
)

#%%

# Summarize the model
from torchsummary import summary

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available
model.to(device)  # Move model to device

summary(model, input_size=(3, img_size, img_size), device=device.type)  # Print model summary

#%%

# Initialize optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=0.00001)  # Adam optimizer with learning rate
loss_fn = nn.CrossEntropyLoss()  # Cross-entropy loss for classification

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available
model.to(device)  # Move model to device

# Initialize training parameters
epochs = 10
best_val_accuracy = 0
best_val_loss = None

train_losses = []  # Store training losses
train_accuracies = []  # Store training accuracies
val_losses = []  # Store validation losses
val_accuracies = []  # Store validation accuracies

#%%

# Training and validation loop
for epoch in range(epochs):
    print(f'####################################  EPOCH {epoch + 1}/{epochs}  #####################################')

    # Train phase
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device).float(), labels.to(device)  # Move data to device

        optimizer.zero_grad()  # Clear gradients
        outputs = model(inputs)  # Forward pass
        loss = loss_fn(outputs, labels)  # Compute loss
        loss.backward()  # Backpropagation
        optimizer.step()  # Update weights

        # Calculate the accuracy in each batch
        _, predicted = torch.max(outputs.data, 1)  # Get predicted labels
        total += labels.size(0)  # Total samples
        correct += (predicted == labels).sum().item()  # Correct predictions
        total_loss += loss.item()  # Accumulate loss

        if batch_idx % 100 == 99:  # Log every 100 batches
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
            inputs, labels = inputs.to(device).float(), labels.to(device)  # Move data to device

            outputs = model(inputs)  # Forward pass
            loss = loss_fn(outputs, labels)  # Compute loss
            val_loss += loss.item()  # Accumulate validation loss

            _, predicted = torch.max(outputs.data, 1)  # Get predicted labels
            total += labels.size(0)  # Total samples
            correct += (predicted == labels).sum().item()  # Correct predictions

    val_loss /= len(val_loader)  # Average validation loss
    val_accuracy = correct / total  # Validation accuracy
    val_losses.append(val_loss)
    val_accuracies.append(val_accuracy)
    print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}')

    # Save the best model
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        best_val_loss = val_loss
        model_path = '/home/foysal/Documents/covid19_dataset/vgg16_from_scratch.pth'
        torch.save(model.state_dict(), model_path)  # Save model state
        print(f'New best model saved with accuracy {best_val_accuracy:.4f} to {model_path}')

print(f'Best model with loss {best_val_loss:.4f} and accuracy {best_val_accuracy:.4f}')
