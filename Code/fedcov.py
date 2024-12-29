# Import necessary libraries
import numpy as np
import random
import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import models
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score
import seaborn as sns
from collections import defaultdict

# Set seed for reproducibility
def set_seed(seed=42):
    """
    Sets random seeds for numpy, torch, and random modules
    to ensure reproducibility of results.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(903)  # Fixed seed for consistent results

# Dataset class for loading images
class ClassificationDataset(Dataset):
    """
    Custom dataset for classification tasks.
    Loads images and labels them based on directory structure:
    root_dir/class_name/image.png
    """
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir  # Path to dataset directory
        self.transform = transform  # Optional image transformations
        self.classes = ['COVID', 'Pneumonia', 'Normal']  # Class labels
        self.data = []  # List to hold (image_path, label) pairs
        self.class_to_id = {clazz: i for i, clazz in enumerate(self.classes)}  # Map class names to IDs
        for clazz in self.classes:
            class_dir = os.path.join(root_dir, clazz)  # Path to class directory
            class_id = self.class_to_id[clazz]
            for img in os.listdir(class_dir):  # Iterate over images
                img_path = os.path.join(class_dir, img)
                self.data.append((img_path, class_id))  # Add (path, label) to dataset

    def __len__(self):
        return len(self.data)  # Number of samples

    def __getitem__(self, id):
        """
        Returns an image and its label for a given index.
        """
        img_path, label = self.data[id]  # Get image path and label
        img = cv2.imread(img_path)  # Load image
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        if self.transform:
            transform = self.transform(image=img)  # Apply transformations
            img = transform['image']
        return img, label

# Partition dataset into non-IID client data
def partition_data(dataset, num_clients):
    """
    Splits the dataset into non-IID partitions for clients.
    Ensures labels are unevenly distributed among clients.
    """
    client_data = defaultdict(list)  # Store indices for each client
    labels = np.array([dataset[i][1] for i in range(len(dataset))])  # Get all labels
    unique_labels = np.unique(labels)  # Find unique labels

    for label in unique_labels:
        label_indices = np.where(labels == label)[0]  # Get indices for current label
        np.random.shuffle(label_indices)  # Shuffle indices
        split = np.array_split(label_indices, num_clients)  # Split into `num_clients` parts
        for i, indices in enumerate(split):
            client_data[i].extend(indices)  # Assign to client i

    for client in client_data:
        np.random.shuffle(client_data[client])  # Shuffle client's data

    return client_data

# DataLoader for a specific client's data
def get_client_loader(client_id, dataset, client_data, batch_size=32):
    """
    Creates a DataLoader for a specific client's data subset.
    """
    indices = client_data[client_id]  # Get indices for client
    client_subset = Subset(dataset, indices)  # Create subset
    return DataLoader(client_subset, batch_size=batch_size, shuffle=True)  # Return DataLoader

# Local training function
def local_train(client_id, model, optimizer, loss_fn, dataset, client_data, epochs=1):
    """
    Trains a model locally on a specific client's data.
    """
    model.train()  # Set model to training mode
    train_loader = get_client_loader(client_id, dataset, client_data)  # Get client's data loader
    total_loss = 0  # Track total loss
    for epoch in range(epochs):
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device).float(), labels.to(device)  # Move data to device
            optimizer.zero_grad()  # Reset gradients
            outputs = model(inputs)  # Forward pass
            loss = loss_fn(outputs, labels)  # Compute loss
            loss.backward()  # Backpropagation
            optimizer.step()  # Update model parameters
            total_loss += loss.item()  # Accumulate loss
    return total_loss / len(train_loader)  # Return average loss

# Compute entropy for a client's predictions
def compute_client_entropy(model, dataloader):
    """
    Computes the average entropy of predictions for a client's model.
    Entropy measures prediction confidence.
    """
    model.eval()  # Set model to evaluation mode
    entropies = []
    with torch.no_grad():
        for inputs, _ in dataloader:
            inputs = inputs.to(device).float()  # Move inputs to device
            outputs = model(inputs)  # Get model outputs
            probabilities = torch.softmax(outputs, dim=1)  # Compute probabilities
            entropy = -torch.sum(probabilities * torch.log(probabilities + 1e-9), dim=1)  # Compute entropy
            entropies.extend(entropy.cpu().numpy())  # Add to list
    return np.mean(entropies)  # Return average entropy

# Federated entropy-weighted averaging
def federated_entropy_weighted_averaging(global_model, client_updates, client_entropies):
    """
    Aggregates client models using entropy-weighted averaging.
    """
    global_state = global_model.state_dict()  # Get global model parameters
    weights = np.exp(-np.array(client_entropies))  # Convert entropy to weights
    normalized_weights = weights / np.sum(weights)  # Normalize weights

    for key in global_state.keys():
        # Weighted sum of parameters from clients
        global_state[key] = sum(
            normalized_weights[client_id] * client_updates[client_id][key]
            for client_id in client_updates
        )

    global_model.load_state_dict(global_state)  # Update global model

# Evaluate model and return metrics
def evaluate_model(model, dataloader):
    """
    Evaluates a model and returns accuracy, F1 score, and confusion matrix.
    """
    model.eval()  # Set model to evaluation mode
    all_labels = []
    all_preds = []
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device).float(), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total  # Compute accuracy
    cm = confusion_matrix(all_labels, all_preds)  # Compute confusion matrix
    f1 = f1_score(all_labels, all_preds, average='weighted')  # Compute F1 score
    return accuracy, f1, cm

# Dataset initialization
classification_root_dir = '/home/foysal/Documents/covid19_dataset'
classification_ds = ClassificationDataset(classification_root_dir)

# Define image transformations
train_transform = A.Compose([
    A.Resize(224, 224),  # Resize images to 224x224
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize
    ToTensorV2()  # Convert to PyTorch tensor
])
classification_ds.transform = train_transform  # Apply transformations

# Partition data into non-IID subsets
num_clients = 5
client_data = partition_data(classification_ds, num_clients)

# Define model architecture
num_classes = 3
base_model = models.vgg16(pretrained=True)
model = nn.Sequential(
    *list(base_model.features),
    nn.AdaptiveAvgPool2d((1, 1)),
    nn.Flatten(),
    nn.Linear(512, num_classes)
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Track previous losses
client_prev_losses = {client_id: None for client_id in range(num_clients)}

# Federated training
loss_fn = nn.CrossEntropyLoss()
num_rounds = 10
local_epochs = 2

for round_num in range(num_rounds):
    print(f"#### Federated Round {round_num + 1}/{num_rounds} ####")
    client_updates = {}
    client_entropies = []
    communicating_clients = 0

    for client_id in range(num_clients):
        # Train locally and compute average loss
        local_model = model.to(device)
        local_model.load_state_dict(model.state_dict())
        optimizer = optim.Adam(local_model.parameters(), lr=0.0001)
        avg_loss = local_train(client_id, local_model, optimizer, loss_fn, classification_ds, client_data, epochs=local_epochs)

        # Check if loss has reduced
        if client_prev_losses[client_id] is None or avg_loss < client_prev_losses[client_id]:
            client_prev_losses[client_id] = avg_loss  # Update previous loss
            client_updates[client_id] = local_model.state_dict()  # Save updates
            entropy = compute_client_entropy(local_model, get_client_loader(client_id, classification_ds, client_data))
            client_entropies.append(entropy)
            communicating_clients += 1
        else:
            print(f"Client {client_id} did not reduce loss in this round.")

    # Calculate communication overhead reduction
    reduction = 100 * (1 - (communicating_clients / num_clients))
    print(f"Communication Overhead Reduction: {reduction:.2f}%")

    # Perform entropy-weighted averaging if updates exist
    if client_updates:
        federated_entropy_weighted_averaging(model, client_updates, client_entropies)
    else:
        print("No updates transmitted this round.")

    # Evaluate global model
    val_loader = DataLoader(classification_ds, batch_size=32, shuffle=False)
    accuracy, f1, cm = evaluate_model(model, val_loader)

    # Save and plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classification_ds.classes,
                yticklabels=classification_ds.classes)
    plt.title(f"Global Model Round {round_num + 1} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig(f"confusion_matrix_global_round_{round_num + 1}.png")
    plt.close()
