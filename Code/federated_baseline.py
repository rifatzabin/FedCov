import numpy as np
import random
import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from torchvision import models
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, classification_report
import seaborn as sns

# Set seed for reproducibility
def set_seed(seed=42):
    """
    Sets a fixed seed for reproducibility across various operations.
    """
    random.seed(seed)  # Python's random module
    np.random.seed(seed)  # NumPy random generator
    torch.manual_seed(seed)  # PyTorch CPU random generator
    torch.cuda.manual_seed_all(seed)  # PyTorch GPU random generator
    # Ensure deterministic behavior in PyTorch
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Set seed for reproducibility
set_seed(903)

# Custom Dataset class for classification
class ClassificationDataset(Dataset):
    """
    Dataset for loading images and their corresponding class labels.
    """
    def __init__(self, root_dir, transform=None):
        """
        Initialize the dataset by reading image paths and labels.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.classes = ['COVID', 'Pneumonia', 'Normal']  # Define class names
        self.data = []  # List to store (image_path, label) tuples
        self.class_to_id = {clazz: i for i, clazz in enumerate(self.classes)}  # Map class names to indices

        # Read all image paths and associate them with class labels
        for clazz in self.classes:
            class_dir = os.path.join(root_dir, clazz)  # Path to class directory
            class_id = self.class_to_id[clazz]  # Get class ID
            for img in os.listdir(class_dir):  # Iterate over images in the class directory
                img_path = os.path.join(class_dir, img)
                self.data.append((img_path, class_id))  # Append image path and class ID

    def __len__(self):
        """
        Return the total number of samples in the dataset.
        """
        return len(self.data)

    def __getitem__(self, id):
        """
        Fetch an image and its label by index.
        """
        img_path, label = self.data[id]  # Get image path and label
        img = cv2.imread(img_path)  # Load the image using OpenCV
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB format
        if self.transform:  # Apply transformations if provided
            transform = self.transform(image=img)
            img = transform['image']
        return img, label  # Return transformed image and label

# Function to partition the dataset into non-IID subsets for federated learning
def partition_data(dataset, num_clients):
    """
    Splits dataset non-IID by class labels into subsets for clients.
    """
    client_data = defaultdict(list)  # Dictionary to store data indices for each client
    labels = np.array([dataset[i][1] for i in range(len(dataset))])  # Extract labels from the dataset
    unique_labels = np.unique(labels)  # Find unique class labels

    # Split data for each class across clients
    for label in unique_labels:
        label_indices = np.where(labels == label)[0]  # Get indices of all samples with this label
        np.random.shuffle(label_indices)  # Shuffle indices
        split = np.array_split(label_indices, num_clients)  # Split indices into equal parts
        for i, indices in enumerate(split):
            client_data[i].extend(indices)  # Assign data to clients

    # Shuffle data within each client
    for client in client_data:
        np.random.shuffle(client_data[client])

    return client_data

# Function to create a DataLoader for a client's data
def get_client_loader(client_id, dataset, client_data, batch_size=32):
    """
    Creates a DataLoader for a specific client's data.
    """
    indices = client_data[client_id]  # Get indices of data for the client
    client_subset = Subset(dataset, indices)  # Create a subset of the dataset
    return DataLoader(client_subset, batch_size=batch_size, shuffle=True)  # Return DataLoader

# Function to save model weights and associated metrics
def save_model_and_metrics(model, path, metrics):
    """
    Saves the model state and training metrics to a file.
    """
    torch.save({
        'model_state_dict': model.state_dict(),  # Save model weights
        'metrics': metrics  # Save performance metrics
    }, path)

# Function to load model weights and metrics from a file
def load_model_and_metrics(path, model):
    """
    Loads model weights and associated metrics from a file.
    """
    checkpoint = torch.load(path)  # Load checkpoint
    model.load_state_dict(checkpoint['model_state_dict'])  # Load model state
    return checkpoint['metrics']  # Return saved metrics

# Function to train the model locally on a client's data
def local_train(client_id, model, optimizer, loss_fn, dataset, client_data, epochs=1):
    """
    Trains the model on a specific client's data.
    """
    model.train()  # Set the model to training mode
    train_loader = get_client_loader(client_id, dataset, client_data)  # Create a DataLoader for the client's data
    train_losses, train_accuracies = [], []  # Lists to store training metrics

    # Perform training over multiple epochs
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0

        # Iterate over batches of data
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device).float(), labels.to(device)  # Move data to the device
            optimizer.zero_grad()  # Clear gradients
            outputs = model(inputs)  # Forward pass
            loss = loss_fn(outputs, labels)  # Compute loss
            loss.backward()  # Backpropagation
            optimizer.step()  # Update weights

            # Calculate batch accuracy
            _, predicted = torch.max(outputs.data, 1)  # Get predicted labels
            total += labels.size(0)  # Total number of samples
            correct += (predicted == labels).sum().item()  # Count correct predictions
            total_loss += loss.item()  # Accumulate loss

        # Calculate epoch metrics
        epoch_loss = total_loss / len(train_loader)  # Average loss
        epoch_accuracy = correct / total  # Accuracy
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_accuracy)
        print(f"Client {client_id} Epoch {epoch + 1}: Loss = {epoch_loss:.4f}, Accuracy = {epoch_accuracy:.4f}")

    return model.state_dict(), train_losses, train_accuracies  # Return updated model state and metrics

# Function to aggregate model updates from multiple clients
def federated_averaging(global_model, client_updates):
    """
    Aggregates model updates from clients using Federated Averaging.
    """
    global_state = global_model.state_dict()  # Get global model state
    # Average weights for each parameter across clients
    for key in global_state.keys():
        global_state[key] = torch.mean(torch.stack([client_updates[client][key] for client in client_updates]), dim=0)
    global_model.load_state_dict(global_state)  # Load aggregated weights into global model

# Function to evaluate a model on a dataset
def evaluate_model(model, dataloader):
    """
    Evaluates the model on the given dataset and returns metrics.
    """
    model.eval()  # Set the model to evaluation mode
    all_labels = []
    all_preds = []
    correct = 0
    total = 0

    with torch.no_grad():  # Disable gradient computation
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device).float(), labels.to(device)  # Move data to the device
            outputs = model(inputs)  # Forward pass
            _, predicted = torch.max(outputs.data, 1)  # Get predicted labels
            all_labels.extend(labels.cpu().numpy())  # Append true labels
            all_preds.extend(predicted.cpu().numpy())  # Append predicted labels
            total += labels.size(0)  # Count samples
            correct += (predicted == labels).sum().item()  # Count correct predictions

    accuracy = correct / total  # Compute accuracy
    cm = confusion_matrix(all_labels, all_preds)  # Confusion matrix
    f1 = f1_score(all_labels, all_preds, average='weighted')  # F1 score
    print(f"Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}")
    return accuracy, f1, cm  # Return metrics

# Visualization of Metrics and Confusion Matrix
def plot_confusion_matrix(cm, classes, title, filename):
    """
    Plot and save confusion matrix.
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig(filename)
    plt.close()

# Visualize and Save Metrics
def plot_metrics(metrics, title, filename):
    """
    Plot and save training metrics (loss, accuracy).
    """
    plt.figure()
    for key in metrics:
        plt.plot(metrics[key], label=key)
    plt.title(title)
    plt.xlabel("Epochs")
    plt.ylabel("Metric")
    plt.legend()
    plt.savefig(filename)
    plt.close()
