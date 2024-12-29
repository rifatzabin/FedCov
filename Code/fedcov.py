# Importing necessary libraries for numerical operations, image processing, machine learning, and plotting
import numpy as np  # For numerical operations
import random  # For generating random numbers
import os  # For interacting with the operating system
import cv2  # For image processing with OpenCV
import torch  # For deep learning with PyTorch
import torch.nn as nn  # For building neural networks
import torch.optim as optim  # For optimizing neural networks
from torch.utils.data import Dataset, DataLoader, Subset  # For creating and managing data sets in PyTorch
from torchvision import models  # For using pre-trained models
import albumentations as A  # For image augmentation
from albumentations.pytorch.transforms import ToTensorV2  # For converting images to PyTorch tensors
import matplotlib.pyplot as plt  # For plotting graphs
from sklearn.metrics import confusion_matrix, f1_score  # For evaluating model performance
import seaborn as sns  # For high-level interface for drawing attractive and informative statistical graphics
from collections import defaultdict  # For creating default dictionary which provides a default value for the key that does not exists.

# Function to set seed for reproducibility of results
def set_seed(seed=42):
    """
    Sets the seed for random number generators in numpy, random, and torch to ensure results
    are reproducible. 
    """
    random.seed(seed)  # Seed the random number generator from Python's random module
    np.random.seed(seed)  # Seed the random number generator from numpy
    torch.manual_seed(seed)  # Seed the random number generators for torch
    torch.cuda.manual_seed_all(seed)  # Seed all GPUs if available for reproducibility
    torch.backends.cudnn.deterministic = True  # Force CuDNN to use only deterministic algorithms
    torch.backends.cudnn.benchmark = False  # Disable the benchmark mode in CuDNN

set_seed(903)  # Applying the seed setting function with a specific seed value

# Custom dataset class for handling image datasets
class ClassificationDataset(Dataset):
    """
    A custom dataset class for image classification tasks. It organizes data by reading images,
    applying optional transformations, and returning images with labels.
    """
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir  # Directory containing all data
        self.transform = transform  # Transformations to apply to the images
        self.classes = ['COVID', 'Pneumonia', 'Normal']  # Classes of images
        self.data = []  # List to store data paths and labels
        self.class_to_id = {clazz: i for i, clazz in enumerate(self.classes)}  # Map class names to numeric labels
        for clazz in self.classes:
            class_dir = os.path.join(root_dir, clazz)  # Directory for each class
            class_id = self.class_to_id[clazz]  # Numeric label for the class
            for img in os.listdir(class_dir):  # List all images in the directory
                img_path = os.path.join(class_dir, img)  # Full path to the image
                self.data.append((img_path, class_id))  # Append image path and label tuple to data list

    def __len__(self):
        return len(self.data)  # Returns the size of the dataset

    def __getitem__(self, idx):
        """
        Fetches an image by index, applies transformations and returns the image with its label.
        """
        img_path, label = self.data[idx]  # Retrieve path and label based on index
        img = cv2.imread(img_path)  # Read the image from file
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert the image from BGR to RGB
        if self.transform:
            img = self.transform(image=img)['image']  # Apply transformations
        return img, label  # Return the transformed image and its label

# Function to partition dataset into non-IID client data
def partition_data(dataset, num_clients):
    """
    Partition the dataset into non-IID subsets for multiple clients. This can simulate a real-world
    scenario where data is not identically distributed across different clients.
    """
    client_data = defaultdict(list)  # Create a dictionary to keep track of data for each client
    labels = np.array([dataset[i][1] for i in range(len(dataset))])  # Extract labels from the dataset
    unique_labels = np.unique(labels)  # Get unique labels from the dataset

    for label in unique_labels:
        label_indices = np.where(labels == label)[0]  # Find indices of each label
        np.random.shuffle(label_indices)  # Shuffle indices to randomize data
        split = np.array_split(label_indices, num_clients)  # Split indices into parts for each client
        for i, indices in enumerate(split):
            client_data[i].extend(indices)  # Assign indices to the appropriate client

    for client in client_data:
        np.random.shuffle(client_data[client])  # Shuffle each client's data indices for further randomization

    return client_data  # Return the dictionary containing data indices for each client

# Function to create a DataLoader for each client's data
def get_client_loader(client_id, dataset, client_data, batch_size=32):
    """
    Create a DataLoader for a specific client's dataset. This loader will be used to fetch batches of data
    during the training of models.
    """
    indices = client_data[client_id]  # Retrieve the indices for the client's data
    client_subset = Subset(dataset, indices)  # Create a subset of the dataset using those indices
    return DataLoader(client_subset, batch_size=batch_size, shuffle=True)  # Return a DataLoader for the client

# Define a local training function for clients
def local_train(client_id, model, optimizer, loss_fn, dataset, client_data, epochs=1):
    """
    Perform local training for a model on a specific client's dataset. This simulates the training
    phase in federated learning where each client trains a model on their local data.
    """
    model.train()  # Set the model to training mode
    train_loader = get_client_loader(client_id, dataset, client_data)  # Get the DataLoader for the client
    total_loss = 0  # Initialize total loss to zero

    for epoch in range(epochs):  # Iterate over the number of epochs
        for inputs, labels in train_loader:  # Iterate over batches of data
            inputs, labels = inputs.to(device).float(), labels.to(device)  # Move data to the appropriate device and ensure data is float
            optimizer.zero_grad()  # Clear gradients from the previous steps
            outputs = model(inputs)  # Forward pass: compute predicted outputs
            loss = loss_fn(outputs, labels)  # Calculate loss
            loss.backward()  # Backward pass: compute gradient of the loss with respect to model parameters
            optimizer.step()  # Perform a single optimization step (parameter update)
            total_loss += loss.item()  # Accumulate the loss

    return total_loss / len(train_loader)  # Return average loss over the training loader

# Function to compute entropy for clients' predictions (not in original code)
def compute_client_entropy(model, dataloader):
    """
    Calculate the entropy of predictions from a model to evaluate the uncertainty of its predictions.
    Higher entropy implies less certainty.
    """
    model.eval()  # Set the model to evaluation mode
    entropies = []  # List to store entropy values

    with torch.no_grad():  # Disable gradient calculation
        for inputs, _ in dataloader:  # Iterate over the data loader
            inputs = inputs.to(device).float()  # Ensure input data is on the correct device and is of type float
            outputs = model(inputs)  # Compute the model output
            probabilities = torch.softmax(outputs, dim=1)  # Calculate class probabilities
            entropy = -torch.sum(probabilities * torch.log(probabilities + 1e-9), dim=1)  # Calculate entropy
            entropies.extend(entropy.cpu().numpy())  # Store entropy values

    return np.mean(entropies)  # Return the average entropy across all predictions

# Define a federated learning averaging function
def federated_entropy_weighted_averaging(global_model, client_updates, client_entropies):
    """
    Update the global model by averaging client models weighted by the inverse of their prediction entropy.
    This helps to give more weight to more confident (less uncertain) models.
    """
    global_state = global_model.state_dict()  # Retrieve global model state
    weights = np.exp(-np.array(client_entropies))  # Convert entropies to weights
    normalized_weights = weights / np.sum(weights)  # Normalize weights

    for key in global_state.keys():
        # Aggregate weighted parameters from clients
        global_state[key] = sum(
            normalized_weights[client_id] * client_updates[client_id][key]
            for client_id in client_updates
        )

    global_model.load_state_dict(global_state)  # Load the updated global state into the global model

# Function to evaluate the model's performance
def evaluate_model(model, dataloader):
    """
    Evaluate the model's performance on a given dataset and return accuracy, F1 score, and a confusion matrix.
    """
    model.eval()  # Set the model to evaluation mode
    all_labels = []  # List to store all true labels
    all_preds = []  # List to store all predictions
    correct = 0  # Counter for correct predictions
    total = 0  # Counter for total predictions

    with torch.no_grad():  # Disable gradient calculation
        for inputs, labels in dataloader:  # Iterate over batches of data
            inputs, labels = inputs.to(device).float(), labels.to(device)  # Move data to the device and ensure it's float
            outputs = model(inputs)  # Compute model outputs
            _, predicted = torch.max(outputs.data, 1)  # Get the predicted classes
            all_labels.extend(labels.cpu().numpy())  # Append true labels
            all_preds.extend(predicted.cpu().numpy())  # Append predictions
            correct += (predicted == labels).sum().item()  # Count correct predictions
            total += labels.size(0)  # Count total predictions

    accuracy = correct / total  # Calculate accuracy
    cm = confusion_matrix(all_labels, all_preds)  # Generate confusion matrix
    f1 = f1_score(all_labels, all_preds, average='weighted')  # Calculate F1 score
    return accuracy, f1, cm  # Return accuracy, F1 score, and confusion matrix

# Initialize the dataset with a directory and optional transformations
classification_root_dir = '/home/foysal/Documents/covid19_dataset'
classification_ds = ClassificationDataset(classification_root_dir)

# Define transformations for the images
train_transform = A.Compose([
    A.Resize(224, 224),  # Resize images to 224x224 pixels
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize images
    ToTensorV2()  # Convert images to PyTorch tensors
])
classification_ds.transform = train_transform  # Apply transformations to the dataset

# Partition the data into subsets for different clients
num_clients = 5
client_data = partition_data(classification_ds, num_clients)

# Define the model architecture using a pre-trained VGG16 model
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

# Prepare for federated training
loss_fn = nn.CrossEntropyLoss()
num_rounds = 10
local_epochs = 2

# Begin federated training rounds
for round_num in range(num_rounds):
    print(f"#### Federated Round {round_num + 1}/{num_rounds} ####")
    client_updates = {}
    client_entropies = []
    communicating_clients = 0

    for client_id in range(num_clients):
        local_model = model.to(device)
        local_model.load_state_dict(model.state_dict())
        optimizer = optim.Adam(local_model.parameters(), lr=0.0001)
        avg_loss = local_train(client_id, local_model, optimizer, loss_fn, classification_ds, client_data, epochs=local_epochs)

        # Check if the local training has improved
        if client_prev_losses[client_id] is None or avg_loss < client_prev_losses[client_id]:
            client_prev_losses[client_id] = avg_loss
            client_updates[client_id] = local_model.state_dict()
            entropy = compute_client_entropy(local_model, get_client_loader(client_id, classification_ds, client_data))
            client_entropies.append(entropy)
            communicating_clients += 1
        else:
            print(f"Client {client_id} did not improve loss; not contributing to model update.")

    # Calculate and display the communication overhead reduction
    reduction = 100 * (1 - (communicating_clients / num_clients))
    print(f"Communication Overhead Reduction: {reduction:.2f}%")

    # Update the global model using entropy-weighted averaging if any clients improved
    if client_updates:
        federated_entropy_weighted_averaging(model, client_updates, client_entropies)
    else:
        print("No updates to perform this round.")

    # Evaluate the updated global model
    val_loader = DataLoader(classification_ds, batch_size=32, shuffle=False)
    accuracy, f1, cm = evaluate_model(model, val_loader)

    # Plot and save the confusion matrix for visual evaluation of the model's performance
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classification_ds.classes, yticklabels=classification_ds.classes)
    plt.title(f"Confusion Matrix - Global Model Round {round_num + 1}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig(f"confusion_matrix_global_round_{round_num + 1}.png")
    plt.close()
