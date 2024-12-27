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
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(903)


# Dataset class
class ClassificationDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = ['COVID', 'Pneumonia', 'Normal']
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
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform:
            transform = self.transform(image=img)
            img = transform['image']
        return img, label


# Non-IID Data Partitioning
def partition_data(dataset, num_clients):
    client_data = defaultdict(list)
    labels = np.array([dataset[i][1] for i in range(len(dataset))])
    unique_labels = np.unique(labels)

    for label in unique_labels:
        label_indices = np.where(labels == label)[0]
        np.random.shuffle(label_indices)
        split = np.array_split(label_indices, num_clients)
        for i, indices in enumerate(split):
            client_data[i].extend(indices)

    for client in client_data:
        np.random.shuffle(client_data[client])

    return client_data


# Data Loaders for Clients
def get_client_loader(client_id, dataset, client_data, batch_size=32):
    indices = client_data[client_id]
    client_subset = Subset(dataset, indices)
    return DataLoader(client_subset, batch_size=batch_size, shuffle=True)


# Save Model and Metrics
def save_model_and_metrics(model, path, metrics):
    torch.save({
        'model_state_dict': model.state_dict(),
        'metrics': metrics
    }, path)


# Load Model and Metrics
def load_model_and_metrics(path, model):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    return checkpoint['metrics']


# Define Training Function for Clients
def local_train(client_id, model, optimizer, loss_fn, dataset, client_data, epochs=1):
    model.train()
    train_loader = get_client_loader(client_id, dataset, client_data)
    train_losses, train_accuracies = [], []
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device).float(), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            # Track accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            total_loss += loss.item()

        epoch_loss = total_loss / len(train_loader)
        epoch_accuracy = correct / total
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_accuracy)
        print(f"Client {client_id} Epoch {epoch + 1}: Loss = {epoch_loss:.4f}, Accuracy = {epoch_accuracy:.4f}")
    return model.state_dict(), train_losses, train_accuracies


# Federated Averaging
def federated_averaging(global_model, client_updates):
    global_state = global_model.state_dict()
    for key in global_state.keys():
        global_state[key] = torch.mean(torch.stack([client_updates[client][key] for client in client_updates]), dim=0)
    global_model.load_state_dict(global_state)


# Evaluation Function
def evaluate_model(model, dataloader):
    model.eval()
    all_labels = []
    all_preds = []
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device).float(), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    cm = confusion_matrix(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1:].sum()) if cm.shape[0] > 1 else None
    print(f"Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}, Specificity: {specificity:.4f}")
    return accuracy, f1, specificity, cm


# Define Dataset and Transforms
classification_root_dir = '/home/foysal/Documents/covid19_dataset'
classification_ds = ClassificationDataset(classification_root_dir)

train_transform = A.Compose([
    A.LongestMaxSize(max_size=256),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])
classification_ds.transform = train_transform

# Partition Data for Federated Learning
num_clients = 5
client_data = partition_data(classification_ds, num_clients)

# Define Global Model
num_classes = 3
base_model = models.vgg16(pretrained=True)
base_features = nn.Sequential(*list(base_model.features))

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
    nn.Linear(256, num_classes)
)

# Training Federated Learning Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
loss_fn = nn.CrossEntropyLoss()

num_rounds = 10
local_epochs = 2

train_metrics = {client_id: {'loss': [], 'accuracy': []} for client_id in range(num_clients)}
val_metrics = {'loss': [], 'accuracy': []}

for round_num in range(num_rounds):
    print(f"#### Federated Round {round_num + 1}/{num_rounds} ####")

    client_updates = {}
    for client_id in range(num_clients):
        local_model = model.to(device)
        local_model.load_state_dict(model.state_dict())
        optimizer = optim.Adam(local_model.parameters(), lr=0.0001)
        client_state, train_losses, train_accuracies = local_train(client_id, local_model, optimizer, loss_fn,
                                                                   classification_ds, client_data, epochs=local_epochs)
        client_updates[client_id] = client_state

        train_metrics[client_id]['loss'].extend(train_losses)
        train_metrics[client_id]['accuracy'].extend(train_accuracies)

        save_model_and_metrics(local_model, f"client_{client_id}_round_{round_num + 1}.pth", {
            'loss': train_losses,
            'accuracy': train_accuracies
        })

        # Evaluate and save confusion matrix for each client
        client_loader = get_client_loader(client_id, classification_ds, client_data)
        _, _, _, cm = evaluate_model(local_model, client_loader)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classification_ds.classes,
                    yticklabels=classification_ds.classes)
        plt.title(f"Confusion Matrix - Client {client_id}")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.savefig(f"confusion_matrix_client_{client_id}_round_{round_num + 1}.png")
        plt.close()

    federated_averaging(model, client_updates)
    print(f"Completed Federated Round {round_num + 1}")

    save_model_and_metrics(model, f"global_model_round_{round_num + 1}.pth", val_metrics)

    # Evaluate Global Model
    val_loader = DataLoader(classification_ds, batch_size=32, shuffle=False)
    accuracy, f1, specificity, cm = evaluate_model(model, val_loader)
    val_metrics['loss'].append(0)  # Placeholder if you don't compute validation loss
    val_metrics['accuracy'].append(accuracy)

    # Save confusion matrix for the global model
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classification_ds.classes,
                yticklabels=classification_ds.classes)
    plt.title("Confusion Matrix - Global Model")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig(f"confusion_matrix_global_round_{round_num + 1}.png")
    plt.close()

# Plot Metrics
for client_id in range(num_clients):
    metrics = load_model_and_metrics(f"client_{client_id}_round_{num_rounds}.pth", model)
    plt.figure()
    plt.plot(metrics['loss'], label="Training Loss")
    plt.plot(metrics['accuracy'], label="Training Accuracy")
    plt.title(f"Client {client_id} Training Metrics")
    plt.xlabel("Epoch")
    plt.ylabel("Metric")
    plt.legend()
    plt.savefig(f"client_{client_id}_metrics.png")

plt.figure()
plt.plot(val_metrics['accuracy'], label="Validation Accuracy")
plt.title("Global Model Validation Accuracy")
plt.xlabel("Round")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig("global_validation_accuracy.png")
