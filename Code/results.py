from federated3 import load_model_and_metrics, evaluate_model, get_client_loader, partition_data
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, f1_score
from torch.utils.data import DataLoader

# Load Model and Generate Results
def load_model_and_generate_results(model_path, model, dataloader, class_names):
    """
    Load a trained model, evaluate it, and generate metrics and confusion matrix.

    Args:
        model_path (str): Path to the saved model file.
        model (torch.nn.Module): Model architecture to load.
        dataloader (DataLoader): DataLoader for evaluation.
        class_names (list): List of class names for the dataset.

    Returns:
        None
    """
    # Load the trained model and metrics
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Initialize metrics
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device).float(), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    # Calculate metrics
    accuracy = (sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels))
    cm = confusion_matrix(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1:].sum()) if cm.shape[0] > 1 else None

    # Print metrics
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print(f"Confusion Matrix:\n{cm}")

    # Plot and save confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig(f"{model_path}_confusion_matrix.png")
    plt.close()

# Example Usage
if __name__ == "__main__":
    from federated import ClassificationDataset, device

    # Replace paths and datasets with appropriate values
    client_id = 0
    client_model_path = f"client_{client_id}_round_2.pth"
    global_model_path = "global_model_round_2.pth"

    # DataLoader for client and global models
    classification_root_dir = '/home/foysal/Documents/covid19_dataset'
    classification_ds = ClassificationDataset(classification_root_dir)

    # Partition data for clients
    num_clients = 5
    client_data = partition_data(classification_ds, num_clients)

    # Ensure no training takes place
    client_loader = get_client_loader(client_id, classification_ds, client_data)
    global_loader = DataLoader(classification_ds, batch_size=32, shuffle=False)

    # Replace with your model architecture
    from federated import model

    # Evaluate client model
    print(f"Results for Client {client_id}:")
    load_model_and_generate_results(client_model_path, model, client_loader, classification_ds.classes)

    # Evaluate global model
    print("Results for Global Model:")
    load_model_and_generate_results(global_model_path, model, global_loader, classification_ds.classes)
