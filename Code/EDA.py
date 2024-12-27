import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import cv2
from collections import Counter
import numpy as np
from collections import defaultdict
from torch.utils.data import Dataset

# Global font size variables
FONT_SIZE_XLABEL = 27
FONT_SIZE_YLABEL = 27
FONT_SIZE_XTICKS = 25
FONT_SIZE_YTICKS = 25
FONT_SIZE_TITLE = 27

# Define Dataset Directory
classification_root_dir = '/home/foysal/Documents/covid19_dataset'
classes = ['COVID', 'Pneumonia', 'Normal']

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

# Analyze Dataset Distribution
def analyze_dataset_distribution(root_dir, classes):
    """Analyze and plot the class distribution of the dataset."""
    class_counts = {}

    for clazz in classes:
        class_dir = os.path.join(root_dir, clazz)
        count = len(os.listdir(class_dir))
        class_counts[clazz] = count

    # Plot distribution
    plt.figure(figsize=(8, 6))
    sns.barplot(x=list(class_counts.keys()), y=list(class_counts.values()), palette='viridis')
    plt.title('Class Distribution', fontsize=FONT_SIZE_TITLE)
    plt.xlabel('Class', fontsize=FONT_SIZE_XLABEL)
    plt.ylabel('Number of Images', fontsize=FONT_SIZE_YLABEL)
    plt.xticks(fontsize=FONT_SIZE_XTICKS)
    plt.yticks(fontsize=FONT_SIZE_YTICKS)
    plt.tight_layout()
    plt.savefig('class_distribution.png')
    plt.show()

# Analyze Image Dimensions
def analyze_image_dimensions(root_dir, classes):
    """Analyze and plot the distribution of image dimensions."""
    dimensions = []
    aspect_ratios = []

    for clazz in classes:
        class_dir = os.path.join(root_dir, clazz)
        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            img = cv2.imread(img_path)
            if img is not None:
                height, width = img.shape[:2]
                dimensions.append((height, width))
                aspect_ratios.append(width / height)

    # Convert to DataFrame
    dims_df = pd.DataFrame(dimensions, columns=['Height', 'Width'])
    dims_df['Aspect Ratio'] = aspect_ratios

    # Plot distributions
    plt.figure(figsize=(8, 6))
    sns.histplot(dims_df['Height'], kde=True, color='blue', label='Height', bins=30)
    sns.histplot(dims_df['Width'], kde=True, color='orange', label='Width', bins=50)
    plt.title('Image Dimensions Distribution', fontsize=FONT_SIZE_TITLE)
    plt.xlabel('Pixels', fontsize=FONT_SIZE_XLABEL)
    plt.ylabel('Frequency', fontsize=FONT_SIZE_YLABEL)
    plt.xticks(fontsize=FONT_SIZE_XTICKS)
    plt.yticks(fontsize=FONT_SIZE_YTICKS)
    plt.legend(fontsize=FONT_SIZE_TITLE)
    plt.tight_layout()
    plt.savefig('image_dimensions_distribution.png')
    plt.show()

    plt.figure(figsize=(8, 6))
    sns.histplot(dims_df['Aspect Ratio'], kde=True, color='green', bins=30)
    plt.title('Aspect Ratio Distribution', fontsize=FONT_SIZE_TITLE)
    plt.xlabel('Aspect Ratio', fontsize=FONT_SIZE_XLABEL)
    plt.ylabel('Frequency', fontsize=FONT_SIZE_YLABEL)
    plt.xticks(fontsize=FONT_SIZE_XTICKS)
    plt.yticks(fontsize=FONT_SIZE_YTICKS)
    plt.tight_layout()
    plt.savefig('aspect_ratio_distribution.png')
    plt.show()

# Analyze Image Statistics
def analyze_image_statistics(root_dir, classes):
    """Analyze image statistics such as mean and standard deviation of pixel intensities."""
    means = []
    stds = []

    for clazz in classes:
        class_dir = os.path.join(root_dir, clazz)
        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = img / 255.0  # Normalize
                means.append(np.mean(img, axis=(0, 1)))
                stds.append(np.std(img, axis=(0, 1)))

    means = np.array(means)
    stds = np.array(stds)

    # Plot RGB Mean Distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(means[:, 0], color='red', label='Red Channel', kde=True)
    #sns.histplot(means[:, 1], color='green', label='Green Channel', kde=True)
    #sns.histplot(means[:, 2], color='blue', label='Blue Channel', kde=True)
    plt.title('Mean Pixel Intensity Distribution', fontsize=FONT_SIZE_TITLE)
    plt.xlabel('Mean Intensity', fontsize=FONT_SIZE_XLABEL)
    plt.ylabel('Frequency', fontsize=FONT_SIZE_YLABEL)
    plt.xticks(fontsize=FONT_SIZE_XTICKS)
    plt.yticks(fontsize=FONT_SIZE_YTICKS)
    plt.legend(fontsize=FONT_SIZE_TITLE)
    plt.tight_layout()
    plt.savefig('mean_pixel_intensity_distribution_red.png')
    plt.show()

    # Plot RGB Mean Distribution
    plt.figure(figsize=(10, 6))
    #sns.histplot(means[:, 0], color='red', label='Red Channel', kde=True)
    sns.histplot(means[:, 1], color='green', label='Green Channel', kde=True)
    #sns.histplot(means[:, 2], color='blue', label='Blue Channel', kde=True)
    plt.title('Mean Pixel Intensity Distribution', fontsize=FONT_SIZE_TITLE)
    plt.xlabel('Mean Intensity', fontsize=FONT_SIZE_XLABEL)
    plt.ylabel('Frequency', fontsize=FONT_SIZE_YLABEL)
    plt.xticks(fontsize=FONT_SIZE_XTICKS)
    plt.yticks(fontsize=FONT_SIZE_YTICKS)
    plt.legend(fontsize=FONT_SIZE_TITLE)
    plt.tight_layout()
    plt.savefig('mean_pixel_intensity_distribution_green.png')
    plt.show()


    # Plot RGB Mean Distribution
    plt.figure(figsize=(10, 6))
    #sns.histplot(means[:, 0], color='red', label='Red Channel', kde=True)
    #sns.histplot(means[:, 1], color='green', label='Green Channel', kde=True)
    sns.histplot(means[:, 2], color='blue', label='Blue Channel', kde=True)
    plt.title('Mean Pixel Intensity Distribution', fontsize=FONT_SIZE_TITLE)
    plt.xlabel('Mean Intensity', fontsize=FONT_SIZE_XLABEL)
    plt.ylabel('Frequency', fontsize=FONT_SIZE_YLABEL)
    plt.xticks(fontsize=FONT_SIZE_XTICKS)
    plt.yticks(fontsize=FONT_SIZE_YTICKS)
    plt.legend(fontsize=FONT_SIZE_TITLE)
    plt.tight_layout()
    plt.savefig('mean_pixel_intensity_distribution_blue.png')
    plt.show()

    # Plot RGB Std Distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(stds[:, 0], color='red', label='Red Channel', kde=True)
    #sns.histplot(stds[:, 1], color='green', label='Green Channel', kde=True)
    #sns.histplot(stds[:, 2], color='blue', label='Blue Channel', kde=True)
    plt.title('St. Dev. of Pixel Intensity Distribution', fontsize=FONT_SIZE_TITLE)
    plt.xlabel('Standard Deviation', fontsize=FONT_SIZE_XLABEL)
    plt.ylabel('Frequency', fontsize=FONT_SIZE_YLABEL)
    plt.xticks(fontsize=FONT_SIZE_XTICKS)
    plt.yticks(fontsize=FONT_SIZE_YTICKS)
    plt.legend(fontsize=FONT_SIZE_TITLE)
    plt.tight_layout()
    plt.savefig('std_pixel_intensity_distribution_red.png')
    plt.show()



    plt.figure(figsize=(10, 6))
    #sns.histplot(stds[:, 0], color='red', label='Red Channel', kde=True)
    sns.histplot(stds[:, 1], color='green', label='Green Channel', kde=True)
    #sns.histplot(stds[:, 2], color='blue', label='Blue Channel', kde=True)
    plt.title('St. Dev. of Pixel Intensity Distribution', fontsize=FONT_SIZE_TITLE)
    plt.xlabel('Standard Deviation', fontsize=FONT_SIZE_XLABEL)
    plt.ylabel('Frequency', fontsize=FONT_SIZE_YLABEL)
    plt.xticks(fontsize=FONT_SIZE_XTICKS)
    plt.yticks(fontsize=FONT_SIZE_YTICKS)
    plt.legend(fontsize=FONT_SIZE_TITLE)
    plt.tight_layout()
    plt.savefig('std_pixel_intensity_distribution_green.png')
    plt.show()


    plt.figure(figsize=(10, 6))
    #sns.histplot(stds[:, 0], color='red', label='Red Channel', kde=True)
    #sns.histplot(stds[:, 1], color='green', label='Green Channel', kde=True)
    sns.histplot(stds[:, 2], color='blue', label='Blue Channel', kde=True)
    plt.title('St. Dev. of Pixel Intensity Distribution', fontsize=FONT_SIZE_TITLE)
    plt.xlabel('Standard Deviation', fontsize=FONT_SIZE_XLABEL)
    plt.ylabel('Frequency', fontsize=FONT_SIZE_YLABEL)
    plt.xticks(fontsize=FONT_SIZE_XTICKS)
    plt.yticks(fontsize=FONT_SIZE_YTICKS)
    plt.legend(fontsize=FONT_SIZE_TITLE)
    plt.tight_layout()
    plt.savefig('std_pixel_intensity_distribution_blue.png')
    plt.show()

# Analyze Client Data Distribution
def analyze_client_data_distribution(dataset, num_clients):
    """Analyze the distribution of data across clients."""
    client_data = partition_data(dataset, num_clients)
    client_counts = []

    for client_id, indices in client_data.items():
        class_counts = Counter([dataset[i][1] for i in indices])
        client_counts.append(class_counts)

    client_df = pd.DataFrame(client_counts).fillna(0).astype(int)
    client_df.index = [f'Client {i}' for i in range(num_clients)]

    # Plot client distributions
    client_df.plot(kind='bar', stacked=True, figsize=(8, 6), colormap='viridis')
    plt.title('Client Data Distribution', fontsize=FONT_SIZE_TITLE)
    plt.xlabel('Clients', fontsize=FONT_SIZE_XLABEL)
    plt.ylabel('Number of Images', fontsize=FONT_SIZE_YLABEL)
    plt.xticks(fontsize=FONT_SIZE_XTICKS, rotation=45)
    plt.yticks(fontsize=FONT_SIZE_YTICKS)
    plt.legend(fontsize=FONT_SIZE_TITLE, ncol=3,  framealpha=1.0)
    plt.tight_layout()
    plt.savefig('client_data_distribution.png')
    plt.show()

# Analyze Sample Images
def display_sample_images(root_dir, classes, num_samples=5):
    """Display a few sample images from each class."""
    plt.figure(figsize=(15, 10))
    for idx, clazz in enumerate(classes):
        class_dir = os.path.join(root_dir, clazz)
        sample_images = os.listdir(class_dir)[:num_samples]

        for j, img_name in enumerate(sample_images):
            img_path = os.path.join(class_dir, img_name)
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            plt.subplot(len(classes), num_samples, idx * num_samples + j + 1)
            plt.imshow(img)
            plt.title(clazz, fontsize=FONT_SIZE_XLABEL)
            plt.axis('off')

    plt.tight_layout()
    plt.savefig('sample_images.png')
    plt.show()

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

# Main Function for EDA
if __name__ == "__main__":
    # Initialize dataset
    classification_ds = ClassificationDataset(classification_root_dir)

    # Analyze class distribution
    print("Analyzing class distribution...")
    analyze_dataset_distribution(classification_root_dir, classes)

    # Analyze image dimensions
    print("Analyzing image dimensions...")
    analyze_image_dimensions(classification_root_dir, classes)

    # Analyze image statistics
    print("Analyzing image statistics...")
    analyze_image_statistics(classification_root_dir, classes)

    # Analyze client data distribution
    print("Analyzing client data distribution...")
    analyze_client_data_distribution(classification_ds, num_clients=5)

    # Display sample images
    print("Displaying sample images...")
    display_sample_images(classification_root_dir, classes)
