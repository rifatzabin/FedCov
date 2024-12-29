import os  # Used for interacting with the operating system, like navigating directories
import matplotlib.pyplot as plt  # Used for creating visualizations
import seaborn as sns  # Provides a high-level interface for drawing attractive statistical graphics
import pandas as pd  # Provides powerful data structures for data analysis
import cv2  # OpenCV for image operations
from collections import Counter, defaultdict  # Specialized container datatypes
import numpy as np  # Fundamental package for scientific computing with Python
from torch.utils.data import Dataset  # PyTorch's dataset class for handling datasets

# Define global font sizes for plots to ensure readability and consistency
FONT_SIZE_XLABEL = 27
FONT_SIZE_YLABEL = 27
FONT_SIZE_XTICKS = 25
FONT_SIZE_YTICKS = 25
FONT_SIZE_TITLE = 27

# Define the directory where images are stored
classification_root_dir = '/home/foysal/Documents/covid19_dataset'
classes = ['COVID', 'Pneumonia', 'Normal']  # Classes in the dataset

# Function to partition dataset into multiple clients for federated learning scenarios
def partition_data(dataset, num_clients):
    client_data = defaultdict(list)  # Default dictionary to store data for each client
    labels = np.array([dataset[i][1] for i in range(len(dataset))])  # Extract labels from dataset
    unique_labels = np.unique(labels)  # Find all unique labels

    for label in unique_labels:
        label_indices = np.where(labels == label)[0]  # Find indices for each label
        np.random.shuffle(label_indices)  # Shuffle indices to randomize data distribution
        split = np.array_split(label_indices, num_clients)  # Split indices into parts for each client
        for i, indices in enumerate(split):
            client_data[i].extend(indices)  # Assign indices to each client

    for client in client_data:
        np.random.shuffle(client_data[client])  # Shuffle data for each client again

    return client_data

# Function to analyze and visualize class distribution in the dataset
def analyze_dataset_distribution(root_dir, classes):
    class_counts = {}  # Dictionary to hold count of images per class
    for clazz in classes:
        class_dir = os.path.join(root_dir, clazz)  # Path to class directory
        count = len(os.listdir(class_dir))  # Count number of files in each class directory
        class_counts[clazz] = count  # Store count in dictionary

    plt.figure(figsize=(8, 6))  # Create a figure for plotting
    sns.barplot(x=list(class_counts.keys()), y=list(class_counts.values()), palette='viridis')  # Create a barplot
    plt.title('Class Distribution', fontsize=FONT_SIZE_TITLE)  # Title of the plot
    plt.xlabel('Class', fontsize=FONT_SIZE_XLABEL)  # X-axis label
    plt.ylabel('Number of Images', fontsize=FONT_SIZE_YLABEL)  # Y-axis label
    plt.xticks(fontsize=FONT_SIZE_XTICKS)  # Font size for X-axis tick labels
    plt.yticks(fontsize=FONT_SIZE_YTICKS)  # Font size for Y-axis tick labels
    plt.tight_layout()  # Adjust layout to make it neat
    plt.savefig('class_distribution.png')  # Save the figure as PNG file
    plt.show()  # Display the plot

# Function to analyze and visualize image dimensions and aspect ratios
def analyze_image_dimensions(root_dir, classes):
    dimensions = []  # List to store dimensions of all images
    aspect_ratios = []  # List to store aspect ratios of all images

    for clazz in classes:
        class_dir = os.path.join(root_dir, clazz)  # Path to class directory
        for img_name in os.listdir(class_dir):  # Iterate over all images in directory
            img_path = os.path.join(class_dir, img_name)  # Full path to image
            img = cv2.imread(img_path)  # Read image
            if img is not None:
                height, width = img.shape[:2]  # Get image height and width
                dimensions.append((height, width))  # Append dimensions to list
                aspect_ratios.append(width / height)  # Calculate and append aspect ratio

    dims_df = pd.DataFrame(dimensions, columns=['Height', 'Width'])  # Create a DataFrame from dimensions
    dims_df['Aspect Ratio'] = aspect_ratios  # Add aspect ratios to DataFrame

    # Plotting dimensions
    plt.figure(figsize=(8, 6))
    sns.histplot(dims_df['Height'], kde=True, color='blue', label='Height', bins=30)  # Histogram for heights
    sns.histplot(dims_df['Width'], kde=True, color='orange', label='Width', bins=50)  # Histogram for widths
    plt.title('Image Dimensions Distribution', fontsize=FONT_SIZE_TITLE)
    plt.xlabel('Pixels', fontsize=FONT_SIZE_XLABEL)
    plt.ylabel('Frequency', fontsize=FONT_SIZE_YLABEL)
    plt.xticks(fontsize=FONT_SIZE_XTICKS)
    plt.yticks(fontsize=FONT_SIZE_YTICKS)
    plt.legend(fontsize=FONT_SIZE_TITLE)
    plt.tight_layout()
    plt.savefig('image_dimensions_distribution.png')
    plt.show()

    # Plotting aspect ratios
    plt.figure(figsize=(8, 6))
    sns.histplot(dims_df['Aspect Ratio'], kde=True, color='green', bins=30)  # Histogram for aspect ratios
    plt.title('Aspect Ratio Distribution', fontsize=FONT_SIZE_TITLE)
    plt.xlabel('Aspect Ratio', fontsize=FONT_SIZE_XLABEL)
    plt.ylabel('Frequency', fontsize=FONT_SIZE_YLABEL)
    plt.xticks(fontsize=FONT_SIZE_XTICKS)
    plt.yticks(fontsize=FONT_SIZE_YTICKS)
    plt.tight_layout()
    plt.savefig('aspect_ratio_distribution.png')
    plt.show()

# Function to analyze and visualize the mean and standard deviation of pixel intensities
def analyze_image_statistics(root_dir, classes):
    means = []  # List to store mean intensities of images
    stds = []  # List to store standard deviations of images

    for clazz in classes:
        class_dir = os.path.join(root_dir, clazz)  # Path to class directory
        for img_name in os.listdir(class_dir):  # Iterate over all images in directory
            img_path = os.path.join(class_dir, img_name)  # Full path to image
            img = cv2.imread(img_path)  # Read image
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert image to RGB
                img = img / 255.0  # Normalize pixel values to [0, 1]
                means.append(np.mean(img, axis=(0, 1)))  # Compute mean intensity and add to list
                stds.append(np.std(img, axis=(0, 1)))  # Compute standard deviation and add to list

    means = np.array(means)  # Convert list to numpy array for easier handling
    stds = np.array(stds)  # Convert list to numpy array for easier handling

    # Plotting mean intensities for the Red channel
    plt.figure(figsize=(10, 6))
    sns.histplot(means[:, 0], color='red', label='Red Channel', kde=True)  # Histogram for red channel mean intensities
    plt.title('Mean Pixel Intensity Distribution', fontsize=FONT_SIZE_TITLE)
    plt.xlabel('Mean Intensity', fontsize=FONT_SIZE_XLABEL)
    plt.ylabel('Frequency', fontsize=FONT_SIZE_YLABEL)
    plt.xticks(fontsize=FONT_SIZE_XTICKS)
    plt.yticks(fontsize=FONT_SIZE_YTICKS)
    plt.legend(fontsize=FONT_SIZE_TITLE)
    plt.tight_layout()
    plt.savefig('mean_pixel_intensity_distribution_red.png')
    plt.show()

    # Additional plots for other channels and standard deviations can be generated in a similar manner

# Function to display sample images from each class
def display_sample_images(root_dir, classes, num_samples=5):
    plt.figure(figsize=(15, 10))  # Create a large figure to hold all subplots
    for idx, clazz in enumerate(classes):  # Iterate over each class
        class_dir = os.path.join(root_dir, clazz)  # Directory for the class
        sample_images = os.listdir(class_dir)[:num_samples]  # Get a few sample images from the directory

        for j, img_name in enumerate(sample_images):  # Iterate over each sample image
            img_path = os.path.join(class_dir, img_name)  # Path to the image file
            img = cv2.imread(img_path)  # Read the image
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB

            plt.subplot(len(classes), num_samples, idx * num_samples + j + 1)  # Create a subplot for each image
            plt.imshow(img)  # Display the image
            plt.title(clazz, fontsize=FONT_SIZE_XLABEL)  # Set the title to the class name
            plt.axis('off')  # Turn off axis labels

    plt.tight_layout()  # Adjust layout to prevent overlap
    plt.savefig('sample_images.png')  # Save the figure as a PNG file
    plt.show()  # Display the figure

# Dataset class to handle data loading and transformation for PyTorch
class ClassificationDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir  # Root directory for the dataset
        self.transform = transform  # Transformations to be applied to each image
        self.classes = ['COVID', 'Pneumonia', 'Normal']  # Classes in the dataset
        self.data = []  # List to store the data
        self.class_to_id = {clazz: i for i, clazz in enumerate(self.classes)}  # Map class names to numeric labels

        for clazz in self.classes:  # Iterate over each class
            class_dir = os.path.join(root_dir, clazz)  # Path to the class directory
            class_id = self.class_to_id[clazz]  # Get the class ID
            for img in os.listdir(class_dir):  # Iterate over each image in the class directory
                img_path = os.path.join(class_dir, img)  # Path to the image file
                self.data.append((img_path, class_id))  # Append the image path and class ID to the data list

    def __len__(self):
        return len(self.data)  # Return the total number of items in the dataset

    def __getitem__(self, id):
        img_path, label = self.data[id]  # Get the image path and label for the given index
        img = cv2.imread(img_path)  # Read the image
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
        if self.transform:
            transform = self.transform(image=img)  # Apply transformations
            img = transform['image']  # Get the transformed image
        return img, label  # Return the image and label

# Main function to execute the exploratory data analysis (EDA)
if __name__ == "__main__":
    classification_ds = ClassificationDataset(classification_root_dir)  # Create an instance of the dataset

    # Perform various analyses
    print("Analyzing class distribution...")
    analyze_dataset_distribution(classification_root_dir, classes)

    print("Analyzing image dimensions...")
    analyze_image_dimensions(classification_root_dir, classes)

    print("Analyzing image statistics...")
    analyze_image_statistics(classification_root_dir, classes)

    print("Analyzing client data distribution...")
    analyze_client_data_distribution(classification_ds, num_clients=5)

    print("Displaying sample images...")
    display_sample_images(classification_root_dir, classes)
