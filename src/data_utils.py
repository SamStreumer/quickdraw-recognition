"""
Data loading and preprocessing utilities for Quick Draw dataset.
"""

import numpy as np
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def load_category_data(category, data_dir='data', max_samples=None):
    """
    Load data for a single category from the Quick Draw dataset.
    
    Args:
        category (str): Category name (e.g. 'apple', 'car')
        data_dir (str): Directory containing the dataset files
        max_samples (int, optional): Maximum number of samples to load
    
    Returns:
        ndarray: Images data of shape (n_samples, 28, 28)
    """
    try:
        # Construct the file path
        file_path = os.path.join(data_dir, f"{category}.npy")
        
        # Check if file exists
        if not os.path.exists(file_path):
            print(f"Warning: {file_path} does not exist.")
            return None
        
        # Load the data
        data = np.load(file_path)
        
        # Limit the number of samples if specified
        if max_samples is not None and max_samples < len(data):
            data = data[:max_samples]
        
        print(f"Loaded {len(data)} samples for '{category}'")
        return data
    
    except Exception as e:
        print(f"Error loading {category}: {e}")
        return None

def load_dataset(categories, data_dir='data', max_samples_per_category=None, test_size=0.2, random_state=42):
    """
    Load and prepare the Quick Draw dataset for multiple categories.
    
    Args:
        categories (list): List of category names to load
        data_dir (str): Directory containing the dataset files
        max_samples_per_category (int, optional): Maximum samples per category
        test_size (float): Proportion of the dataset to include in the test split
        random_state (int): Random seed for reproducibility
    
    Returns:
        tuple: (X_train, X_test, y_train, y_test, categories)
            - X_train, X_test: Training and test images (normalized to 0-1)
            - y_train, y_test: Training and test labels (indices)
            - categories: List of successfully loaded categories
    """
    X = []  # Images
    y = []  # Labels
    loaded_categories = []  # Successfully loaded categories
    
    # Load each category
    for i, category in enumerate(categories):
        data = load_category_data(category, data_dir, max_samples_per_category)
        
        if data is not None and len(data) > 0:
            X.append(data)
            y.append(np.full(len(data), i))
            loaded_categories.append(category)
    
    # Check if any categories were loaded
    if not X:
        raise ValueError("No data was loaded. Check category names and data directory.")
    
    # Combine all categories
    X = np.vstack(X)
    y = np.hstack(y)
    
    # Normalize pixel values to 0-1
    X = X.astype('float32') / 255.0
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"Dataset prepared: {len(X_train)} training samples, {len(X_test)} test samples")
    print(f"Categories: {', '.join(loaded_categories)}")
    
    return X_train, X_test, y_train, y_test, loaded_categories

def prepare_data_for_training(X_train, X_test):
    """
    Prepare the data for training by reshaping it to 2D arrays.
    
    Args:
        X_train (ndarray): Training images of shape (n_samples, 28, 28)
        X_test (ndarray): Test images of shape (n_samples, 28, 28)
    
    Returns:
        tuple: (X_train_flat, X_test_flat)
            - X_train_flat: Flattened training images (n_samples, 784)
            - X_test_flat: Flattened test images (n_samples, 784)
    """
    # Reshape images to vectors
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    
    return X_train_flat, X_test_flat

def visualize_samples(X, y, categories, samples_per_category=5, figsize=(15, 10)):
    """
    Visualize sample images from the dataset.
    
    Args:
        X (ndarray): Images data of shape (n_samples, 28, 28) or (n_samples, 784)
        y (ndarray): Labels of shape (n_samples,)
        categories (list): List of category names
        samples_per_category (int): Number of samples to show per category
        figsize (tuple): Figure size
    """
    n_categories = len(categories)
    fig, axes = plt.subplots(n_categories, samples_per_category, figsize=figsize)
    
    # Reshape images if they're flattened
    if len(X.shape) == 2 and X.shape[1] == 784:
        X_display = X.reshape(-1, 28, 28)
    else:
        X_display = X
    
    for i, category in enumerate(categories):
        # Get indices of samples from this category
        indices = np.where(y == i)[0]
        
        # Randomly select samples
        selected_indices = np.random.choice(indices, min(samples_per_category, len(indices)), replace=False)
        
        for j, idx in enumerate(selected_indices):
            if n_categories > 1:
                ax = axes[i, j]
            else:
                ax = axes[j]
            
            ax.imshow(X_display[idx], cmap='gray')
            ax.axis('off')
            
            if j == 0:
                ax.set_title(category)
    
    plt.tight_layout()
    plt.show()

def convert_drawing_to_image(drawing, size=28):
    """
    Convert a drawing (represented as a 2D array of 0s and 1s or grayscale values)
    to a properly formatted image for the neural network.
    
    Args:
        drawing (ndarray): Drawing data of shape (height, width)
        size (int): Target size (both width and height)
    
    Returns:
        ndarray: Processed image of shape (size, size)
    """
    # Resize if necessary
    if drawing.shape[0] != size or drawing.shape[1] != size:
        # Simple resize by copying values (not interpolation)
        h_ratio = drawing.shape[0] / size
        w_ratio = drawing.shape[1] / size
        
        resized = np.zeros((size, size))
        for i in range(size):
            for j in range(size):
                orig_i = min(int(i * h_ratio), drawing.shape[0] - 1)
                orig_j = min(int(j * w_ratio), drawing.shape[1] - 1)
                resized[i, j] = drawing[orig_i, orig_j]
        
        drawing = resized
    
    # Ensure values are between 0 and 1
    drawing = drawing.astype(np.float32)
    if drawing.max() > 1.0:
        drawing /= 255.0
    
    return drawing