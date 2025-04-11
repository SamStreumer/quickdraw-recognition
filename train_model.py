"""
Script for training the Quick Draw recognition model.
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Add the project directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.model import NeuralNetwork
from src.data_utils import load_dataset, prepare_data_for_training, visualize_samples

def train_model(categories, output_dir='models', model_name='quickdraw_model.npz',
               max_samples=10000, epochs=100, learning_rate=0.005, batch_size=128,
               hidden_size=256, second_hidden_size=128, visualize=True):
    """
    Train a neural network model on the Quick Draw dataset.
    
    Args:
        categories (list): List of categories to recognize
        output_dir (str): Directory to save the model
        model_name (str): Name of the model file
        max_samples (int): Maximum samples per category
        epochs (int): Number of training epochs
        learning_rate (float): Learning rate for optimization
        batch_size (int): Mini-batch size for training
        hidden_size (int): Number of neurons in the first hidden layer
        second_hidden_size (int): Number of neurons in the second hidden layer
        visualize (bool): Whether to visualize sample images and training curves
    
    Returns:
        NeuralNetwork: The trained model
    """
    print(f"Loading dataset for categories: {', '.join(categories)}...")
    
    # Load and prepare the dataset with more samples per category
    X_train, X_test, y_train, y_test, loaded_categories = load_dataset(
        categories, max_samples_per_category=max_samples
    )
    
    # Visualize sample images
    if visualize:
        print("Visualizing sample images from the dataset...")
        visualize_samples(X_train, y_train, loaded_categories)
    
    # Prepare data for training (flatten the images)
    X_train_flat, X_test_flat = prepare_data_for_training(X_train, X_test)
    
    # Create the model with two hidden layers
    input_size = X_train_flat.shape[1]  # 784 for 28x28 images
    output_size = len(loaded_categories)
    
    print(f"Creating model with {input_size} input neurons, {hidden_size} & {second_hidden_size} hidden neurons, and {output_size} output neurons...")
    
    model = NeuralNetwork(input_size, hidden_size, output_size, second_hidden_size)
    
    # Train the model
    print(f"Training model for {epochs} epochs...")
    losses, accuracies = model.train(
        X_train_flat, y_train, 
        epochs=epochs, 
        learning_rate=learning_rate, 
        batch_size=batch_size
    )
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the model
    model_path = os.path.join(output_dir, model_name)
    model.save_weights(model_path)
    
    # Evaluate on test set
    y_pred = model.predict(X_test_flat)
    y_pred_classes = np.argmax(y_pred, axis=1)
    test_accuracy = np.mean(y_pred_classes == y_test)
    
    print(f"Test accuracy: {test_accuracy:.4f}")
    
    # Visualize training curves and confusion matrix
    if visualize:
        # Plot training curves
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(losses)
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(accuracies)
        plt.title('Training Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'training_curves.png'))
        plt.show()
        
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        cm = confusion_matrix(y_test, y_pred_classes)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=loaded_categories, 
                   yticklabels=loaded_categories)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
        plt.show()
        
        # Show some example predictions
        plt.figure(figsize=(15, 8))
        for i in range(10):
            plt.subplot(2, 5, i+1)
            idx = np.random.choice(len(X_test))
            # Reshape image for display if needed
            if len(X_test.shape) == 2 and X_test.shape[1] == 784:
                display_img = X_test[idx].reshape(28, 28)
            else:
                display_img = X_test[idx]
            plt.imshow(display_img, cmap='gray')
            pred = loaded_categories[y_pred_classes[idx]]
            true = loaded_categories[y_test[idx]]
            plt.title(f"Pred: {pred}\nTrue: {true}", 
                     color='green' if pred == true else 'red')
            plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'example_predictions.png'))
        plt.show()
    
    return model

def main():
    """
    Main function for training the model from command line.
    """
    parser = argparse.ArgumentParser(description='Train Quick Draw recognition model')
    parser.add_argument('--categories', type=str, nargs='+',
                       default=['apple', 'banana', 'car', 'cat', 'dog', 'fish', 
                               'house', 'tree', 'bicycle', 'airplane', 'book',
                               'clock', 'flower', 'guitar', 'hat', 'moon',
                               'pizza', 'star', 'sun', 'umbrella'],
                       help='Categories to train on')
    parser.add_argument('--output-dir', type=str, default='models',
                       help='Directory to save the model')
    parser.add_argument('--model-name', type=str, default='quickdraw_model.npz',
                       help='Name of the model file')
    parser.add_argument('--max-samples', type=int, default=10000,
                       help='Maximum samples per category')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--learning-rate', type=float, default=0.005,
                       help='Learning rate for optimization')
    parser.add_argument('--batch-size', type=int, default=128,
                       help='Mini-batch size for training')
    parser.add_argument('--hidden-size', type=int, default=256,
                       help='Number of neurons in the first hidden layer')
    parser.add_argument('--second-hidden-size', type=int, default=128,
                       help='Number of neurons in the second hidden layer')
    parser.add_argument('--no-visualize', action='store_true',
                       help='Disable visualization of samples and training curves')
    
    args = parser.parse_args()
    
    train_model(
        categories=args.categories,
        output_dir=args.output_dir,
        model_name=args.model_name,
        max_samples=args.max_samples,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        hidden_size=args.hidden_size,
        second_hidden_size=args.second_hidden_size,
        visualize=not args.no_visualize
    )

if __name__ == "__main__":
    # Check for required packages
    try:
        import matplotlib
        import seaborn
    except ImportError:
        print("Error: This script requires matplotlib and seaborn for visualization.")
        print("Please install them with: pip install matplotlib seaborn")
        sys.exit(1)
    
    main()