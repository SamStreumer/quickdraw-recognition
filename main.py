"""
Main entry point for the Quick Draw Recognition application.
"""

import os
import sys
import argparse

# Add the project directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.model import NeuralNetwork
from src.app import DrawingApp
from src.data_utils import load_dataset, prepare_data_for_training

def main():
    """
    Main function to run the application.
    """
    parser = argparse.ArgumentParser(description='Quick Draw Recognition')
    parser.add_argument('--train', action='store_true', help='Train the model before running the app')
    parser.add_argument('--model', type=str, default='models/quickdraw_model.npz',
                       help='Path to the model file')
    parser.add_argument('--categories', type=str, nargs='+',
                       default=['apple', 'banana', 'car', 'cat', 'dog', 'fish', 
                                'house', 'tree', 'bicycle', 'airplane', 'book',
                                'clock', 'flower', 'guitar', 'hat', 'moon',
                                'pizza', 'star', 'sun', 'umbrella'],
                       help='Categories to recognize')
    parser.add_argument('--samples', type=int, default=10000,
                       help='Maximum samples per category for training')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--width', type=int, default=1024,
                       help='Initial window width')
    parser.add_argument('--height', type=int, default=768,
                       help='Initial window height')
    parser.add_argument('--hidden-size', type=int, default=256,
                       help='Number of neurons in the first hidden layer')
    parser.add_argument('--second-hidden-size', type=int, default=128,
                       help='Number of neurons in the second hidden layer')
    
    args = parser.parse_args()
    
    # Create model directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Check for existing model
    model_exists = os.path.exists(args.model)
    
    # Load or train the model
    if args.train or not model_exists:
        print("Training a new model...")
        train_model(args.model, args.categories, args.samples, args.epochs, 
                   args.hidden_size, args.second_hidden_size)
    
    # Load the model
    if os.path.exists(args.model):
        model = NeuralNetwork.load_model(args.model)
        if model is None:
            print(f"Failed to load model from {args.model}")
            return
    else:
        print(f"Model file {args.model} not found. Please train the model first.")
        return
    
    # Run the application
    app = DrawingApp(model=model, categories=args.categories, width=args.width, height=args.height)
    app.run()

def train_model(model_path, categories, max_samples_per_category, epochs, hidden_size, second_hidden_size):
    """
    Train the neural network model.
    
    Args:
        model_path (str): Path to save the trained model
        categories (list): List of categories to recognize
        max_samples_per_category (int): Maximum samples per category
        epochs (int): Number of training epochs
        hidden_size (int): Number of neurons in the first hidden layer
        second_hidden_size (int): Number of neurons in the second hidden layer
    """
    print(f"Loading dataset for categories: {', '.join(categories)}...")
    
    # Load and prepare the dataset
    try:
        X_train, X_test, y_train, y_test, loaded_categories = load_dataset(
            categories, max_samples_per_category=max_samples_per_category
        )
        
        # Prepare data for training (flatten the images)
        X_train_flat, X_test_flat = prepare_data_for_training(X_train, X_test)
        
        # Create and train the model
        input_size = X_train_flat.shape[1]  # 784 for 28x28 images
        output_size = len(loaded_categories)
        
        print(f"Creating model with {input_size} input neurons, "
              f"{hidden_size} & {second_hidden_size} hidden neurons, and {output_size} output neurons")
        
        model = NeuralNetwork(input_size, hidden_size, output_size, second_hidden_size)
        
        print(f"Training for {epochs} epochs...")
        model.train(X_train_flat, y_train, epochs=epochs, learning_rate=0.005, batch_size=128)
        
        # Evaluate on test set
        y_pred = model.predict(X_test_flat)
        y_pred_classes = np.argmax(y_pred, axis=1)
        accuracy = np.mean(y_pred_classes == y_test)
        
        print(f"Test accuracy: {accuracy:.4f}")
        
        # Save the model
        model.save_weights(model_path)
        print(f"Model saved to {model_path}")
        
        return model
    
    except Exception as e:
        print(f"Error training model: {e}")
        return None

if __name__ == "__main__":
    # Add missing import for numpy if needed during model evaluation
    try:
        import numpy as np
    except ImportError:
        print("Error: NumPy is required. Please install it with 'pip install numpy'.")
        sys.exit(1)
    
    main()