"""
Neural Network model for Quick Draw recognition.
This module implements an improved feedforward neural network using NumPy.
"""

import numpy as np
import os
import time

class NeuralNetwork:
    """
    An improved feedforward neural network with two hidden layers.
    
    Attributes:
        input_size (int): Number of input neurons (pixels in the image)
        hidden_size (int): Number of neurons in the first hidden layer
        second_hidden_size (int): Number of neurons in the second hidden layer
        output_size (int): Number of output neurons (categories)
        W1, W2, W3 (ndarray): Weight matrices
        b1, b2, b3 (ndarray): Bias vectors
    """
    
    def __init__(self, input_size, hidden_size, output_size, second_hidden_size=None):
        """
        Initialize a neural network with two hidden layers.
        
        Args:
            input_size (int): Number of input neurons (pixels in the image)
            hidden_size (int): Number of neurons in the first hidden layer
            output_size (int): Number of output neurons (categories)
            second_hidden_size (int, optional): Number of neurons in the second hidden layer
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.second_hidden_size = second_hidden_size or hidden_size // 2
        self.output_size = output_size
        
        # Initialize weights with He initialization (better for ReLU)
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0/input_size)
        self.b1 = np.zeros((1, hidden_size))
        
        self.W2 = np.random.randn(hidden_size, self.second_hidden_size) * np.sqrt(2.0/hidden_size)
        self.b2 = np.zeros((1, self.second_hidden_size))
        
        self.W3 = np.random.randn(self.second_hidden_size, output_size) * np.sqrt(2.0/self.second_hidden_size)
        self.b3 = np.zeros((1, output_size))
        
        # For tracking training statistics
        self.train_history = {
            'loss': [],
            'accuracy': []
        }
    
    def relu(self, x):
        """ReLU activation function: f(x) = max(0, x)"""
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        """Derivative of ReLU"""
        return np.where(x > 0, 1, 0)
    
    def softmax(self, x):
        """Softmax activation function for output layer, numerically stable."""
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def forward(self, X):
        """
        Forward pass through the network.
        
        Args:
            X (ndarray): Input data of shape (batch_size, input_size)
        
        Returns:
            ndarray: Output probabilities of shape (batch_size, output_size)
        """
        # First hidden layer
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = self.relu(self.Z1)
        
        # Second hidden layer
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = self.relu(self.Z2)
        
        # Output layer
        self.Z3 = np.dot(self.A2, self.W3) + self.b3
        self.A3 = self.softmax(self.Z3)
        
        return self.A3
    
    def predict(self, X):
        """
        Predict class probabilities for input data.
        
        Args:
            X (ndarray): Input data of shape (batch_size, input_size)
        
        Returns:
            ndarray: Predicted probabilities of shape (batch_size, output_size)
        """
        return self.forward(X)
    
    def compute_loss(self, y_pred, y_true):
        """Compute categorical cross-entropy loss."""
        m = y_true.shape[0]
        
        # Add small epsilon for numerical stability
        y_pred = np.clip(y_pred, 1e-10, 1 - 1e-10)
        
        if len(y_true.shape) == 1:  # If y_true is indices
            log_likelihood = -np.log(y_pred[range(m), y_true])
        else:  # If y_true is one-hot encoded
            log_likelihood = -np.sum(y_true * np.log(y_pred), axis=1)
            
        loss = np.sum(log_likelihood) / m
        return loss
    
    def backward(self, X, y, learning_rate=0.01):
        """Backward pass with gradient descent update."""
        m = X.shape[0]
        
        # Convert y to one-hot encoded if it's not already
        if len(y.shape) == 1:
            one_hot_y = np.zeros((m, self.output_size))
            one_hot_y[range(m), y] = 1
        else:
            one_hot_y = y
        
        # Gradient for output layer
        dZ3 = self.A3 - one_hot_y
        dW3 = np.dot(self.A2.T, dZ3) / m
        db3 = np.sum(dZ3, axis=0, keepdims=True) / m
        
        # Gradient for second hidden layer
        dA2 = np.dot(dZ3, self.W3.T)
        dZ2 = dA2 * self.relu_derivative(self.Z2)
        dW2 = np.dot(self.A1.T, dZ2) / m
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m
        
        # Gradient for first hidden layer
        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * self.relu_derivative(self.Z1)
        dW1 = np.dot(X.T, dZ1) / m
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m
        
        # Learning rate decay during training
        curr_learning_rate = learning_rate
        
        # Update weights and biases
        self.W3 -= curr_learning_rate * dW3
        self.b3 -= curr_learning_rate * db3
        self.W2 -= curr_learning_rate * dW2
        self.b2 -= curr_learning_rate * db2
        self.W1 -= curr_learning_rate * dW1
        self.b1 -= curr_learning_rate * db1
    
    def train(self, X, y, epochs=100, learning_rate=0.01, batch_size=32, verbose=True):
        """
        Train the network with mini-batch gradient descent.
        
        Args:
            X (ndarray): Training data of shape (n_samples, input_size)
            y (ndarray): Training labels of shape (n_samples,)
            epochs (int): Number of training epochs
            learning_rate (float): Learning rate for gradient descent
            batch_size (int): Mini-batch size
            verbose (bool): Whether to print training progress
        
        Returns:
            tuple: (losses, accuracies)
        """
        m = X.shape[0]
        losses = []
        accuracies = []
        
        for epoch in range(epochs):
            epoch_start_time = time.time()
            
            # Shuffle the data
            indices = np.random.permutation(m)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            # Mini-batch processing
            for i in range(0, m, batch_size):
                end = min(i + batch_size, m)
                X_batch = X_shuffled[i:end]
                y_batch = y_shuffled[i:end]
                
                # Forward pass
                y_pred = self.forward(X_batch)
                
                # Backward pass
                # Adjust learning rate, higher at beginning, lower at end
                adjusted_lr = learning_rate * (1.0 / (1.0 + 0.01 * epoch))
                self.backward(X_batch, y_batch, adjusted_lr)
            
            # Calculate loss and accuracy for monitoring
            y_pred = self.forward(X)
            loss = self.compute_loss(y_pred, y)
            losses.append(loss)
            
            y_pred_classes = np.argmax(y_pred, axis=1)
            accuracy = np.mean(y_pred_classes == y)
            accuracies.append(accuracy)
            
            # Save statistics
            self.train_history['loss'].append(loss)
            self.train_history['accuracy'].append(accuracy)
            
            if verbose and (epoch % 5 == 0 or epoch == epochs - 1):
                epoch_time = time.time() - epoch_start_time
                print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}, Time: {epoch_time:.2f}s")
        
        return losses, accuracies
    
    def save_weights(self, filepath):
        """
        Save model weights to a file.
        
        Args:
            filepath (str): Path to save the weights
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save weights and model configuration
        np.savez(filepath, 
                 W1=self.W1, 
                 b1=self.b1, 
                 W2=self.W2, 
                 b2=self.b2,
                 W3=self.W3,
                 b3=self.b3,
                 input_size=self.input_size,
                 hidden_size=self.hidden_size,
                 second_hidden_size=self.second_hidden_size,
                 output_size=self.output_size)
        
        print(f"Model saved to {filepath}")
    
    def load_weights(self, filepath):
        """
        Load model weights from a file.
        
        Args:
            filepath (str): Path to the weights file
        
        Returns:
            bool: True if loaded successfully, False otherwise
        """
        try:
            data = np.load(filepath, allow_pickle=True)
            
            # Check if the model has one or two hidden layers
            if 'W3' in data:
                # Load weights for two hidden layers
                self.W1 = data['W1']
                self.b1 = data['b1']
                self.W2 = data['W2']
                self.b2 = data['b2']
                self.W3 = data['W3']
                self.b3 = data['b3']
                
                # Update model configuration if available
                if 'input_size' in data and 'hidden_size' in data and 'output_size' in data:
                    self.input_size = data['input_size'].item()
                    self.hidden_size = data['hidden_size'].item()
                    self.second_hidden_size = data['second_hidden_size'].item()
                    self.output_size = data['output_size'].item()
            else:
                # Handle loading from older model with one hidden layer
                self.W1 = data['W1']
                self.b1 = data['b1']
                
                if 'input_size' in data and 'hidden_size' in data and 'output_size' in data:
                    self.input_size = data['input_size'].item()
                    self.hidden_size = data['hidden_size'].item()
                    self.output_size = data['output_size'].item()
                
                # Initialize second hidden layer
                self.second_hidden_size = self.hidden_size // 2
                self.W2 = np.random.randn(self.hidden_size, self.second_hidden_size) * np.sqrt(2.0/self.hidden_size)
                self.b2 = np.zeros((1, self.second_hidden_size))
                
                # Initialize output layer weights from old W2/b2
                self.W3 = data['W2']
                self.b3 = data['b2']
            
            print(f"Model loaded from {filepath}")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False

    @classmethod
    def load_model(cls, filepath):
        """
        Load a model from a file.
        
        Args:
            filepath (str): Path to the model file
        
        Returns:
            NeuralNetwork: Loaded model or None if loading failed
        """
        try:
            data = np.load(filepath, allow_pickle=True)
            
            # Get model configuration
            input_size = data['input_size'].item()
            
            # Check if it's a one or two hidden layer model
            if 'second_hidden_size' in data:
                hidden_size = data['hidden_size'].item()
                second_hidden_size = data['second_hidden_size'].item()
                output_size = data['output_size'].item()
                
                # Create model with two hidden layers
                model = cls(input_size, hidden_size, output_size, second_hidden_size)
                
                # Load weights
                model.W1 = data['W1']
                model.b1 = data['b1']
                model.W2 = data['W2']
                model.b2 = data['b2']
                model.W3 = data['W3']
                model.b3 = data['b3']
            else:
                # It's an older model with one hidden layer
                hidden_size = data['hidden_size'].item()
                output_size = data['output_size'].item()
                
                # Create new model with two hidden layers
                second_hidden_size = hidden_size // 2
                model = cls(input_size, hidden_size, output_size, second_hidden_size)
                
                # Load first layer weights
                model.W1 = data['W1']
                model.b1 = data['b1']
                
                # Initialize second hidden layer
                model.W2 = np.random.randn(hidden_size, second_hidden_size) * np.sqrt(2.0/hidden_size)
                model.b2 = np.zeros((1, second_hidden_size))
                
                # Use old output layer weights
                model.W3 = data['W2']
                model.b3 = data['b2']
            
            print(f"Model loaded from {filepath}")
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            return None