import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import threading
import queue
from pandastable import Table
from sklearn.metrics import mean_squared_error
from NeuronalNetwork import initialize_weights, backpropagation, feedforward_for_prediction
from DataManager import load_data
import pandas as pd
import warnings
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Global variables to store training and testing data, weights, epochs, and training errors
X_train, Y_train, X_test, Y_test, weights, epochs, training_errors = None, None, None, None, None, None, []

# A thread-safe queue for updating the GUI
gui_queue = queue.Queue()

# Initialize the final mean squared error to a default value
final_mse = 0

# Function to initialize the data and weights
def initialize(file_path, test_size=0.2, drop_columns=None, target_column='Close', 
               hidden_layer_sizes=[10], reshape_target=True):
    # Load the data from the specified file
    X_train, Y_train, X_test, Y_test = load_data(
        file_path, test_size, drop_columns, target_column, reshape_target
    )
    
    # Reshape Y_train and Y_test to 2D arrays if needed
    if reshape_target:
        Y_train = Y_train.reshape(-1, 1)
        Y_test = Y_test.reshape(-1, 1)
    
    # Initialize the weights for the neural network
    input_size = X_train.shape[1]
    output_size = Y_train.shape[1] if reshape_target else 1
    weights = initialize_weights(input_size, output_size, hidden_layer_sizes)
    
    return X_train, Y_train, X_test, Y_test, weights

# Function to train the neural network
def train(X_train, Y_train, weights, learning_rate=0.01, epochs=100):
    for epoch in range(epochs):
        # Perform backpropagation and calculate the mean squared error
        mse = backpropagation(weights, X_train, Y_train, learning_rate)
        # Print the MSE every 100 epochs
        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Training MSE: {mse}")

# Function to test the neural network
def test(X_test, Y_test, weights):
    # Use the trained weights to make predictions
    predictions = feedforward_for_prediction(weights, X_test)
    # Calculate and print the mean squared error for the test data
    mse = mean_squared_error(Y_test, predictions)
    print(f"Test MSE: {mse}")

# Function to load data and display it in a GUI
def Load_Data():
    # Load the data from the CSV file
    X_train, Y_train, X_test, Y_test = load_data("Microsoft_Stock.csv", 0.2, drop_columns=['Date'], reshape_target=False)

    # Check if data loading was successful
    if X_train is None:
        messagebox.showerror("Data Loading Failed", "Failed to load data.")
        return

    # Create a new tkinter window to display the data
    root_overview = tk.Tk()
    root_overview.title("Data Overview")

    # Add a frame to hold the table
    frame = tk.Frame(root_overview)
    frame.pack(fill=tk.BOTH, expand=True)

    # Combine training and testing data for display
    data_for_display = pd.concat([
        pd.DataFrame(X_train, columns=[f"Feature_{i}" for i in range(X_train.shape[1])]),
        pd.Series(Y_train, name='Y_train'),
        pd.DataFrame(X_test, columns=[f"Feature_{i}" for i in range(X_test.shape[1])]),
        pd.Series(Y_test, name='Y_test')
    ], axis=1)

    # Display the data in a pandastable frame
    pt = Table(frame, dataframe=data_for_display, showtoolbar=False, showstatusbar=False)
    pt.show()

    # Show a message box indicating successful data loading
    messagebox.showinfo("Data Loaded", "Data has been loaded successfully!")

# Set the file path and neural network parameters
file_path = "Microsoft_Stock.csv"
hidden_layer_sizes = [20, 35, 40]  # Define the architecture of the neural network
learning_rate = 0.002  # Set the learning rate for training
epochs = 9500  # Set the number of training epochs

# Initialize the data and weights
X_train, Y_train, X_test, Y_test, weights = initialize(
    file_path, drop_columns=['Date'], hidden_layer_sizes=hidden_layer_sizes
)

# Train the neural network
train(X_train, Y_train, weights, learning_rate, epochs)

# Test the neural network
test(X_test, Y_test, weights)

class NeuralNetworkGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Neural Network Configuration")
        
        # Create main frame
        main_frame = ttk.Frame(root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Initialize plot axes as instance variables
        self.ax1 = None
        self.ax2 = None
        
        # Epochs input
        ttk.Label(main_frame, text="Epochs:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.epochs_var = tk.StringVar(value="9500")
        self.epochs_entry = ttk.Entry(main_frame, textvariable=self.epochs_var)
        self.epochs_entry.grid(row=0, column=1, sticky=(tk.W, tk.E), pady=5)
        
        # Hidden layers input
        ttk.Label(main_frame, text="Hidden Layers (x,y,z):").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.hidden_layers_var = tk.StringVar(value="20,35,40")
        self.hidden_layers_entry = ttk.Entry(main_frame, textvariable=self.hidden_layers_var)
        self.hidden_layers_entry.grid(row=1, column=1, sticky=(tk.W, tk.E), pady=5)
        
        # Learning rate input
        ttk.Label(main_frame, text="Learning Rate:").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.learning_rate_var = tk.StringVar(value="0.002")
        self.learning_rate_entry = ttk.Entry(main_frame, textvariable=self.learning_rate_var)
        self.learning_rate_entry.grid(row=2, column=1, sticky=(tk.W, tk.E), pady=5)
        
        # Load Data button
        self.load_button = ttk.Button(main_frame, text="Load Data", command=Load_Data)
        self.load_button.grid(row=3, column=0, columnspan=2, pady=5)
        
        # Train button
        self.train_button = ttk.Button(main_frame, text="Train Network", command=self.start_training)
        self.train_button.grid(row=4, column=0, columnspan=2, pady=5)
        
    def start_training(self):
        try:
            # Get values from inputs
            epochs = int(self.epochs_var.get())
            hidden_layers = [int(x.strip()) for x in self.hidden_layers_var.get().split(',')]
            learning_rate = float(self.learning_rate_var.get())
            
            # Initialize the data and weights
            global X_train, Y_train, X_test, Y_test, weights
            X_train, Y_train, X_test, Y_test, weights = initialize(
                file_path, drop_columns=['Date'], hidden_layer_sizes=hidden_layers
            )
            
            # Train the network
            train(X_train, Y_train, weights, learning_rate, epochs)
            
            # Test the network
            test(X_test, Y_test, weights)
            
            messagebox.showinfo("Training Complete", "Neural network training has completed successfully!")
            
        except ValueError as e:
            messagebox.showerror("Input Error", "Please check your input values. Make sure they are in the correct format.")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")

# Replace the direct training code at the bottom with this:
if __name__ == "__main__":
    root = tk.Tk()
    app = NeuralNetworkGUI(root)
    root.mainloop()