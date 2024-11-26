import tkinter as tk
from tkinter import ttk, messagebox
import threading
import queue
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from NeuronalNetwork import initialize, backpropagation, feedforward_for_prediction  # Adjust import as needed
import pandas as pd
from pandastable import Table
from DataManager import load_data

# Create a queue for communication between threads
task_queue = queue.Queue()

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
        
        # Other GUI setup code...
        self.setup_gui(main_frame)

    def setup_gui(self, main_frame):
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
        self.load_button = ttk.Button(main_frame, text="Load Data", command=self.load_data)
        self.load_button.grid(row=3, column=0, columnspan=2, pady=5)
        
        # Train button
        self.train_button = ttk.Button(main_frame, text="Train Network", command=self.start_training)
        self.train_button.grid(row=4, column=0, columnspan=2, pady=5)

    def load_data(self):
        # Load the data from the CSV file
        X_train, Y_train, X_test, Y_test = load_data("Microsoft_Stock.csv", 0.2, drop_columns=['Date'], reshape_target=False)

        # Check if data loading was successful
        if X_train is None:
            messagebox.showerror("Data Loading Failed", "Failed to load data.")
            return  # Exit the function if data loading fails

        # Use X_train or other variables as needed
        # For example, you might want to store them as instance variables for later use
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.Y_test = Y_test

        # Create a new tkinter window to display the data
        root_overview = tk.Tk()
        root_overview.title("Data Overview")

        # Add a frame to hold the table
        frame = tk.Frame(root_overview)
        frame.pack(fill=tk.BOTH, expand=True)

        # Combine training and testing data for display
        data_for_display = pd.concat([
            pd.DataFrame(self.X_train, columns=[f"Feature_{i}" for i in range(self.X_train.shape[1])]),
            pd.Series(self.Y_train, name='Y_train'),
            pd.DataFrame(self.X_test, columns=[f"Feature_{i}" for i in range(self.X_test.shape[1])]),
            pd.Series(self.Y_test, name='Y_test')
        ], axis=1)

        # Display the data in a pandastable frame
        pt = Table(frame, dataframe=data_for_display, showtoolbar=False, showstatusbar=False)
        pt.show()

        # Show a message box indicating successful data loading
        messagebox.showinfo("Data Loaded", "Data has been loaded successfully!")

    def start_training(self):
        def train():
            try:
                # Get values from inputs
                epochs = int(self.epochs_var.get())
                hidden_layers = [int(x.strip()) for x in self.hidden_layers_var.get().split(',')]
                learning_rate = float(self.learning_rate_var.get())
                
                file_path = "Microsoft_Stock.csv"  # Adjust the path as needed
                
                # Initialize the data and weights
                global X_train, Y_train, X_test, Y_test, weights, training_errors
                X_train, Y_train, X_test, Y_test, weights = initialize(
                    file_path, drop_columns=['Date'], hidden_layer_sizes=hidden_layers
                )
                
                # Prepare for live plotting
                training_errors.clear()
                fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(10, 8))
                plt.ion()  # Turn on interactive mode

                # Training loop
                for epoch in range(epochs):
                    mse = backpropagation(weights, X_train, Y_train, learning_rate)
                    training_errors.append(mse)
                    if (epoch + 1) % 100 == 0:
                        task_queue.put(self.update_plot)  # Add plot update task to the queue

                # Finalize the plots
                plt.ioff()  # Turn off interactive mode
                plt.show(block=False)  # Show the plot without blocking
                plt.close(fig)  # Close the plot window after showing
                
                # Notify training completion
                task_queue.put(lambda: messagebox.showinfo("Training Complete", "Neural network training has completed successfully!"))
                
            except ValueError as e:
                task_queue.put(lambda: messagebox.showerror("Input Error", "Please check your input values. Make sure they are in the correct format."))
            except Exception as e:
                task_queue.put(lambda: messagebox.showerror("Error", f"An error occurred: {str(e)}"))

        # Start the training in a new thread to keep the GUI responsive
        training_thread = threading.Thread(target=train)
        training_thread.start()

        # Start the queue management
        self.manage_queue()

    def update_plot(self):
        self.ax1.clear()
        self.ax1.plot(training_errors, label='Training MSE')
        self.ax1.set_title('Training Progress')
        self.ax1.set_xlabel('Epoch')
        self.ax1.set_ylabel('MSE')
        self.ax1.legend()
        self.ax1.grid(True)
        
        predictions = feedforward_for_prediction(weights, X_test)
        self.ax2.clear()
        self.ax2.plot(Y_test, label='Original Data')
        self.ax2.plot(predictions, label='Predicted Data', linestyle='--')
        self.ax2.set_title('Original vs Predicted Data')
        self.ax2.set_xlabel('Sample')
        self.ax2.set_ylabel('Value')
        self.ax2.legend()
        self.ax2.grid(True)
        
        plt.tight_layout()
        plt.draw()

    def manage_queue(self):
        try:
            while not task_queue.empty():
                task = task_queue.get_nowait()
                task()  # Execute the task
        except queue.Empty:
            pass
        finally:
            # Schedule the next queue check
            self.root.after(100, self.manage_queue)
