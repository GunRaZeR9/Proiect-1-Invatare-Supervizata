import numpy as np
from sklearn.metrics import mean_squared_error

# Function to initialize weights for the neural network
def initialize_weights(input_size, output_size, hidden_layer_sizes):
    # Set a random seed for reproducibility
    np.random.seed(42)
    # Define the sizes of each layer in the network
    layer_sizes = [input_size] + hidden_layer_sizes + [output_size]
    weights = []
    # Initialize weights for each layer using a scaled normal distribution
    for i in range(len(layer_sizes) - 1):
        std_dev = np.sqrt(2 / (layer_sizes[i] + layer_sizes[i + 1]))
        weights.append(np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * std_dev)
    return weights

# Activation function: ReLU (Rectified Linear Unit)
def relu(x):
    return np.maximum(0, x)

# Derivative of the ReLU activation function
def relu_derivative(x):
    return np.where(x > 0, 1, 0)

# Function to perform the feedforward process in the network
def feedforward(weights, input_data):
    activations = [input_data]
    # Pass the input data through each layer of the network
    for w in weights:
        input_data = relu(np.dot(input_data, w))
        activations.append(input_data)
    return activations

# Function to perform backpropagation for supervised learning
def backpropagation(weights, input_data, output_data, learning_rate):
    # Perform feedforward to get activations
    activations = feedforward(weights, input_data)
    output = activations[-1]
    # Calculate the error in the output layer
    output_error = 2 * (output - output_data) / output_data.shape[0]
    # Calculate the delta for the output layer
    delta = output_error * relu_derivative(output)
    # Update weights for each layer starting from the last
    for i in reversed(range(len(weights))):
        layer_input = activations[i]
        weight_gradient = np.dot(layer_input.T, delta)
        weights[i] -= learning_rate * weight_gradient
        if i > 0:
            delta = np.dot(delta, weights[i].T) * relu_derivative(layer_input)
    # Calculate and return the mean squared error
    mse = mean_squared_error(output_data, output)
    return mse

# Function to perform feedforward for prediction purposes
def feedforward_for_prediction(weights, input_data):
    # Pass the input data through each layer of the network
    for w in weights:
        input_data = relu(np.dot(input_data, w))
    return input_data
