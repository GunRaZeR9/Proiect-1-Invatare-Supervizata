import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# Function to normalize data using MinMaxScaler
def normalize(data):
    scaler = MinMaxScaler()
    return pd.DataFrame(scaler.fit_transform(data), index=data.index, columns=data.columns)

# Function to load and preprocess data from a CSV file
def load_data(file_path, test_size, drop_columns=None, target_column='Close', random_state=None,
              reshape_target=False):
    try:
        # Load the data from the CSV file
        data = pd.read_csv(file_path)

        # Drop specified columns if any
        if drop_columns is not None:
            data = data.drop(drop_columns, axis=1)

        # Ensure the target column exists in the data
        if target_column not in data.columns:
            raise ValueError(f"Target column '{target_column}' not found in data.")

        # Convert all data to float type
        data = data.astype(float)

        # Normalize the entire dataset
        normalized_data = normalize(data)

        # Split the data into features and target
        x = normalized_data.drop(target_column, axis=1)
        y = normalized_data[target_column]

        # Split the data into training and testing sets
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state)

        # Optionally reshape y_train and y_test to ensure they are 2-dimensional
        if reshape_target:
            y_train = np.array(y_train).reshape(-1, 1)
            y_test = np.array(y_test).reshape(-1, 1)

        return np.array(x_train), np.array(y_train), np.array(x_test), np.array(y_test)

    except Exception as e:
        # Print an error message if data loading fails
        print(f"An error occurred: {e}")
        return None