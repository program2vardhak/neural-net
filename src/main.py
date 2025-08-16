import numpy as np
import pandas as pd

# Load dataset
df = pd.read_csv('Data/mnist_train.csv')
X = df.iloc[:, 1:].to_numpy()  # Features (784 pixels)
y = df.iloc[:, 0].to_numpy()   # Labels (digits 0-9)

# Shuffle before splitting to cause randmness while training to become more accurate
indices = np.random.permutation(len(df))
X = X[indices]
y = y[indices]

# Split into train (80%) and validation (20%)
train_size = int(0.8 * len(df))
X_train = X[:train_size]
y_train = y[:train_size]
X_val = X[train_size:]
y_val = y[train_size:]

# Transpose so each column is one training example
X_train = X_train.T
X_val = X_val.T

# (make y shape (1,m) instead of (m,))
y_train = y_train.reshape(1, -1)
y_val = y_val.reshape(1, -1)

# Setting number of neurons in each layer
layer_dims = [784, 128, 64, 10]

"""
Initializing Parameters and Bias for the layers
"""
def initialize_parameters(layer_dims):
    np.random.seed(42)
    parameters = {}
    L = len(layer_dims)  # number of layers in network

    for i in range(1, L):
        parameters[f"W{i}"] = np.random.randn(layer_dims[i], layer_dims[i-1]) * 0.01
        parameters[f"b{i}"] = np.zeros((layer_dims[i], 1))  # bias starts at 0

    return parameters


# Example: initialize weights
parameters = initialize_parameters(layer_dims)
m=X_train.shape()
"""
Applying Forward Propogation

"""
def linear_forward(A_prev, W, b):
    Z = np.dot(W, A_prev) + b
    cache = (A_prev, W, b, Z)  # Save values for backprop later
    return Z, cache

def Leaky_relu(Z,k=0.01):   #Leaky Relu to prevent dead neurons
    return np.maximum(k*Z, Z)

def softmax(Z):
    expZ = np.exp(Z - np.max(Z, axis=0, keepdims=True))  # stable softmax
    return expZ / np.sum(expZ, axis=0, keepdims=True)

def forward_propagation(X, parameters):
    caches = []   # store all (A_prev, W, b, Z) for backprop
    A = X
    L = len(parameters) // 2   # number of layers

    for l in range(1, L):  # hidden layers (ReLU)
        W = parameters[f"W{l}"]
        b = parameters[f"b{l}"]
        Z, cache = linear_forward(A, W, b)
        A = Leaky_relu(Z)
        caches.append(cache)

    # Final layer (softmax for classification)
    W = parameters[f"W{L}"]
    b = parameters[f"b{L}"]
    Z, cache = linear_forward(A, W, b)
    A = softmax(Z)
    caches.append(cache)

    return A, caches
