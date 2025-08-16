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
m=X_train.shape[1]
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

# One Hot Encoding
def one_hot(Y, num_classes=10):
    return np.eye(num_classes)[Y.reshape(-1)].T   # (num_classes, m)


#Cost Function
def compute_cost(AL, Y):
    """
    AL -- softmax probabilities (num_classes, m)
    Y  -- one-hot labels (num_classes, m)
    """
    m = Y.shape[1]
    cost = -(1/m) * np.sum(Y * np.log(AL + 1e-8))
    return np.squeeze(cost)


#Backward Propogation
def relu_backward(dA, Z):
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0
    return dZ

def backward_propagation(AL, Y, caches, parameters):
    grads = {}
    L = len(parameters) // 2
    m = AL.shape[1]

    # Output layer
    dZL = AL - Y
    A_prev, W, b, Z = caches[-1]
    grads[f"dW{L}"] = (1/m) * np.dot(dZL, A_prev.T)
    grads[f"db{L}"] = (1/m) * np.sum(dZL, axis=1, keepdims=True)

    dA_prev = np.dot(W.T, dZL)

    # Hidden layers
    for l in reversed(range(1, L)):
        A_prev, W, b, Z = caches[l-1]
        dZ = relu_backward(dA_prev, Z)
        grads[f"dW{l}"] = (1/m) * np.dot(dZ, A_prev.T)
        grads[f"db{l}"] = (1/m) * np.sum(dZ, axis=1, keepdims=True)
        dA_prev = np.dot(W.T, dZ)

    return grads


#Update Parameters
def update_parameters(parameters, grads, learning_rate):
    L = len(parameters) // 2
    for l in range(1, L+1):
        parameters[f"W{l}"] -= learning_rate * grads[f"dW{l}"]
        parameters[f"b{l}"] -= learning_rate * grads[f"db{l}"]
    return parameters


"""
Training the Model

"""
def model(X, Y, layer_dims, learning_rate=0.01, num_epochs=500, print_cost=True):
    parameters = initialize_parameters(layer_dims)
    Y_oh = one_hot(Y, layer_dims[-1])

    for i in range(num_epochs):
        # Forward
        AL, caches = forward_propagation(X, parameters)

        # Cost
        cost = compute_cost(AL, Y_oh)

        # Backward
        grads = backward_propagation(AL, Y_oh, caches, parameters)

        # Update
        parameters = update_parameters(parameters, grads, learning_rate)

        if print_cost and i % 50 == 0:
            print(f"Epoch {i} | Cost: {cost:.4f}")

    return parameters


#Prediction
def predict(X, parameters):
    AL, _ = forward_propagation(X, parameters)
    return np.argmax(AL, axis=0)


#Training
parameters = model(X_train, y_train, layer_dims, learning_rate=0.01, num_epochs=500)

y_pred = predict(X_val, parameters)
accuracy = np.mean(y_pred == y_val.flatten())
print("Validation Accuracy:", accuracy)
