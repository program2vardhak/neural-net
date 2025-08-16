import numpy as np
import pandas as pd

# Load dataset
df = pd.read_csv('Data/mnist_train.csv')
X = df.iloc[:, 1:].to_numpy()  # Features (784 pixels)
y = df.iloc[:, 0].to_numpy()   # Labels (digits 0-9)

# Shuffle before splitting to cause randmness while training to become more accurate
indi = np.random.permutation(len(df))
X = X[indi]
y = y[indi]

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
layers = [784, 128, 64, 10]

"""
Initializing Parameters and Bias for the layers

"""
def initialize_parameters(layers):
    np.random.seed(42)
    parameters = {}
    L = len(layers)  # number of layers in network

    for i in range(1, L):
        parameters[f"W{i}"] = np.random.randn(layers[i], layers[i-1]) * 0.01
        parameters[f"b{i}"] = np.zeros((layers[i], 1))  # bias starts at 0

    return parameters


# Example: initialize weights
parameters = initialize_parameters(layers)
m=X_train.shape[1]
"""
Applying Forward Propogation

"""
def linear_forward(A_prev, W, b):
    Z = np.dot(W, A_prev) + b
    con = (A_prev, W, b, Z)  # Save values for backprop later
    return Z, con

def Leaky_relu(Z,k=0.01):   #Leaky Relu to prevent dead neurons
    return np.maximum(k*Z, Z)

def softmax(Z):
    expZ = np.exp(Z - np.max(Z, axis=0, keepdims=True))  # stable softmax
    return expZ / np.sum(expZ, axis=0, keepdims=True)

def forward_propagation(X, parameters):
    cons = []   # store all (A_prev, W, b, Z) for backprop
    A = X
    L = len(parameters) // 2   # number of layers

    for l in range(1, L):  # hidden layers (ReLU)
        W = parameters[f"W{l}"]
        b = parameters[f"b{l}"]
        Z, con = linear_forward(A, W, b)
        A = Leaky_relu(Z)
        cons.append(con)

    # Final layer (softmax for classification)
    W = parameters[f"W{L}"]
    b = parameters[f"b{L}"]
    Z, con = linear_forward(A, W, b)
    A = softmax(Z)
    cons.append(con)

    return A, cons

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

def backward_propagation(AL, Y, cons, parameters):
    grads = {}
    L = len(parameters) // 2
    m = AL.shape[1]

    # Output layer
    dZL = AL - Y
    A_prev, W, b, Z = cons[-1]
    grads[f"dW{L}"] = (1/m) * np.dot(dZL, A_prev.T)
    grads[f"db{L}"] = (1/m) * np.sum(dZL, axis=1, keepdims=True)

    dA_prev = np.dot(W.T, dZL)

    # Hidden layers
    for l in reversed(range(1, L)):
        A_prev, W, b, Z = cons[l-1]
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
def model(X, Y, layers, learning_rate=0.01, num_epochs=500, print_cost=True):
    parameters = initialize_parameters(layers)
    Y_oh = one_hot(Y, layers[-1])
    costs = []

    for i in range(num_epochs):
        # Forward
        AL, cons = forward_propagation(X, parameters)

        # Cost
        cost = compute_cost(AL, Y_oh)
        costs.append(cost)

        # Backward
        grads = backward_propagation(AL, Y_oh, cons, parameters)

        # Update
        parameters = update_parameters(parameters, grads, learning_rate)

        if print_cost and i % 50 == 0:
            print(f"Epoch {i} | Cost: {cost:.4f}")

    return parameters, costs


#Prediction
def predict(X, parameters):
    AL, _ = forward_propagation(X, parameters)
    return np.argmax(AL, axis=0)


#Training
parameters, training_costs = model(X_train, y_train, layers, learning_rate=0.01, num_epochs=500)

y_pred = predict(X_val, parameters)
accuracy = np.mean(y_pred == y_val.flatten())
print(f"\n Final Validation Accuracy: {accuracy*100:.2f}%")



"""
VISUALIZATION AND ANALYSIS SECTION
"""

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Set style for better plots
plt.style.use('default')
sns.set_palette("husl")

print("\n" + "="*50)
print(" VISUALIZATION AND ANALYSIS")
print("="*50)

# 1. Training Cost Curve
plt.figure(figsize=(10, 6))
plt.plot(training_costs, linewidth=2, color='blue')
plt.title('Training Cost Over Epochs', fontsize=16, fontweight='bold')
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Cost (Cross-Entropy Loss)', fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Prepare data for visualization (transpose back to original format)
X_val_viz = X_val.T  # (samples, features)
y_val_viz = y_val.flatten()
y_pred_viz = y_pred

# 2. Confusion Matrix
plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_val_viz, y_pred_viz)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar_kws={'label': 'Count'})
plt.title('Confusion Matrix - Validation Set', fontsize=16, fontweight='bold')
plt.xlabel('Predicted Label', fontsize=12)
plt.ylabel('True Label', fontsize=12)
plt.tight_layout()
plt.show()

# 3. Per-class Accuracy Analysis
print("\n PER-CLASS ACCURACY ANALYSIS:")
print("-" * 40)
class_accuracies = []
for digit in range(10):
    mask = (y_val_viz == digit)
    if np.sum(mask) > 0:
        class_acc = np.mean(y_pred_viz[mask] == y_val_viz[mask])
        class_accuracies.append(class_acc)
        print(f"Digit {digit}: {class_acc*100:.1f}% ({np.sum(mask)} samples)")
    else:
        class_accuracies.append(0)

# Plot per-class accuracy
plt.figure(figsize=(10, 6))
bars = plt.bar(range(10), [acc*100 for acc in class_accuracies],
               color=sns.color_palette("viridis", 10))
plt.title('Per-Class Accuracy', fontsize=16, fontweight='bold')
plt.xlabel('Digit Class', fontsize=12)
plt.ylabel('Accuracy (%)', fontsize=12)
plt.ylim(0, 100)

# Add value labels on bars
for bar, acc in zip(bars, class_accuracies):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
             f'{acc*100:.1f}%', ha='center', va='bottom', fontweight='bold')

plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.show()

# 4. Sample Correct Predictions
print("\n SAMPLE CORRECT PREDICTIONS:")
correct_mask = (y_pred_viz == y_val_viz)
correct_indices = np.where(correct_mask)[0]

plt.figure(figsize=(15, 6))
for i in range(10):
    plt.subplot(2, 5, i+1)
    idx = correct_indices[i]
    plt.imshow(X_val_viz[idx].reshape(28, 28), cmap='gray')
    plt.title(f'True: {y_val_viz[idx]}, Pred: {y_pred_viz[idx]}',
              fontsize=10, color='green', fontweight='bold')
    plt.axis('off')
plt.suptitle('Correct Predictions', fontsize=16, fontweight='bold', color='green')
plt.tight_layout()
plt.show()

# 5. Sample Incorrect Predictions (if any)
incorrect_mask = (y_pred_viz != y_val_viz)
num_incorrect = np.sum(incorrect_mask)

if num_incorrect > 0:
    print(f"\n FOUND {num_incorrect} MISCLASSIFIED SAMPLES:")
    incorrect_indices = np.where(incorrect_mask)[0]

    plt.figure(figsize=(15, 6))
    num_show = min(10, num_incorrect)

    for i in range(num_show):
        plt.subplot(2, 5, i+1)
        idx = incorrect_indices[i]
        plt.imshow(X_val_viz[idx].reshape(28, 28), cmap='gray')
        plt.title(f'True: {y_val_viz[idx]}, Pred: {y_pred_viz[idx]}',
                  fontsize=10, color='red', fontweight='bold')
        plt.axis('off')

    plt.suptitle('Misclassified Examples', fontsize=16, fontweight='bold', color='red')
    plt.tight_layout()
    plt.show()

    # Error analysis by true class
    print("\n ERROR ANALYSIS BY TRUE CLASS:")
    print("-" * 35)
    for digit in range(10):
        true_mask = (y_val_viz == digit)
        if np.sum(true_mask) > 0:
            errors_for_digit = np.sum(incorrect_mask & true_mask)
            error_rate = errors_for_digit / np.sum(true_mask) * 100
            print(f"Digit {digit}: {errors_for_digit} errors ({error_rate:.1f}% error rate)")
else:
    print("\n No misclassified samples!")

# 6. Digit Distribution in Validation Set
plt.figure(figsize=(10, 6))
unique, counts = np.unique(y_val_viz, return_counts=True)
bars = plt.bar(unique, counts, color=sns.color_palette("Set3", 10))
plt.title('Distribution of Digits in Validation Set', fontsize=16, fontweight='bold')
plt.xlabel('Digit Class', fontsize=12)
plt.ylabel('Count', fontsize=12)

# Add count labels on bars
for bar, count in zip(bars, counts):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
             f'{count}', ha='center', va='bottom', fontweight='bold')

plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.show()

# 7. Model Summary
print(f"\n MODEL SUMMARY:")
print("=" * 30)
print(f"Architecture: {layers}")
print(f"Total Parameters: {sum(p.size for p in parameters.values() if hasattr(p, 'size'))}")
print(f"Training Samples: {X_train.shape[1]}")
print(f"Validation Samples: {X_val.shape[1]}")
print(f"Final Training Cost: {training_costs[-1]:.4f}")
print(f"Validation Accuracy: {accuracy*100:.2f}%")

# Classification Report
print(f"\n DETAILED CLASSIFICATION REPORT:")
print("-" * 40)
print(classification_report(y_val_viz, y_pred_viz,
                          target_names=[f'Digit {i}' for i in range(10)]))

print("\n Analysis Complete! Check the plots above for detailed insights.")