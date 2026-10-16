
import numpy as np
import matplotlib.pyplot as plt

from data_loader import (
    X_train,
    X_test,
    y_train,
    y_test,
    y_train_encoded,
)

np.random.seed(42)

# Network architecture
input_size = 784
hidden1_size = 128
hidden2_size = 64
output_size = 10

print("Hidden layer 1:", hidden1_size)
print("Hidden layer 2:", hidden2_size)



# Initialize weights and biases
W1 = np.random.randn(input_size, hidden1_size) * np.sqrt(2 / input_size)
b1 = np.zeros((1, hidden1_size))

W2 = np.random.randn(hidden1_size, hidden2_size) * np.sqrt(2 / hidden1_size)
b2 = np.zeros((1, hidden2_size))

W3 = np.random.randn(hidden2_size, output_size) * np.sqrt(2 / hidden2_size)
b3 = np.zeros((1, output_size))

print("W1 shape:", W1.shape)
print("W2 shape:", W2.shape)
print("W3 shape:", W3.shape)

def relu(x):
    return np.maximum(0, x)


def relu_derivative(x):
    return (x > 0).astype(float)


def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


def forward(X):
    z1 = X @ W1 + b1
    a1 = relu(z1)

    z2 = a1 @ W2 + b2
    a2 = relu(z2)

    z3 = a2 @ W3 + b3
    y_pred = softmax(z3)

    return z1, a1, z2, a2, y_pred


def cross_entropy(y_true, y_pred):
    epsilon = 1e-8
    return -np.mean(
        np.sum(y_true * np.log(y_pred + epsilon), axis=1)
    )


# Training settings
learning_rate = 0.01
epochs = 10
batch_size = 64

print("\nStarting neural network training...\n")

for epoch in range(epochs):
    # Shuffle the training data every epoch.
    indices = np.random.permutation(len(X_train))
    X_shuffled = X_train[indices]
    y_shuffled = y_train_encoded[indices]

    epoch_loss = 0.0

    for i in range(0, len(X_shuffled), batch_size):
        X_batch = X_shuffled[i:i + batch_size]
        y_batch = y_shuffled[i:i + batch_size]

        m = len(X_batch)

        # 1. Forward propagation
        z1, a1, z2, a2, y_pred = forward(X_batch)

        # 2. Calculate loss
        loss = cross_entropy(y_batch, y_pred)
        epoch_loss += loss * m

        # 3. Backpropagation
        dz3 = y_pred - y_batch
        dW3 = (a2.T @ dz3) / m
        db3 = np.sum(dz3, axis=0, keepdims=True) / m

        da2 = dz3 @ W3.T
        dz2 = da2 * relu_derivative(z2)
        dW2 = (a1.T @ dz2) / m
        db2 = np.sum(dz2, axis=0, keepdims=True) / m

        da1 = dz2 @ W2.T
        dz1 = da1 * relu_derivative(z1)
        dW1 = (X_batch.T @ dz1) / m
        db1 = np.sum(dz1, axis=0, keepdims=True) / m

        # 4. Update weights and biases
        W3 -= learning_rate * dW3
        b3 -= learning_rate * db3

        W2 -= learning_rate * dW2
        b2 -= learning_rate * db2

        W1 -= learning_rate * dW1
        b1 -= learning_rate * db1

    average_loss = epoch_loss / len(X_train)

    print(
        f"Epoch {epoch + 1}/{epochs}, "
        f"Loss: {average_loss:.4f}"
    )


# Evaluate the entire test dataset.
_, _, _, _, test_predictions = forward(X_test)

predicted_digits = np.argmax(test_predictions, axis=1)

accuracy = np.mean(predicted_digits == y_test)

print(f"\nTest accuracy: {accuracy * 100:.2f}%")

print("\nFirst 20 predictions:")
print("Predicted:", predicted_digits[:20])
print("Actual:   ", y_test[:20])

print(
    "Total incorrect predictions:",
    np.sum(predicted_digits != y_test)
)

# Display one incorrectly classified test image, if any.
wrong = np.where(predicted_digits != y_test)[0]

if len(wrong) > 0:
    index = np.random.choice(wrong)

    plt.imshow(
        X_test[index].reshape(28, 28),
        cmap="gray"
    )

    plt.title(
        f"Predicted: {predicted_digits[index]}, "
        f"Actual: {y_test[index]}"
    )

    plt.axis("off")
    plt.show()
else:
    print("No incorrect predictions found in the test set.")
