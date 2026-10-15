
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split


def one_hot(y, num_classes=10):
    encoded = np.zeros((len(y), num_classes))
    encoded[np.arange(len(y)), y] = 1
    return encoded


print("Loading MNIST dataset...")

mnist = fetch_openml(
    "mnist_784",
    version=1,
    as_frame=False
)

X = mnist.data.astype(np.float32)
y = mnist.target.astype(int)

print("X shape:", X.shape)
print("y shape:", y.shape)

# Normalize pixel values to the range 0 to 1.
X = X / 255.0

# Split the dataset into training and testing sets.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Convert labels to one-hot vectors.
y_train_encoded = one_hot(y_train)
y_test_encoded = one_hot(y_test)

print("Training:", X_train.shape)
print("Testing:", X_test.shape)
print("Data loading complete.")


if __name__ == "__main__":
    # Display one example image when this file is run directly.
    plt.imshow(X[0].reshape(28, 28), cmap="gray")
    plt.title(f"Label: {y[0]}")
    plt.axis("off")
    plt.show()
