# MNIST Neural Network From Scratch

A handwritten digit classification project built from scratch using Python and NumPy, without deep learning frameworks like TensorFlow or PyTorch.

## Features
- Loads and preprocesses the MNIST dataset.
- Implements a neural network with two hidden layers.
- Uses ReLU and Softmax activation functions.
- Implements cross-entropy loss, backpropagation, and gradient descent.
- Evaluates model accuracy on test data.

## Tech Stack
- Python
- NumPy
- Matplotlib
- Scikit-learn

## Getting Started

```bash
git clone YOUR_REPOSITORY_URL
cd mnist-from-scratch

python -m venv venv
```

Activate the environment on Windows:

```powershell
.\venv\Scripts\Activate.ps1
```

Install dependencies and run:

```bash
pip install -r requirements.txt
python neural_network.py
```

The MNIST dataset will be downloaded when the data loader runs for the first time.

## Learning Goals

This project helped me understand neural network architecture, forward propagation, loss functions, backpropagation, gradient descent, and model evaluation.

**Built as part of my deep learning journey.**
