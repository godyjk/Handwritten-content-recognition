# Handwritten-content-recognition
简单的图像识别入门程序，可以实现对手写内容的识别
# MNIST Handwritten Digit Recognition with CNN

This project implements a Convolutional Neural Network (CNN) to recognize handwritten digits from the MNIST dataset using PyTorch.

## Overview

The goal is to train a deep learning model that can accurately classify grayscale images of handwritten digits (0-9) from the MNIST dataset. A CNN is used due to its effectiveness in capturing spatial patterns in images, which is ideal for digit recognition.

## Requirements

- Python 3.6+
- PyTorch
- torchvision
- matplotlib
- numpy (implicitly used via PyTorch)

You can install the required packages using `pip`:
```bash
pip install torch torchvision matplotlib
```

## Dataset

The [MNIST dataset](http://yann.lecun.com/exdb/mnist/) is used, which consists of:
- 60,000 training images
- 10,000 test images
- Each image is a 28x28 grayscale image of a handwritten digit (0-9).

The dataset is automatically downloaded by torchvision when running the code for the first time.

## Model Architecture

The CNN architecture consists of:
1. **Convolutional Layers**:
   - First layer: 1 input channel (grayscale), 32 output channels, 5x5 kernel, ReLU activation, and 2x2 max pooling.
   - Second layer: 32 input channels, 64 output channels, 5x5 kernel, ReLU activation, and 2x2 max pooling.
2. **Fully Connected Layers**:
   - First layer: 64 * 7 * 7 input features, 1000 output features, ReLU activation, and dropout (p=0.4).
   - Second layer: 1000 input features, 10 output features (for 10 digits), with Softmax activation.

## Training

- **Optimizer**: Adam with a learning rate of 0.0003.
- **Loss Function**: Cross-Entropy Loss, suitable for multi-class classification.
- **Epochs**: The model is trained for 20 epochs.
- **Batch Size**: 64 samples per batch.

During training, the loss and accuracy on both training and test sets are recorded.

## Evaluation

After training, the model's performance is evaluated on the test set. The code also generates plots to visualize:
- Training and test loss over epochs.
- Training and test accuracy over epochs.

## Usage

1. Ensure all required packages are installed.
2. Run the Jupyter notebook or Python script. The code will:
   - Download the MNIST dataset.
   - Define the CNN model.
   - Train the model for 20 epochs.
   - Evaluate the model on the test set.
   - Plot loss and accuracy curves.

## Results

The model typically achieves high accuracy (often >98%) on the MNIST test set, demonstrating its effectiveness in handwritten digit recognition. The generated plots will show the loss decreasing and accuracy increasing as training progresses, with the test metrics closely following the training metrics (indicating minimal overfitting, thanks to dropout).

## Customization

To modify the model or training process:
- Adjust the CNN architecture (e.g., add/remove layers, change kernel sizes).
- Modify hyperparameters like learning rate, batch size, or number of epochs.
- Try different optimizers (e.g., SGD, RMSprop) or loss functions.
