"""
helper.py

Helper functions for model evaluation, model saving and plotting curves.

Includes:
- evaluate(): Computes classification error and average loss on a validation/test set.
- plot_curve(): Plots training accuracy and loss over epochs from saved CSV files.
- get_model_name(): Generate a unique name for models based on hyperparameters.

Author: Karma Namgyal
Date: 2025-07-04
"""

import torch 
import matplotlib.pyplot as plt
import numpy as np

def evaluate(model, loader, criterion, device):
    """
     Args:
         net: PyTorch neural network object
         loader: PyTorch data loader for the validation set
         criterion: The loss function
     Returns:
         err: A scalar for the avg classification error over the validation set
         loss: A scalar for the average loss function over the validation set
     """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    error = 1 - (correct / total)
    return error, avg_loss

def plot_curve(path):
    """ 
    Plots the training and validation curves for accuracy and loss.

    Args:
        path: The base path of the csv files produced during training
    """
    train_err = np.loadtxt(f"{path}_train_err.csv")
    train_loss = np.loadtxt(f"{path}_train_loss.csv")
    val_err = np.loadtxt(f"{path}_val_err.csv")
    val_loss = np.loadtxt(f"{path}_val_loss.csv")

    train_acc = 1 - train_err
    val_acc = 1 - val_err
    n = len(train_acc)

    # Accuracy plot
    plt.figure()
    plt.title("Accuracy Over Epochs")
    plt.plot(range(1, n+1), train_acc, label="Train Accuracy")
    plt.plot(range(1, n+1), val_acc, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.grid(True)
    plt.legend()
    plt.show()

    # Loss plot
    plt.figure()
    plt.title("Loss Over Epochs")
    plt.plot(range(1, n+1), train_loss, label="Train Loss")
    plt.plot(range(1, n+1), val_loss, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()
    plt.show()

def get_model_name(name, batch_size, learning_rate, epoch):
    """ Generate a name for the model consisting of all the hyperparameter values

    Args:
        config: Configuration object containing the hyperparameters
    Returns:
        path: A string with the hyperparameter name and value concatenated
    """
    path = "{0}_bs{1}_lr{2}_epoch{3}".format(name,
                                                   batch_size,
                                                   learning_rate,
                                                   epoch)
    return path