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
from sklearn.metrics import f1_score, confusion_matrix, classification_report

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for eeg_data, labels in dataloader:
            eeg_data, labels = eeg_data.to(device), labels.to(device)
            outputs = model(eeg_data)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    error = 1 - (total_correct / total_samples)
    avg_loss = total_loss / len(dataloader)

    f1_macro = f1_score(all_labels, all_preds, average='macro')
    class_report = classification_report(all_labels, all_preds, digits=4, output_dict=True, zero_division=0)
    conf_mat = confusion_matrix(all_labels, all_preds)

    return error, avg_loss, f1_macro, class_report, conf_mat

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

def log_run(file_path, params, metrics):
    with open(file_path, "a") as f:
        f.write("=== New Model Run ===\n")
        for key, value in params.items():
            f.write(f"{key}: {value}\n")
        for key, value in metrics.items():
            f.write(f"{key}: {value}\n")
        f.write("\n")