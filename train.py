"""
train.py

Main training script for sleep stage classification using EEG data.

This script:
- Loads and preprocesses EEG and hypnogram data from Sleep-EDF
- Splits the data into training, validation, and test sets
- Wraps the data into PyTorch DataLoaders
- Trains a CNN-LSTM model to classify sleep stages
- Evaluates performance on the validation and test sets

Author: Karma Namgyal
Date: 2006-09-27
"""

import torch 
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from helper import evaluate, plot_curve, get_model_name


def train_CNN(model, train_loader, val_loader, batch_size=64, learning_rate=0.001, num_epochs=30, plot_path=None):

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Arrays to track metrics
    train_err = np.zeros(num_epochs)
    train_loss = np.zeros(num_epochs)
    val_err = np.zeros(num_epochs)
    val_loss = np.zeros(num_epochs)

    for epoch in range(num_epochs):
        model.train()
        correct_train = 0
        total_train = 0
        total_train_loss = 0

        for eeg_data, labels in train_loader:
            eeg_data, labels = eeg_data.to(device), labels.to(device)

            # Forward and backward pass
            optimizer.zero_grad()
            outputs = model(eeg_data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Metrics
            total_train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        # Save training metrics
        train_loss[epoch] = total_train_loss / len(train_loader)
        train_err[epoch] = 1 - (correct_train / total_train)

        # Evaluate on validation set
        val_err[epoch], val_loss[epoch] = evaluate(model, val_loader, criterion, device)

        # Print metrics
        print(f"Epoch {epoch+1}/{num_epochs}: "
              f"Train Loss: {train_loss[epoch]:.4f}, Train Acc: {1-train_err[epoch]:.4f} | "
              f"Val Loss: {val_loss[epoch]:.4f}, Val Acc: {1-val_err[epoch]:.4f}")

    # Save metrics after training
    if plot_path:
        np.savetxt(f"{plot_path}_train_err.csv", train_err)
        np.savetxt(f"{plot_path}_train_loss.csv", train_loss)
        np.savetxt(f"{plot_path}_val_err.csv", val_err)
        np.savetxt(f"{plot_path}_val_loss.csv", val_loss)

    return train_loss, train_err, val_loss, val_err