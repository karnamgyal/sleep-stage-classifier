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
import os
import torch 
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from model.model import EEG_Model
from model.utils import preprocess_data, create_data_loaders
from helper import evaluate, plot_curve, get_model_name

def train_CNN(model, train_loader, val_loader, batch_size=64, learning_rate=0.0001, num_epochs=30, plot_path=None):

    # Make sure the directory exists
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print("Using device:", torch.cuda.get_device_name(torch.cuda.current_device()) if torch.cuda.is_available() else "CPU")

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)

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

    test_err, test_loss = evaluate(model, test_loader, criterion, device)
    print(f"\nTest Loss: {test_loss:.4f}, Test Accuracy: {1-test_err:.4f}")

    return train_loss, train_err, val_loss, val_err

# Preprocess data
if os.path.exists("data/X.npy") and os.path.exists("data/y.npy"):
    X = np.load("data/X.npy")
    y = np.load("data/y.npy")
else:
    X, y = preprocess_data("C:/Users/namgy/Downloads/sleep-edf-database-expanded-1.0.0/sleep-edf-database-expanded-1.0.0/sleep-cassette")
    np.save("data/X.npy", X)
    np.save("data/y.npy", y)

# Create DataLoaders
train_loader, val_loader, test_loader, user_loader = create_data_loaders(X, y)

# Initialize model
model = EEG_Model()

# Train the model
train_loss, train_err, val_loss, val_err = train_CNN(model, train_loader, val_loader, plot_path="plots/training_metrics")

# Plot training curves
plot_curve("plots/training_metrics")

