"""
train.py

Main training script for sleep stage classification using EEG data.

- Load and preprocesses EEG and hypnogram data from Sleep-EDF
- Split data into training, validation, and test sets
- Create data loaders for each set
- Trains a CNN-LSTM model to classify sleep stages
- Evaluates performance on the validation and test sets

Author: Karma Namgyal
Date: 2025-07-25
"""
import torch
from model.model import EEG_Model
from model.utils import preprocess_data, create_data_loaders
from helper import evaluate
import numpy as np
import os 
import warnings
from torch.utils.data import TensorDataset, DataLoader
warnings.filterwarnings("ignore")

# Set device 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = EEG_Model().to(device)
model.load_state_dict(torch.load("model_weights.pth", map_location=device))
model.eval()

# Preprocess data (four unseen subjects)
if os.path.exists("data2/X.npy") and os.path.exists("data2/y.npy"):
    X_unseen = np.load("data2/X.npy")
    y_unseen = np.load("data2/y.npy")
else:
    X_unseen, y_unseen = preprocess_data("C:/Users/namgy/OneDrive/Desktop/sleep-stage-visualizer/data2")
    np.save("data2/X.npy", X_unseen)
    np.save("data2/y.npy", y_unseen)

# Create test loader with unseen subjects
X_tensor = torch.tensor(X_unseen).float()
y_tensor = torch.tensor(y_unseen).long()
test_dataset1 = TensorDataset(X_tensor, y_tensor)
test_loader1 = DataLoader(test_dataset1, batch_size=64, shuffle=False)
print("Number of test samples:", len(test_loader1.dataset))

# Evaluate
criterion = torch.nn.CrossEntropyLoss()
error, avg_loss, f1_macro, test_report, conf_mat = evaluate(model, test_loader1, criterion, device)

# Show results
print(f"Test acc: {1-error:.4f}")
print(f"Avg Loss: {avg_loss:.4f}")
print("\nPer-class F1 Scores (on test set):")
for cls_id in range(5): 
    f1 = test_report[str(cls_id)]['f1-score']
    print(f"  Class {cls_id}: F1 = {f1:.4f}")

print("\nConfusion Matrix (on test set):")
print(conf_mat)
