"""
model.py

Contains the EEG_Model class implementing a 1D Convolutional Neural Network
for classifying sleep stages from EEG time-series data.
- 2 convolutional + max-pooling layers for a basic feature extraction for now
- Fully connected layers for classification into sleep stages 

Input shape: (batch_size, 2, 3000)

Author: Karma Namgyal
Date edited: 2025-06-30

"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class EEG_Model(nn.Module):
    def __init__(self):
        super(EEG_Model, self).__init__()
        self.conv1 = nn.Conv1d(2, 32, 5, 2)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(32, 64, 3, 1)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(2)
        self.fc1 = nn.Linear(64 * 373, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 5)

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
