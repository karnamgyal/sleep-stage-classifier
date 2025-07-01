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

from model import utils