"""
utils.py

Preprocesses EEG data from the Sleep-EDF dataset by:
- Loading PSG (EEG) and Hypnogram (label) EDF files
- Selecting 2 specific EEG channels (Fpz-Cz, Pz-Oz)
- Filtering EEG signals (0.5–30 Hz bandpass)
- Split the data into 30-second epochs
- Assign sleep stage annotations to numbered labels
- Returning NumPy arrays X (epochs) and y (labels)

Author: Karma Namgyal
Date edited: 2025-06-24
"""

import os
import mne
import numpy as np
import mne
import warnings
from torch.utils.data import TensorDataset, DataLoader, random_split
import torch

# labeling each sleep stage to a class
LABEL_MAP = {
    "Sleep stage W": 0,
    "Sleep stage 1": 1,
    "Sleep stage 2": 2,
    "Sleep stage 3": 3,
    "Sleep stage 4": 3,  
    "Sleep stage R": 4
}

# Function to preprocess the .edf EEG data
def preprocess_data(data_dir, epoch_duration=30, sfreq=100):
    """
    Preprocesses EEG data from the Sleep-EDF dataset.
    
    Args:
        data_dir (str): Path to your Sleep-EDF dataset folder
        epoch_duration (int): Duration of each epoch in seconds (default: 30)
        sfreq (int): Target sampling frequency (default: 100 Hz)
    
    Returns:
        tuple: (X, y) where X is EEG data and y is sleep stage labels
    """
    # Create arrays for data
    X_list = []
    y_list = []

    # Process all PSG.edf files in the directory
    for file in os.listdir(data_dir):
        if not file.endswith("PSG.edf"):
            continue  

        psg_path = os.path.join(data_dir, file)

        # Get base ID (e.g., 'SC4001')
        base_id = file[:6]

        # Find corresponding hypnogram file
        hypnogram_candidates = [f for f in os.listdir(data_dir)
                                if f.startswith(base_id) and "Hypnogram" in f]

        if not hypnogram_candidates:
            print(f"Skipping {file}: missing hypnogram")
            continue

        hypnogram_path = os.path.join(data_dir, hypnogram_candidates[0])

        try:
            # Load EEG data and sleep stage annotations
            raw = mne.io.read_raw_edf(psg_path, preload=True)
            annotations = mne.read_annotations(hypnogram_path)
            raw.set_annotations(annotations)

            # Select specific EEG channels and apply bandpass filter
            raw.pick_channels(['EEG Fpz-Cz', 'EEG Pz-Oz', 'EOG horizontal', 'EMG submental'])
            raw.filter(0.5, 30)  # 0.5-30 Hz bandpass filter

            # Extract events from annotations
            available_event_ids = {
                k: v for k, v in LABEL_MAP.items()
                if k in [ann['description'] for ann in annotations]
            }
            events, _ = mne.events_from_annotations(raw, event_id=available_event_ids, verbose=False)

            # Skip if no valid events found
            if events.shape[0] == 0:
                print(f"Skipping {file}: no valid events")
                continue

            # Create 30-second epochs
            epochs = mne.Epochs(
                raw, events=events, event_id=available_event_ids,
                tmin=0, tmax=epoch_duration, baseline=None,
                preload=True, verbose=False
            )

            # Convert to numpy array
            X = epochs.get_data()

            # Process labels
            stage_labels = [LABEL_MAP.get(ann['description'], -1) for ann in annotations]
            valid_idx = [i for i, stage in enumerate(stage_labels) if stage != -1]

            # Ensure matching lengths
            min_len = min(len(X), len(valid_idx))
            X = X[:min_len]
            y = [stage_labels[i] for i in valid_idx[:min_len]]

            # Add to collections
            X_list.append(X)
            y_list.append(y)
            
            print(f"Processed {file}: {len(X)} epochs")

        except Exception as e:
            print(f"Error processing {file}: {str(e)}")
            continue

    # Combine all subjects
    X = np.vstack(X_list)
    y = np.hstack(y_list)
    
    # Normalize
    X = (X - X.mean()) / X.std()

    print(f"Total processed: {len(X)} epochs from {len(X_list)} subjects")

    # Create data directory if it doesn't exist
    os.makedirs("data", exist_ok=True)
    
    # Save preprocessed data for future use
    np.save("data/X.npy", X)
    np.save("data/y.npy", y)

    return X, y

def preprocess_file(file_path, epoch_duration=30, sfreq=100):
    """
    Preprocess a single .edf file for inference.
    
    Args:
        file_path (str): Path to the PSG .edf file
        epoch_duration (int): Duration of each epoch in seconds
        sfreq (int): Target sampling frequency (Hz)
    
    Returns:
        torch.Tensor: EEG tensor of shape [num_epochs, channels, time]
    """
    try:
        raw = mne.io.read_raw_edf(file_path, preload=True)
        raw.pick_channels(['EEG Fpz-Cz', 'EEG Pz-Oz', 'EOG horizontal', 'EMG submental'])
        raw.filter(0.5, 30)
        raw.resample(sfreq)

        data = raw.get_data()  # shape [C, T]
        num_channels, total_samples = data.shape

        epoch_len = epoch_duration * sfreq
        total_epochs = total_samples // epoch_len

        segments = []
        for i in range(total_epochs):
            start = i * epoch_len
            end = start + epoch_len
            epoch = data[:, start:end]
            if epoch.shape[1] == epoch_len:
                segments.append(epoch)

        segments = np.array(segments)  # [N, C, T]

        # Normalize each epoch independently
        segments = (segments - segments.mean(axis=(1, 2), keepdims=True)) / segments.std(axis=(1, 2), keepdims=True)

        return torch.tensor(segments, dtype=torch.float32)

    except Exception as e:
        print(f"Error preprocessing file {file_path}: {e}")
        return None

# Function create the data loaders
def create_data_loaders(X, y, batch_size=64):
    """
    Create DataLoader objects for training, validation, and test datasets.
    
    Args:
        X (np.ndarray): Input EEG data (epochs)
        y (np.ndarray): Sleep stage labels
        batch_size (int): Batch size for training
        
    Returns:
        tuple: (train_loader, val_loader, test_loader, user_loader)
    """
    # Convert to PyTorch tensors
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    dataset = TensorDataset(X_tensor, y_tensor)
    generator = torch.Generator().manual_seed(41)

    # Define split sizes
    train_size = int(0.7 * len(dataset))   # 70% training
    val_size = int(0.15 * len(dataset))    # 15% validation
    test_size = int(0.15 * len(dataset))   # 15% testing

    # Split the dataset
    train_dataset, val_dataset, test_dataset, user_dataset = random_split(
        dataset, [train_size, val_size, test_size], generator=generator
    )
    
    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"Data split - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    return train_loader, val_loader, test_loader,