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

warnings.filterwarnings("ignore", category=RuntimeWarning)
mne.set_log_level("WARNING")

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

    # Create arrays for data
    X_list = []
    y_list = []

    # Finding valid data pairs
    for file in os.listdir(data_dir):
        if not file.endswith("PSG.edf"):
            continue  

        psg_path = os.path.join(data_dir, file)

        # Get base ID like 'SC4001'
        base_id = file[:6]

        # Find any hypnogram file that starts with same base ID
        hypnogram_candidates = [f for f in os.listdir(data_dir)
                                if f.startswith(base_id) and "Hypnogram" in f]

        if not hypnogram_candidates:
            print(f" Skipping {file}: (missing hypnogram)")
            continue

        hypnogram_path = os.path.join(data_dir, hypnogram_candidates[0])

        # Load EEG and annotations
        raw = mne.io.read_raw_edf(psg_path, preload=True)
        annotations = mne.read_annotations(hypnogram_path)
        raw.set_annotations(annotations)

        # Pick EEG channels and filter
        raw.pick_channels(['EEG Fpz-Cz', 'EEG Pz-Oz'])
        raw.filter(0.5, 30)

        # Extract matching events from annotations
        available_event_ids = {
            k: v for k, v in LABEL_MAP.items()
            if k in [ann['description'] for ann in annotations]
        }
        events, _ = mne.events_from_annotations(raw, event_id=available_event_ids, verbose=False)

        # Skip if no valid events
        if events.shape[0] == 0:
            print(f" Skipping {file}: (no valid events)")
            continue

        # Split into 30-second epochs
        epochs = mne.Epochs(
            raw, events=events, event_id=available_event_ids,
            tmin=0, tmax=epoch_duration, baseline=None,
            preload=True, verbose=False
        )

        # Turn into a numpy array
        X = epochs.get_data()

        # Convert annotation labels
        stage_labels = [LABEL_MAP.get(ann['description'], -1) for ann in annotations]
        valid_idx = [i for i, stage in enumerate(stage_labels) if stage != -1]

        # Truncate to the minimum length
        min_len = min(len(X), len(valid_idx))
        X = X[:min_len]
        y = [stage_labels[i] for i in valid_idx[:min_len]]

        # Append to data lists
        X_list.append(X)
        y_list.append(y)

    # Stack all subjects
    X = np.vstack(X_list)
    y = np.hstack(y_list)

    # Save preprocessed data 
    np.save("data/X.npy", X)
    np.save("data/y.npy", y)

    return X, y

# Function create the data loaders
def create_data_loaders(X, y, batch_size=64):
    """
    Create DataLoader objects for training and validation datasets.
    
    Args:
        X (np.ndarray): Input data (epochs).
        y (np.ndarray): Labels corresponding to the epochs.
        batch_size (int): Size of each batch.
        
    Returns:
        DataLoader: DataLoader object for the dataset.
    """
    from torch.utils.data import TensorDataset, DataLoader, random_split
    import torch
    
    # Convert to PyTorch tensors
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    dataset = TensorDataset(X_tensor, y_tensor)

    # Declare lengths of sets
    train_size = int(0.7 * len(dataset))  
    val_size = int(0.15 * len(dataset))   
    test_size = int(0.15 * len(dataset))  
    user_size = len(dataset) - train_size - val_size - test_size 

    # Split the data
    train_dataset, val_dataset, test_dataset, user_dataset = random_split(dataset, [train_size, val_size, test_size, user_size])
    
    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    user_loader = DataLoader(user_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, user_loader
