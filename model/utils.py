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

    # Finding the a valid data pair.
    for file in os.listdir(data_dir):
        if not file.endswith("PSG.edf"):
            continue  

        psg_path = os.path.join(data_dir, file)
        hypnogram_path = psg_path.replace("E0-PSG.edf", "EC-Hypnogram.edf")

        if not os.path.exists(hypnogram_path):
            print(f" Skipping {file}: (missing hypnogram)")
            continue

    # Load EEG and annotations
    raw = mne.io.read_raw_edf("data/SC4001E0-PSG.edf", preload=True)
    annotations = mne.read_annotations("data/SC4001EC-Hypnogram.edf")
    raw.set_annotations(annotations)

    # Pick EEG channel and filter
    raw.pick_channels(['EEG Fpz-Cz', 'EEG Pz-Oz'])
    raw.filter(0.5, 30)

    # Turn the annotations into events
    events, _ = mne.events_from_annotations(raw, event_id=None)

    # Put epoch into 30s windows
    epochs = mne.Epochs(raw, events=events, event_id=None, tmin = 0, tmax=epoch_duration, baseline=None, preload=True, verbose=False)

    # Turn into a numpy array
    X = epochs.get_data()

    # loop through annotations and convert to label
    stage_labels = [LABEL_MAP.get(ann['description'], -1) for ann in annotations]

    # Align labels to epoch count (filter invalid ones)
    valid_idx = [i for i, stage in enumerate(stage_labels) if stage != -1]
    
    # Obtaining input data and labels
    X = X[:len(valid_idx)]
    y = [stage_labels[i] for i in valid_idx]

    # Appending X and y data to lists
    X_list.append(X)
    y_list.append(y)   

    # Stack all subjects
    X = np.vstack(X_list)
    y = np.hstack(y_list)

    return X, y