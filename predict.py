"""
predict.py

Run a trained EEG sleep stage classification model on a single PSG (.edf) file.

Includes:
- print_patient_info(): Displays demographic and recording details (hardcoded for known files).
- predict(): Loads the trained model, preprocesses the PSG file, and outputs predicted sleep stages.
- mode_filter(): Applies mode-based smoothing to predictions, to fix when the model may have misclassified.
- plot_hypnogram(): Plots raw and smoothed predictions as hypnograms.

Author: Karma Namgyal
Date: 2025-08-07
"""

import os
import warnings
import numpy as np
import torch
import mne
import matplotlib.pyplot as plt
from scipy.stats import mode

from model.model import EEG_Model
from model.utils import preprocess_file

warnings.filterwarnings("ignore")
mne.set_log_level("ERROR")

DATA_DIR = "./data2"
MODEL_PATH = "model_weights.pth"
STAGE_LABELS = ['W', 'N1', 'N2', 'N3', 'REM']

# Hardcoded patient info (by PSG filename)
PATIENT_INFO = {
    "SC4071E0-PSG.edf": ("Female", 30),
    "SC4332F0-PSG.edf": ("Male",   60),
    "SC4412E0-PSG.edf": ("Female", 66),
    "SC4721E0-PSG.edf": ("Male",   88),
}

# Mode smoothing
def mode_filter(preds, window_size=5):
    out = np.empty_like(preds)
    for i in range(len(preds)):
        s = max(0, i - window_size // 2)
        e = min(len(preds), i + window_size // 2 + 1)
        out[i] = int(mode(preds[s:e], keepdims=False).mode)
    return out

# Patient info printout
def print_patient_info(psg_path):
    filename = os.path.basename(psg_path)
    gender, age = PATIENT_INFO.get(filename, ("Unknown", "Unknown"))
    raw = mne.io.read_raw_edf(psg_path, preload=False, verbose=False)

    desired = ['EEG Fpz-Cz', 'EEG Pz-Oz', 'EOG horizontal', 'EMG submental']
    used_channels = [ch for ch in desired if ch in raw.ch_names]

    print("\nPatient Info:")
    print(f"1.File: {filename}")
    print(f"2. Gender: {gender}")
    print(f"3. Age: {age}")
    print(f"4. Duration: {raw.times[-1] / 3600:.2f} hours")

# Prediction
def predict(psg_path, model_path=MODEL_PATH, device='cpu'):
    model = EEG_Model().to(device)
    try:
        state = torch.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(state)
    except TypeError:
        model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    data = preprocess_file(psg_path)
    if data is None:
        print("Preprocessing failed.")
        return np.array([])
    data = data.to(device)

    with torch.no_grad():
        logits = model(data)
        preds = torch.argmax(logits, dim=1).cpu().numpy()
    return preds

# Plot hypnogram
def plot_hypnogram(preds, title):
    plt.figure(figsize=(12, 3))
    plt.step(np.arange(len(preds)), preds, where='mid')
    plt.yticks(np.arange(len(STAGE_LABELS)), STAGE_LABELS)
    plt.gca().invert_yaxis()
    plt.title(title)
    plt.xlabel("Epoch (30s)")
    plt.ylabel("Stage")
    plt.tight_layout()
    plt.show()

# Main
if __name__ == "__main__":
    print("\nEEG Sleep Stage Predictor")
    filename = input("\nEnter PSG filename: ").strip()
    psg_path = os.path.join(DATA_DIR, filename)
    if not os.path.isfile(psg_path):
        print(f"File not found: {psg_path}")
        raise SystemExit

    # Show patient info
    print_patient_info(psg_path)

    # Predict
    print("\nRunning model prediction...")
    preds = predict(psg_path)
    if preds.size == 0:
        print("Prediction failed.")
        raise SystemExit

    smoothed = mode_filter(preds, window_size=5)

    print("\nRaw predictions:", preds[:100])
    print("Smoothed:", smoothed[:100])
    print(f"Total predicted epochs: {len(preds)}")

    # Plot option
    if input("\nPlot hypnograms? (y/n): ").strip().lower() == 'y':
        plot_hypnogram(preds,    "Predicted Sleep Stages (Raw)")
        plot_hypnogram(smoothed, "Predicted Sleep Stages (Smoothed)")