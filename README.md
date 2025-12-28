# Full day-night EEG/EMG/EOG Sleep Stage Classifier 

This project involved building a complete deep learning pipeline from scratch: from raw EEG/EOG/EMG data preprocessing with MNE, to designing and training a CNN-LSTM model, to evaluating test performance and deploying an interactive Streamlit web app for visualization!

This model predicts sleep stages (Wake, N1, N2, N3, REM) from EEG, EOG, and EMG recordings using a CNN-LSTM deep learning architecture.  
I have deployed this project on HuggingFace as a web app here: 

## Technologies Used
Python, PyTorch, NumPy, MNE, scikit-learn, Matplotlib, Streamlit, plotly.

## Dataset 

This project uses the sleep-cassette file from the Sleep-EDF Expanded dataset from PhysioNet: [https://physionet.org/content/sleep-edfx/1.0.0/](https://physionet.org/content/sleep-edfx/1.0.0/)

- Each recording spans an entire day-night cycle (~20 hours) from 153 caucasian subjects aged 25-101.
- Four of the subjects were excluded from training and testing to use for the web app visualization.
- Hypnogram annotations (labels) are provided in 22k+ 30-second epochs, including both daytime wake and nighttime sleep stages.

## Features
- Multimodal input: EEG (Fpz-Cz, Pz-Oz), EOG, EMG channels
- CNN layers for spatial feature extraction  
- Stacked LSTMs for capturing temporal dependencies
- Supports raw and smoothed predictions for comparison and visualization  

## Baseline 
- Built a 1D-CNN with simpler architecture to test ability to capture local patterns, trained on EEG only.
- Accuracy: 0.6198.
- Test Loss: 0.9669.
- Test Macro-F1: 0.6073.

## Evaluation
The primary CNN-LSTM model was trained and after rigorous hyperparameter tuning and architecture changes, we obtained: 
- Test Accuracy: 0.7357.
- Test Macro-F1: 0.7341.
- Test Loss: 0.6520.

## Deployment 
I have deployed the model on Streamlit for interactive visualization: [Streamlit Demo](https://sleep-stage-classifier.streamlit.app/)

## Reflection 
This was also a highly valuable hands-on experience with the entire machine learning pipeline, preparing me to take on more challenging projects. While the model does not perform as accurately as those in research papers, it still performs well and provides valuable insight into how sleep stages evolve throughout the day-night cycle.

## Full Project Report
This contains a full deep dive explaining each part of the project's deep learning pipeline: [Detailed Write-up on Notion](https://tropical-address-50b.notion.site/Karma-D-Namgyal-1cc209af64a9800f8660f4bbf5c5ce53?p=24d209af64a980bba7afccb8298b46ed&pm=c)  




