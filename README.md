# EEG Sleep Stage Classifier 

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

## Evaluation
The model was trained and then tested on a smaller dataset for evaluation. 
- Test accuracy: 0.7288
- Test loss: 0.6814 

The model was also tested on the unseen subjects four that were taken out for the visualization on the web app.
- Test accuracy: 0.7610
- Test loss: 0.5872

## Reflection 
This was also a highly valuable hands-on experience with the entire machine learning pipeline, preparing me to take on more challenging projects. While the model does not perform as accurately as those in research papers, it still performs well and provides valuable insight into how sleep stages evolve throughout the day-night cycle.

## Full Project Report
This contains a full deep dive explaining each part of the project's deep learning pipeline: [Detailed Write-up on Notion](https://tropical-address-50b.notion.site/Sleep-Stage-Classification-from-Full-Night-EEG-EOG-EMG-Recording-Using-CNN-LSTM-213209af64a9803a8ca0ccb6b78e73c1)  


