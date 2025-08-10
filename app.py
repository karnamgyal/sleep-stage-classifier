"""
Sleep Stage Classification Web App

A Streamlit web application for EEG/EMG/EOG sleep stage classification using my pre-trained CNN-LSTM model.

Author: Karma Namgyal
Date: 2025-08-10
"""

import streamlit as st
import os
import warnings
import numpy as np
import torch
import mne
import matplotlib.pyplot as plt
from scipy.stats import mode
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")
mne.set_log_level("ERROR")

# Import your model (you'll need to ensure these are available)
try:
    from model.model import EEG_Model
    from model.utils import preprocess_file
except ImportError:
    st.error("Model files not found. Please ensure model.model and model.utils are available.")
    st.stop()

# Configuration
DATA_DIR = "./data2"
MODEL_PATH = "model_weights.pth"
STAGE_LABELS = ['Wake', 'N1', 'N2', 'N3', 'REM']
STAGE_COLORS = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']

# Patient information (hardcoded from your original code)
PATIENT_INFO = {
    "SC4071E0-PSG.edf": ("Female", 30),
    "SC4332F0-PSG.edf": ("Male", 60),
    "SC4412E0-PSG.edf": ("Female", 66),
    "SC4721E0-PSG.edf": ("Male", 88),
}

# Available PSG files for selection
AVAILABLE_FILES = list(PATIENT_INFO.keys())

def mode_filter(preds, window_size=5):
    """Apply mode-based smoothing to predictions."""
    out = np.empty_like(preds)
    for i in range(len(preds)):
        s = max(0, i - window_size // 2)
        e = min(len(preds), i + window_size // 2 + 1)
        out[i] = int(mode(preds[s:e], keepdims=False).mode)
    return out

@st.cache_data
def get_patient_info(psg_path):
    """Get patient demographic and recording information."""
    filename = os.path.basename(psg_path)
    gender, age = PATIENT_INFO.get(filename, ("Unknown", "Unknown"))
    
    try:
        raw = mne.io.read_raw_edf(psg_path, preload=False, verbose=False)
        duration_hours = raw.times[-1] / 3600
        desired = ['EEG Fpz-Cz', 'EEG Pz-Oz', 'EOG horizontal', 'EMG submental']
        used_channels = [ch for ch in desired if ch in raw.ch_names]
        
        return {
            'filename': filename,
            'gender': gender,
            'age': age,
            'duration': duration_hours,
            'channels': used_channels
        }
    except Exception as e:
        st.error(f"Error reading file information: {e}")
        return None

@st.cache_data
def load_and_predict(psg_path, device='cpu'):
    """Load model and make predictions on PSG file."""
    try:
        # Load model
        model = EEG_Model().to(device)
        try:
            state = torch.load(MODEL_PATH, map_location=device, weights_only=True)
            model.load_state_dict(state)
        except TypeError:
            model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        
        model.eval()
        
        # Preprocess data
        data = preprocess_file(psg_path)
        if data is None:
            st.error("Preprocessing failed.")
            return None, None
        
        data = data.to(device)
        
        # Make predictions
        with torch.no_grad():
            logits = model(data)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            
        # Apply smoothing
        smoothed = mode_filter(preds, window_size=5)
        
        return preds, smoothed
        
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None, None

def create_hypnogram_plotly(preds, title, stage_labels=STAGE_LABELS):
    """Create an interactive hypnogram using Plotly."""
    epochs = np.arange(len(preds))
    
    fig = go.Figure()
    
    # Add the sleep stage trace
    fig.add_trace(go.Scatter(
        x=epochs,
        y=preds,
        mode='lines',
        name='Sleep Stages',
        line=dict(width=2, shape='hv'),
        hovertemplate='Epoch: %{x}<br>Stage: %{text}<extra></extra>',
        text=[stage_labels[p] for p in preds]
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Epoch (30s intervals)",
        yaxis_title="Sleep Stage",
        yaxis=dict(
            tickmode='array',
            tickvals=list(range(len(stage_labels))),
            ticktext=stage_labels,
            autorange="reversed"
        ),
        hovermode='x unified',
        height=400
    )
    
    return fig

def calculate_sleep_statistics(preds, stage_labels=STAGE_LABELS):
    """Calculate sleep stage statistics."""
    total_epochs = len(preds)
    stage_counts = np.bincount(preds, minlength=len(stage_labels))
    stage_percentages = (stage_counts / total_epochs) * 100
    
    # Convert to minutes (assuming 30-second epochs)
    stage_minutes = stage_counts * 0.5
    
    stats = {}
    for i, label in enumerate(stage_labels):
        stats[label] = {
            'count': stage_counts[i],
            'percentage': stage_percentages[i],
            'minutes': stage_minutes[i]
        }
    
    return stats

def main():
    st.set_page_config(
        page_title="Sleep Stage Classification",
        page_icon="🛌",
        layout="wide"
    )
    
    st.title("🛌 Sleep Stage Classification Model")
    st.markdown("**Full day/night EEG-based sleep stage classification using deep learning**")
    
    # Sidebar for file selection
    st.sidebar.header("📁 Select PSG Recording")
    st.sidebar.markdown("Choose from available polysomnography files:")
    
    selected_file = st.sidebar.selectbox(
        "Available Files:",
        AVAILABLE_FILES,
        help="Select a PSG (.edf) file for analysis"
    )
    
    # Sidebar info
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Model Info:**")
    st.sidebar.markdown("- CNN-LSTM Architecture")
    st.sidebar.markdown("- 5 Sleep Stages: Wake, N1, N2, N3, REM")
    st.sidebar.markdown("- 30-second epoch analysis")
    
    if selected_file:
        psg_path = os.path.join(DATA_DIR, selected_file)
        
        # Check if file exists
        if not os.path.isfile(psg_path):
            st.error(f"File not found: {psg_path}")
            st.stop()
        
        # Display patient information
        st.header("👤 Patient Information")
        patient_info = get_patient_info(psg_path)
        
        if patient_info:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Gender", patient_info['gender'])
            with col2:
                st.metric("Age", f"{patient_info['age']} years")
            with col3:
                st.metric("Duration", f"{patient_info['duration']:.2f} hours")
            with col4:
                st.metric("Channels", len(patient_info['channels']))
        
        # Prediction section
        st.header("🧠 Sleep Stage Analysis")
        
        if st.button("🔮 Analyze Sleep Stages", type="primary"):
            with st.spinner("Loading model and analyzing sleep data..."):
                raw_preds, smoothed_preds = load_and_predict(psg_path)
            
            if raw_preds is not None and smoothed_preds is not None:
                st.success(f"✅ Analysis complete! Processed {len(raw_preds)} epochs.")
                
                # Create tabs for different views
                tab1, tab2, tab3, tab4 = st.tabs(["📊 Hypnograms", "📈 Statistics", "🔢 Raw Data", "ℹ️ About"])
                
                with tab1:
                    st.subheader("Sleep Stage Hypnograms")
                    
                    # Raw predictions
                    fig_raw = create_hypnogram_plotly(raw_preds, "Raw Predictions")
                    st.plotly_chart(fig_raw, use_container_width=True)
                    
                    # Smoothed predictions
                    fig_smooth = create_hypnogram_plotly(smoothed_preds, "Smoothed Predictions (Mode Filter)")
                    st.plotly_chart(fig_smooth, use_container_width=True)
                
                with tab2:
                    st.subheader("Sleep Architecture Analysis")
                    
                    # Calculate statistics for smoothed predictions
                    stats = calculate_sleep_statistics(smoothed_preds)
                    
                    # Create pie chart
                    labels = list(stats.keys())
                    values = [stats[label]['percentage'] for label in labels]
                    
                    fig_pie = px.pie(
                        values=values,
                        names=labels,
                        title="Sleep Stage Distribution",
                        color_discrete_sequence=STAGE_COLORS
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)
                    
                    # Detailed statistics table
                    st.subheader("Detailed Sleep Statistics")
                    stats_data = []
                    for stage, data in stats.items():
                        stats_data.append({
                            'Sleep Stage': stage,
                            'Duration (minutes)': f"{data['minutes']:.1f}",
                            'Percentage': f"{data['percentage']:.1f}%",
                            'Epochs': data['count']
                        })
                    
                    st.dataframe(stats_data, use_container_width=True)
                
                with tab3:
                    st.subheader("Raw Prediction Data")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**First 20 Raw Predictions:**")
                        st.code([STAGE_LABELS[p] for p in raw_preds[:20]])
                    
                    with col2:
                        st.write("**First 20 Smoothed Predictions:**")
                        st.code([STAGE_LABELS[p] for p in smoothed_preds[:20]])
                    
                    # Download option
                    if st.button("📥 Download Predictions as CSV"):
                        import pandas as pd
                        df = pd.DataFrame({
                            'Epoch': range(len(raw_preds)),
                            'Raw_Prediction': [STAGE_LABELS[p] for p in raw_preds],
                            'Raw_Numeric': raw_preds,
                            'Smoothed_Prediction': [STAGE_LABELS[p] for p in smoothed_preds],
                            'Smoothed_Numeric': smoothed_preds
                        })
                        csv = df.to_csv(index=False)
                        st.download_button(
                            label="Download CSV",
                            data=csv,
                            file_name=f"sleep_stages_{selected_file.replace('.edf', '')}.csv",
                            mime="text/csv"
                        )
                
                with tab4:
                    st.subheader("About This System")
                    st.markdown("""
                    This sleep stage classification system uses a deep learning model (CNN-LSTM) 
                    to automatically analyze polysomnography (PSG) recordings and classify sleep stages.
                    
                    **Sleep Stages:**
                    - **Wake**: Conscious wakefulness
                    - **N1**: Light sleep (transition from wake to sleep)
                    - **N2**: Light sleep (sleep spindles and K-complexes)
                    - **N3**: Deep sleep (slow wave sleep)
                    - **REM**: Rapid Eye Movement sleep (dream sleep)
                    
                    **Model Features:**
                    - Uses EEG, EOG, and EMG signals
                    - 30-second epoch classification
                    - Mode filtering for noise reduction
                    - Trained on Sleep-EDF database
                    
                    **Citation:** Based on work by Karma Namgyal
                    """)
            
            else:
                st.error("❌ Analysis failed. Please check the model files and data.")

if __name__ == "__main__":
    main()