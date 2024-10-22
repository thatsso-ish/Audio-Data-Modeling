# app.py

import streamlit as st
import librosa
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import librosa.display
import sounddevice as sd
import wavio
import os

# -------------------------------
# Function Definitions
# -------------------------------

@st.cache_resource
def load_resources():
    """
    Load the trained Hybrid CNN-RNN model and LabelEncoder.
    """
    try:
        model = load_model('saved_models/final_hybrid_cnn_rnn_model.keras')
        with open('saved_encoders/label_encoder.pkl', 'rb') as le_file:
            le = pickle.load(le_file)
        return model, le
    except Exception as e:
        st.error(f"Error loading resources: {e}")
        return None, None

def preprocess_audio(audio_file, n_mfcc=40, max_pad_len=174):
    """
    Preprocess the audio file to extract MFCC features suitable for the model.
    
    Parameters:
        audio_file (str or file-like object): Path to the audio file or a file-like object.
        n_mfcc (int): Number of MFCC features to extract.
        max_pad_len (int): Maximum length of MFCC features to pad/truncate.
    
    Returns:
        tuple: (features, y, sr)
            - features (np.ndarray): Preprocessed MFCC features reshaped for the model.
            - y (np.ndarray): Audio time series.
            - sr (int): Sampling rate of y.
    """
    try:
        # Load audio file
        y, sr = librosa.load(audio_file, sr=None)
        
        # Extract MFCC features
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        
        # Normalize MFCC
        mfcc = (mfcc - np.mean(mfcc)) / np.std(mfcc)
        
        # Pad or truncate to ensure fixed length
        pad_width = max_pad_len - mfcc.shape[1]
        if pad_width > 0:
            mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
        else:
            mfcc = mfcc[:, :max_pad_len]
        
        # Reshape for model input
        features = mfcc.reshape(1, n_mfcc, max_pad_len, 1)
        
        return features, y, sr
    except Exception as e:
        st.error(f"Error processing audio file: {e}")
        return None, None, None

def record_audio(duration=10, fs=44100):
    """
    Record audio from the microphone.
    
    Parameters:
        duration (int): Duration of the recording in seconds.
        fs (int): Sampling rate.
    
    Returns:
        str: Path to the recorded audio file.
    """
    try:
        st.info(f"Recording for {duration} seconds...")
        recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
        sd.wait()  # Wait until recording is finished
        audio_file = 'recorded_audio.wav'
        wavio.write(audio_file, recording, fs, sampwidth=2)  # Save as WAV file
        st.success("Recording complete!")
        return audio_file
    except Exception as e:
        st.error(f"Error recording audio: {e}")
        return None

def plot_waveform(y, sr):
    """
    Plot the waveform of the audio.
    
    Parameters:
        y (np.ndarray): Audio time series.
        sr (int): Sampling rate.
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    librosa.display.waveshow(y, sr=sr, ax=ax)
    ax.set(title='Waveform')
    plt.tight_layout()
    st.pyplot(fig)

def plot_mfcc(y, sr, n_mfcc=40):
    """
    Plot the MFCC spectrogram of the audio.
    
    Parameters:
        y (np.ndarray): Audio time series.
        sr (int): Sampling rate.
        n_mfcc (int): Number of MFCC features to extract.
    """
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    fig, ax = plt.subplots(figsize=(10, 4))
    img = librosa.display.specshow(mfcc, x_axis='time', ax=ax, cmap='viridis')
    fig.colorbar(img, ax=ax)
    ax.set(title='MFCC Spectrogram')
    plt.tight_layout()
    st.pyplot(fig)

# -------------------------------
# Load Model and Encoder
# -------------------------------

model, le = load_resources()

if model is None or le is None:
    st.stop()  # Stop execution if resources couldn't be loaded

# -------------------------------
# Streamlit App Layout
# -------------------------------

st.title("Hybrid CNN-RNN Audio Classification")
st.write("Upload an audio file or record one to classify its sound event.")

# Add options to upload or record audio
option = st.radio("Choose an option:", ('Upload Audio', 'Record Audio'))

uploaded_file = None  # Initialize

if option == 'Upload Audio':
    uploaded_file = st.file_uploader("Choose an audio file...", type=["wav", "mp3"])
elif option == 'Record Audio':
    if st.button("Record Audio"):
        recorded_file = record_audio(duration=5)  # You can adjust the duration as needed
        if recorded_file:
            uploaded_file = recorded_file

if uploaded_file is not None:
    # Display the uploaded or recorded audio
    try:
        if isinstance(uploaded_file, str):
            # Recorded audio file path
            st.audio(uploaded_file, format='audio/wav')
            audio_path = uploaded_file
        else:
            # Uploaded audio file (BytesIO object)
            st.audio(uploaded_file, format='audio/wav')
            audio_path = uploaded_file
    except Exception as e:
        st.error(f"Error displaying audio: {e}")
        st.stop()
    
    # Preprocess the audio
    features, y, sr = preprocess_audio(audio_path)
    
    if features is not None:
        # Make prediction
        try:
            prediction = model.predict(features)
            predicted_class = np.argmax(prediction, axis=1)
            confidence = np.max(prediction) * 100

            # Decode the labels
            predicted_label = le.inverse_transform(predicted_class)[0]

            # Display prediction
            st.success(f"**Predicted Class:** {predicted_label}")
            st.info(f"**Confidence:** {confidence:.2f}%")
        except Exception as e:
            st.error(f"Error making prediction: {e}")
        
        # Display waveform
        st.write("### Waveform:")
        plot_waveform(y, sr)
        
        # Display MFCC spectrogram
        st.write("### MFCC Spectrogram:")
        plot_mfcc(y, sr)