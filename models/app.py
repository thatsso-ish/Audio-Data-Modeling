from flask import Flask, request, render_template, jsonify
import os
import librosa
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import pickle
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename
import warnings

# Suppress specific warnings
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load pre-trained model and label encoder
model = load_model('saved_models/hybrid_yamnet.h5')
with open('saved_encoders/ylabel_e.pkl', 'rb') as le_file:
    label_encoder = pickle.load(le_file)

# Load YAMNet model from TensorFlow Hub
yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')

# Helper function to extract YAMNet features
def extract_yamnet_features(file_path):
    y, sr = librosa.load(file_path, sr=16000)
    waveform = tf.convert_to_tensor(y, dtype=tf.float32)
    _, embeddings, _ = yamnet_model(waveform)
    embedding = np.mean(embeddings, axis=0)
    return embedding, y, sr  # Also return the audio for waveform and Mel-spectrogram plotting

# Helper function to plot waveform and Mel-spectrogram for the dominating class
def plot_waveform_and_mel_spectrogram(file_path, predicted_class, y, sr):
    if predicted_class != 10:  # Skip 'neutral class'
        plt.figure(figsize=(14, 6))

        # Plot Waveform
        plt.subplot(1, 2, 1)
        librosa.display.waveshow(y, sr=sr, alpha=0.6)
        plt.title(f'Waveform - Class: {predicted_class}')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')

        # Plot Mel-spectrogram
        plt.subplot(1, 2, 2)
        mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
        mel_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
        librosa.display.specshow(mel_db, sr=sr, x_axis='time', y_axis='mel', cmap='magma')
        plt.colorbar(format='%+2.0f dB')
        plt.title(f'Mel-spectrogram - Class: {predicted_class}')
        plt.tight_layout()

        buffer = BytesIO()
        plt.savefig(buffer, format="png")
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        plt.close()
        return image_base64
    return None

# Helper function to generate percentage distribution
def generate_percentage_distribution(predictions):
    distribution = {}
    for i, prob in enumerate(predictions[0]):
        class_name = label_encoder.inverse_transform([i])[0]
        distribution[class_name] = round(prob * 100, 2)
    return distribution

# Route for uploading file and displaying results
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Extract features and classify
            features, y, sr = extract_yamnet_features(filepath)
            features = features.reshape(1, -1)
            predictions = model.predict(features)
            predicted_class = np.argmax(predictions, axis=1)
            confidence = np.max(predictions)

            # Assign neutral class if confidence is below threshold
            if confidence < 0.5:  # Adjust threshold as needed
                class_name = "Neutral Class"
                predicted_class = 10
            else:
                class_name = label_encoder.inverse_transform(predicted_class)[0]

            # Plot visuals for dominating class
            image_base64 = plot_waveform_and_mel_spectrogram(filepath, predicted_class, y, sr)

            # Prepare audio for playback
            with open(filepath, "rb") as audio_file:
                audio_base64 = base64.b64encode(audio_file.read()).decode("utf-8")

            # Generate percentage distribution
            percentage_distribution = generate_percentage_distribution(predictions)

            # Find and highlight dominating sound
            dominant_class = max(percentage_distribution, key=percentage_distribution.get)
            dominant_percentage = percentage_distribution[dominant_class]

            return render_template(
                'result.html',
                class_name=dominant_class,
                confidence=dominant_percentage,
                audio_base64=audio_base64,
                image_base64=image_base64,
                percentage_distribution=percentage_distribution
            )

    return render_template('index.html')

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
