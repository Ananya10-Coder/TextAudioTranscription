# Create a sample.txt file in the same directory as the notebook
!echo "THIS IS A SAMPLE TEXT FOR SPEECH to TEXT CONVERSION" > sample.txt
# Step 1: Install Required Libraries
!pip install torch torchaudio transformers librosa gtts evaluate jiwer

# Step 2: Import Libraries
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from gtts import gTTS  # For text-to-speech
import librosa
import numpy as np
import evaluate  # For Word Error Rate (WER)
import jiwer

# Step 3: Load Pre-trained Wav2Vec 2.0 Model and Processor
model_name = "facebook/wav2vec2-large-960h"  # Pre-trained Wav2Vec 2.0 model
processor = Wav2Vec2Processor.from_pretrained(model_name)
model = Wav2Vec2ForCTC.from_pretrained(model_name)

# Step 4: Read a Text File
def read_text_file(file_path):
    with open(file_path, "r") as file:
        text = file.read()
    return text

# Step 5: Convert Text to Audio (Text-to-Speech)
def text_to_audio(text, output_audio_file):
    tts = gTTS(text=text, lang="en")  # Convert text to speech
    tts.save(output_audio_file)  # Save as .flac file
    print(f"Audio file saved: {output_audio_file}")

# Step 6: Transcribe Audio to Text
def transcribe_audio(audio_file):
    # Load and preprocess audio
    audio, sr = librosa.load(audio_file, sr=16000)  # Resample to 16kHz
    audio = librosa.util.normalize(audio)  # Normalize audio

    # Convert audio to input features
    input_values = processor(audio, return_tensors="pt", sampling_rate=16000).input_values

    # Perform inference
    with torch.no_grad():
        logits = model(input_values).logits

    # Decode the predicted text
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.decode(predicted_ids[0])
    return transcription

# Step 7: Compare Original Text and Generated Text
def evaluate_transcription(original_text, generated_text):
    wer = evaluate.load("wer").compute(predictions=[generated_text], references=[original_text])
    print(f"Word Error Rate (WER): {wer * 100:.2f}%")

# Step 8: Main Workflow
def main():
    # Step 8.1: Read the text file
    text_file = "sample.txt"  # Replace with your text file path
    original_text = read_text_file(text_file)
    print("Original Text:")
    print(original_text)

    # Step 8.2: Convert text to audio
    output_audio_file = "output_audio.flac"
    text_to_audio(original_text, output_audio_file)

     # Step 8.3: Transcribe audio back to text
    generated_text = transcribe_audio(output_audio_file)
    print("Generated Text:")
    print(generated_text)

    # Step 8.4: Compare original and generated text
    evaluate_transcription(original_text, generated_text)

# Run the program
main()
