# TextAudioTranscription
# Text-to-Audio-to-Text Transcription Project

This project demonstrates a pipeline for converting text to audio, transcribing the audio back to text, and evaluating the accuracy of the transcription using **Word Error Rate (WER)**. It uses the following technologies:
- **Text-to-Speech (TTS)**: Converts text into an audio file using `gTTS`.
- **Speech-to-Text (STT)**: Transcribes audio back into text using the **Wav2Vec 2.0** model from Hugging Face.
- **Evaluation**: Compares the original text and generated text using **Word Error Rate (WER)**.

## Features
1. **Text-to-Audio Conversion**:
   - Converts a text file (e.g., `sample_text.txt`) into an audio file (`.flac` format).
2. **Audio-to-Text Transcription**:
   - Transcribes the generated audio file back into text using the Wav2Vec 2.0 model.
3. **Evaluation**:
   - Compares the original text and generated text to calculate the **Word Error Rate (WER)**.

## Prerequisites
Before running the project, ensure you have the following installed:
- Python 3.8 or higher
- Required Python libraries (listed in `requirements.txt`)

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/text-to-audio-to-text.git
   cd text-to-audio-to-text

