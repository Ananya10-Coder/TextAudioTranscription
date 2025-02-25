# Text-to-Audio-to-Text Transcription Project

This project demonstrates a pipeline for converting text to audio, transcribing the audio back to text, and evaluating the accuracy of the transcription using **Word Error Rate (WER)**. It leverages modern technologies like **gTTS** for text-to-speech conversion and **Wav2Vec 2.0** for speech-to-text transcription.

---

## Table of Contents
- [Overview](#overview)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [Workflow](#workflow)
- [Evaluation](#evaluation)
- [License](#license)

---

## Overview

The project provides a Python script that performs the following tasks:
1. **Text-to-Speech (TTS)**: Converts a given text into an audio file using **gTTS**.
2. **Speech-to-Text (STT)**: Transcribes the generated audio back into text using the **Wav2Vec 2.0** model from Hugging Face.
3. **Evaluation**: Compares the original text with the transcribed text and calculates the **Word Error Rate (WER)** to evaluate the accuracy of the transcription.

This pipeline is useful for testing the performance of speech-to-text models and understanding the challenges of audio transcription.

---

## Technologies Used

- **Text-to-Speech (TTS)**: [gTTS](https://gtts.readthedocs.io/) (Google Text-to-Speech)
- **Speech-to-Text (STT)**: [Wav2Vec 2.0](https://huggingface.co/facebook/wav2vec2-large-960h) from Hugging Face Transformers
- **Audio Processing**: [Librosa](https://librosa.org/) for audio loading and normalization
- **Evaluation**: [Word Error Rate (WER)](https://huggingface.co/spaces/evaluate-metric/wer) using the `evaluate` library
- **Dependencies**: PyTorch, Torchaudio, Transformers, NumPy

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/TextAudioTranscription.git
   cd TextAudioTranscription
   ```

2. Install the required Python libraries:
   ```bash
   pip install torch torchaudio transformers librosa gtts evaluate jiwer
   ```

---

## Usage

1. Place the text you want to convert in a file named `sample.txt` in the same directory as the script.

2. Run the script:
   ```bash
   python text_audio_transcription.py
   ```

3. The script will:
   - Generate an audio file (`output_audio.flac`) from the text.
   - Transcribe the audio back to text.
   - Print the original text, the transcribed text, and the Word Error Rate (WER).

---

## Workflow

1. **Read Text File**: The script reads the text from `sample.txt`.
2. **Text-to-Speech (TTS)**: The text is converted to speech using **gTTS** and saved as `output_audio.flac`.
3. **Speech-to-Text (STT)**: The audio file is transcribed back to text using the **Wav2Vec 2.0** model.
4. **Evaluation**: The script compares the original text with the transcribed text and calculates the **Word Error Rate (WER)**.

---

## Evaluation

The accuracy of the transcription is evaluated using the **Word Error Rate (WER)** metric. WER is calculated as:
\[
\text{WER} = \frac{\text{Number of Errors (Insertions + Deletions + Substitutions)}}{\text{Total Number of Words in Original Text}}
\]

A lower WER indicates better transcription accuracy.

---

## License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.
