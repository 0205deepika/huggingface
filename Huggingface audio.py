from transformers import AutoProcessor, AutoModelForTextToSpeech
import torch
import torchaudio

# Load the Hugging Face model and processor
model_id = "facebook/mms-tts-eng"  # Change this to another model if needed
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForTextToSpeech.from_pretrained(model_id)

# Input text
input_text = "I love data science and machine learning"

# Convert text to tokens
inputs = processor(input_text, return_tensors="pt")

# Generate speech waveform
with torch.no_grad():
    speech = model(**inputs).waveform

# Save the generated speech as an audio file
speech_file_path = "speech.wav"
torchaudio.save(speech_file_path, speech, sample_rate=16000)

print("Speech synthesis complete! File saved as", speech_file_path)
