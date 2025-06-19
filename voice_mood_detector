import sounddevice as sd
import numpy as np
import torch
import librosa
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, pipeline
from textblob import TextBlob

# Define mood keywords
mood_keywords = {
    "happy": ["joy", "glad", "cheerful", "excited", "delighted", "grateful", "content", "ecstatic", "elated", "happy"],
    "sad": ["down", "depressed", "unhappy", "blue", "melancholy", "heartbroken", "gloomy", "miserable", "crying", "sad"],
    "angry": ["mad", "furious", "irritated", "annoyed", "frustrated", "enraged", "agitated", "angry"],
    "calm": ["relaxed", "peaceful", "chill", "easygoing", "serene", "laid-back", "calm"],
    "anxious": ["nervous", "worried", "uneasy", "panicked", "scared", "stressed", "anxious"],
    "tired": ["exhausted", "fatigued", "sleepy", "drained", "burned out", "tired"],
    "neutral": ["fine", "okay", "meh", "average", "normal", "neutral", "just there"]
}

def correct_spelling(text):
    return str(TextBlob(text).correct())

def map_to_mood(text, mood_keywords):
    words = text.lower().split()
    for word in words:
        for mood, keywords in mood_keywords.items():
            if word in keywords:
                return mood
    return "neutral"

def get_voice_mood():
    duration = 10  # seconds
    fs = 16000     # sample rate

    print("🎤 Recording..How are you feeling?...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    print("✅ Recording finished!")

    audio = audio.squeeze()

    print("🔍 Detecting mood ...")
    asr_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    asr_model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

    input_values = asr_processor(audio, sampling_rate=fs, return_tensors="pt").input_values

    with torch.no_grad():
        logits = asr_model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = asr_processor.decode(predicted_ids[0])
    print(f"📝 You said: {transcription}")

    corrected_transcription = correct_spelling(transcription)
    print(f"✅ Corrected: {corrected_transcription}")

    mapped_mood = map_to_mood(corrected_transcription, mood_keywords)
    print(f"🔍 Mapped Mood: {mapped_mood}")

    print("🔍 Loading emotion detection pipeline...")
    emotion_classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=True)

    emotion_scores = emotion_classifier(transcription)[0]
    detected_emotion = max(emotion_scores, key=lambda x: x['score'])['label']

    final_mood = mapped_mood if mapped_mood != "neutral" else detected_emotion.lower()
    print(f"🎶 Playing music for the detected mood: {final_mood}")
    return final_mood
