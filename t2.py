import torch
from pyannote.audio import Pipeline
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import librosa
import os
import noisereduce as nr
from pydub import AudioSegment
from pydub.effects import high_pass_filter, low_pass_filter

pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization")

processor = Wav2Vec2Processor.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-russian")
model = Wav2Vec2ForCTC.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-russian")

audio_dir = "C:\\onemoretest\\t2\\audio"
results_dir = "C:\\onemoretest\\t2\\results"

os.makedirs(results_dir, exist_ok=True)

def transcribe(audio_segment):
    input_values = processor(audio_segment, sampling_rate=16000, return_tensors="pt").input_values
    with torch.no_grad():
        logits = model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.decode(predicted_ids[0])
    return transcription.lower()

def denoise_audio(audio, sr):
    # Удаление шума
    reduced_noise = nr.reduce_noise(y=audio, sr=sr)
    return reduced_noise

def filter_audio(audio_path):
    # Загрузка аудио с помощью pydub для применения фильтров
    audio = AudioSegment.from_file(audio_path)
    
    filtered_audio = high_pass_filter(audio, cutoff=300)  # Убираем частоты ниже 300 Гц
    
    filtered_audio = low_pass_filter(filtered_audio, cutoff=3000)  # Убираем частоты выше 3000 Гц
    
    filtered_audio_path = "filtered_temp.wav"
    filtered_audio.export(filtered_audio_path, format="wav")
    
    return filtered_audio_path

for audio_file in os.listdir(audio_dir):
    if audio_file.endswith(".wav"):
        audio_path = os.path.join(audio_dir, audio_file)
        
        filtered_audio_path = filter_audio(audio_path)
        
        diarization = pipeline(filtered_audio_path)
        
        audio, sr = librosa.load(filtered_audio_path, sr=16000)
        
        audio = denoise_audio(audio, sr)
        
        segments = {}
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            start_sample = int(turn.start * sr)
            end_sample = int(turn.end * sr)
            if speaker not in segments:
                segments[speaker] = []
            segments[speaker].append(audio[start_sample:end_sample])
        
        transcriptions = {}
        for speaker, audio_segments in segments.items():
            transcriptions[speaker] = []
            for segment in audio_segments:
                if len(segment) > 320:  # Проверка длины сегмента
                    try:
                        transcription = transcribe(segment)
                        transcriptions[speaker].append(transcription)
                    except Exception as e:
                        print(f"Error processing segment for speaker {speaker}: {e}")
        
        result_file = os.path.join(results_dir, os.path.splitext(audio_file)[0] + ".txt")
        with open(result_file, 'w', encoding='utf-8') as f:
            for speaker, texts in transcriptions.items():
                f.write(f"Speaker {speaker}:\n")
                for text in texts:
                    f.write(f" - {text}\n")
        
        print(f"Results for {audio_file} saved to {result_file}")
        
        os.remove(filtered_audio_path)
