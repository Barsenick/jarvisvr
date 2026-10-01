import numpy as np
import sounddevice as sd
import time
from jarvis_voice.config import Config
from jarvis_voice.stt import FasterWhisperSTT
from jarvis_voice.tts import PiperTTS
from jarvis_voice.audio import play_wav_bytes

def main():
    print('Initializing STT and TTS...')
    cfg = Config()
    # Override to use our models if needed (should already be set via .env)
    stt = FasterWhisperSTT(cfg)
    tts = PiperTTS(cfg)
    print('STT and TTS initialized.')

    # Audio settings
    sample_rate = cfg.sample_rate  # 16000
    duration = 5  # seconds to record
    print(f'Recording {duration} seconds of audio...')
    print('Speak now...')

    # Record audio
    recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='int16')
    sd.wait()  # Wait until recording is finished
    print('Recording finished.')

    # Flatten and convert to bytes
    audio_buffer = recording.flatten().tobytes()

    # Transcribe
    print('Transcribing...')
    transcript_result = stt.transcribe(audio_buffer)
    transcription = transcript_result.text.strip()
    print(f'You said: {transcription}')

    if transcription:
        # Create a response
        response_text = f'You said: {transcription}'
        print(f'Synthesizing: {response_text}')
        wav_data = tts.synthesize(response_text)
        print(f'Playing response ({len(wav_data)} bytes)...')
        play_wav_bytes(wav_data, output_device=cfg.output_device)
        print('Response played.')
    else:
        print('No speech detected. Try again.')

if __name__ == '__main__':
    main()
