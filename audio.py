import pyaudio
from pydub import AudioSegment
import wave

FORMAT = pyaudio.paInt16  # Audio format (16-bit PCM)
CHANNELS = 1              # Number of audio channels
RATE = 44100              # Sample rate (44100 samples per second)
CHUNK = 1024              # Number of frames per buffer
RECORD_SECONDS = 5        # Duration of recording
WAVE_OUTPUT_FILENAME = "output.wav"
MP3_OUTPUT_FILENAME = "output.mp3"

audio = pyaudio.PyAudio()

# Start Recording
stream = audio.open(format=FORMAT, channels=CHANNELS,
                    rate=RATE, input=True,
                    frames_per_buffer=CHUNK)
print("recording...")
frames = []

for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    data = stream.read(CHUNK)
    frames.append(data)
print("finished recording")

# Stop Recording
stream.stop_stream()
stream.close()
audio.terminate()

# Save the recorded data as a WAV file
wave_file = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
wave_file.setnchannels(CHANNELS)
wave_file.setsampwidth(audio.get_sample_size(FORMAT))
wave_file.setframerate(RATE)
wave_file.writeframes(b''.join(frames))
wave_file.close()

# Convert WAV to MP3
sound = AudioSegment.from_wav(WAVE_OUTPUT_FILENAME)
sound.export(MP3_OUTPUT_FILENAME, format="mp3")

print(f"File saved as {MP3_OUTPUT_FILENAME}")