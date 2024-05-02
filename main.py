import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import board
import busio
import adafruit_mlx90640
import time
from non_ml import classify
import pickle
import pyaudio
import tflite_runtime.interpreter as tflite
import socket
import wave

FORMAT = pyaudio.paFloat32  # Typical format for microphone
CHANNELS = 1
RATE = 44100  # Sample rate
CHUNK = 1024  # Block size

#def send_file(audio_data, filename, server_ip, server_port):
def record_audio(duration=5):
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

    print("Recording...")
    frames = []
    for _ in range(int(RATE / CHUNK * duration)):
        data = stream.read(CHUNK)
        frames.append(data)

    print("Finished recording.")
    stream.stop_stream()
    stream.close()
    p.terminate()
    return b''.join(frames)

def collect_data(sensor, duration=3, frequency=15):
    # Calculate total frames to collect
    total_frames = duration * frequency
    frames = []
    
    # Start collecting frames
    start_time = time.time()
    while len(frames) < total_frames:
        frame = [0] * 768
        try:
            sensor.getFrame(frame)
            frames.append(np.reshape(frame, (24, 32)))
        except ValueError:
            continue  # Skip the frame on read error
    end_time = time.time()  # End timing
    gather_time = end_time - start_time  # Calculate inference time
    print(f'IR Time: {gather_time}')
    return frames

def IR_thread(mlx):
    # Collect data
    data_frames = collect_data(mlx)
    prediction = classify(model, data_frames)
    print(f'{"Fall Detected" if prediction[0] == 1 else "Normal Activity"}')

    # Set up the plot for the animation
    fig, ax = plt.subplots()
    heatmap = ax.imshow(data_frames[0], cmap='inferno')
    
    # Animation function to update heatmap
    def update(frame):
        heatmap.set_data(frame)
        return [heatmap]

    # Create animation
    ani = FuncAnimation(fig, update, frames=data_frames, interval=63, blit=True)
    
    # Save animation
    ani.save('heatmap_video.mp4', writer='ffmpeg', fps=15)

    plt.show()



def main():
    model_filename = "finalized_model.pkl"

    # REMOTE_IP = "172.26.128.140"
    # LOCAL_IP = "172.26.128.166"


    # REMOTE_IP = "192.168.1.154"
    REMOTE_IP = "192.168.1.155"
    PORT = 50007

    # Initialize I2C and MLX90640 sensor
    # i2c = busio.I2C(board.SCL, board.SDA)
    # mlx = adafruit_mlx90640.MLX90640(i2c)
    # mlx.refresh_rate = adafruit_mlx90640.RefreshRate.REFRESH_32_HZ

    # load model
    with open(model_filename, 'rb') as file:
        model = pickle.load(file)

    # Load the TensorFlow Lite model
    interpreter = tflite.Interpreter(model_path="ei_danger.lite")
    interpreter.allocate_tensors()

    # while True:
    raw_audio_data = record_audio()
    
    # Get input and output details from the model
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Prepare audio data for model input
    input_shape = input_details[0]['shape']
    
    raw_audio_data = np.frombuffer(raw_audio_data, dtype=np.float32)
    audio_data = np.resize(raw_audio_data, (input_details[0]['shape'][1],))
    audio_data = np.expand_dims(audio_data, axis=0)  # Reshape to [1, 16000*3]

    # Predict
    interpreter.set_tensor(input_details[0]['index'], audio_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    predicted_label_index = np.argmax(output_data)
    print("Model output:", output_data)
    print("Predicted label index:", predicted_label_index)

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.connect((REMOTE_IP, PORT))
        # Save the recorded data as a WAV file
        filename = 'a.wav'
        audio_int16 = np.int16(raw_audio_data / np.max(np.abs(raw_audio_data)) * 32767)
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(pyaudio.PyAudio().get_sample_size(pyaudio.paInt16))
            wf.setframerate(RATE)
            wf.writeframes(audio_int16)

        print(f"Audio saved to {filename}")

        # Send the file
        with open(filename, 'rb') as f:
            while True:
                data = f.read(1024)
                if not data:
                    break
                sock.sendall(data)
        print(f"File {filename} sent successfully.")

        sock.shutdown(socket.SHUT_WR)

        response = sock.recv(1024)
        print("Server response:", response.decode())

        # Act based on the server's response
        if response.decode() == "File received successfully.":
            print("Action confirmed: Server has received the file.")
        else:
            print("Action required: Check file integrity or resend.")

        # start_time = time.time()  # Start timing
        
        # end_time = time.time()  # End timing
        # inference_time = end_time - start_time  # Calculate inference time
        

if __name__ == "__main__":
    main()
