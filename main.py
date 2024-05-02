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


FORMAT = pyaudio.paInt16  # Typical format for microphone
CHANNELS = 1
RATE = 44100  # Sample rate
CHUNK = 1024  # Block size

def send_file(filename, server_ip, server_port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.connect((server_ip, server_port))
        with open(filename, 'rb') as f:
            while True:
                data = f.read(1024)
                if not data:
                    break
                sock.sendall(data)
        print(f"File {filename} sent successfully.")

def record_audio(duration=5, sample_rate=44100):
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paFloat32,
                    channels=1,
                    rate=sample_rate,
                    input=True,
                    frames_per_buffer=1024)

    print("Recording...")
    frames = []
    for _ in range(int(sample_rate / 1024 * duration)):
        data = stream.read(1024)
        frames.append(np.frombuffer(data, dtype=np.float32))

    print("Finished recording.")
    stream.stop_stream()
    stream.close()
    p.terminate()

    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(p.get_sample_size(pyaudio.paFloat32))
        wf.setframerate(sample_rate)
        wf.writeframes(b''.join(frames))
    print(f"Audio saved to {filename}")

    return np.concatenate(frames)

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
    print(gather_time)
    return frames

def main():
    model_filename = "finalized_model.pkl"

    REMOTE_IP = "172.26.128.140"
    PORT = 50007
    LOCAL_IP = "172.26.128.166"
    # Initialize I2C and MLX90640 sensor
    i2c = busio.I2C(board.SCL, board.SDA)
    mlx = adafruit_mlx90640.MLX90640(i2c)
    mlx.refresh_rate = adafruit_mlx90640.RefreshRate.REFRESH_32_HZ

    # load model
    with open(model_filename, 'rb') as file:
        model = pickle.load(file)

    # Load the TensorFlow Lite model
    interpreter = tflite.Interpreter(model_path="ei_danger.lite")
    interpreter.allocate_tensors()

    while True:
        audio_data = record_audio()
        # Collect data
        data_frames = collect_data(mlx)

        # Get input and output details from the model
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        # Prepare audio data for model input
        input_shape = input_details[0]['shape']
        
        audio_data = np.resize(audio_data, (input_shape[1],))
        audio_data = np.expand_dims(audio_data, axis=0)  # Reshape to [1, 16000*3]

        # Predict
        interpreter.set_tensor(input_details[0]['index'], audio_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        predicted_label_index = np.argmax(output_data)

        # start_time = time.time()  # Start timing
        prediction = classify(model, data_frames)

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.connect((REMOTE_IP, PORT))
            sock.sendall(data)
        # end_time = time.time()  # End timing
        # inference_time = end_time - start_time  # Calculate inference time

        print(predicted_label_index)
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


if __name__ == "__main__":
    main()
