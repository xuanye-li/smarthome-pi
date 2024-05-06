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
import threading
import os
import sys
import json
import requests
import csv
from scipy.ndimage import gaussian_filter
import warnings
import cv2

FORMAT = pyaudio.paFloat32  # Typical format for microphone
CHANNELS = 1
RATE = 44100  # Sample rate
CHUNK = 1024  # Block size

def save_frames_to_csv(frames, filename='output.csv'):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        for frame in frames:
            # Flatten each frame from 24x32 to a single row of 768 elements
            flattened_frame = frame.flatten()
            writer.writerow(flattened_frame)

def send_to_webapp(label, file_path):
    url = 'http://172.26.173.56:8000/receive/'
    data = {'label': label}  # This is the form data

    # Determine file type based on label
    if label == 'fall':
        file_type = 'video/mp4'
    else:
        file_type = 'audio/wav'

    # Using a context manager to ensure the file is closed after the request
    with open(file_path, 'rb') as f:
        files = {'media_file': (file_path, f, file_type)}
        response = requests.post(url, data=data, files=files)

    print("Response received")
    try:
        print(response.json())
    except Exception as e:
        print(f"Failed to decode JSON from response: {e}")
        print("Response content:", response.text)

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

def create_wav(raw_audio_data, filename):
    audio_int16 = np.int16(raw_audio_data / np.max(np.abs(raw_audio_data)) * 32767)
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(pyaudio.PyAudio().get_sample_size(pyaudio.paInt16))
        wf.setframerate(RATE)
        wf.writeframes(audio_int16)

    print(f"Audio saved to {filename}")

def audio_thread(interpreter):
    # REMOTE_IP = "192.168.1.155"
    REMOTE_IP = "172.26.189.161"
    PORT = 50007
    CONFIDENCE = 0.6
    filename = 'recording.wav'
    labels = ['crying', 'glass breaking', 'gunshot', 'normal']

    while True:
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

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.connect((REMOTE_IP, PORT))
            # Save the recorded data as a WAV file
            create_wav(raw_audio_data, filename)

            # Send the file
            with open(filename, 'rb') as f:
                while True:
                    data = f.read(1024)
                    if not data:
                        break
                    sock.sendall(data)
            print(f"File {filename} sent successfully.")

            sock.shutdown(socket.SHUT_WR)
            print("waiting for response")
            response = sock.recv(1024)
            CLAP_label = response.decode()
            print("CLAP response:", CLAP_label)
            if CLAP_label != 'Normal':
                send_to_webapp(CLAP_label, filename)
        

        # if predicted_label_index != 3:
        #     print("Predicted label index:", predicted_label_index)
        #     create_wav(raw_audio_data, filename)
        #     send_to_webapp(labels[predicted_label_index], filename)

        # elif output_data[0, predicted_label_index] < CONFIDENCE:
        #     print("Predicted label index:", predicted_label_index)
        #     try:
        #         with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        #             sock.connect((REMOTE_IP, PORT))
        #             # Save the recorded data as a WAV file
        #             create_wav(raw_audio_data, filename)

        #             # Send the file
        #             with open(filename, 'rb') as f:
        #                 while True:
        #                     data = f.read(1024)
        #                     if not data:
        #                         break
        #                     sock.sendall(data)
        #             print(f"File {filename} sent successfully.")

        #             sock.shutdown(socket.SHUT_WR)

        #             response = sock.recv(1024)
        #             CLAP_label = response.decode()
        #             print("CLAP response:", CLAP_label)
        #             if CLAP_label != 'normal':
        #                 send_to_webapp(CLAP_label, filename)
        #     except Exception as e:
        #         print(f"Error in audio_thread: {e}")
        # else:
        #     create_wav(raw_audio_data, filename)            


# Function to preprocess data frames and detect falls
def detect_fall(binary_frames):
    FALL_THRESHOLD = 5
    FRAME_THRESHOLD = 6
    # List to store centroids
    centroids = []

    # Loop through each binary frame
    for binary_frame in binary_frames:
        # Find contours in the binary image
        contours, _ = cv2.findContours(binary_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Loop through each contour
        for contour in contours:
            # Get centroid of the contour
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                centroids.append((cX, cY))  # Save centroid coordinates
    if len(centroids) == 0:
        return False, None
    starting_centroid = centroids[0]
    falling_frames = 0
    # Check if any centroid exceeds the fall threshold
    for centroid in centroids:
        if centroid[1] - starting_centroid[1]  > FALL_THRESHOLD:
            falling_frames += 1  # Fall detected and return all centroids
    if falling_frames > FRAME_THRESHOLD:
        return True, centroids
    return False, centroids  # No fall detected

def main():
    model_filename = "finalized_model.pkl"

    WEBAPP_IP = "172.26.173.56"
    # LOCAL_IP = "172.26.128.166"
    # REMOTE_IP = "192.168.1.154"

    # Initialize I2C and MLX90640 sensor
    i2c = busio.I2C(board.SCL, board.SDA)
    mlx = adafruit_mlx90640.MLX90640(i2c)
    mlx.refresh_rate = adafruit_mlx90640.RefreshRate.REFRESH_32_HZ

    # # load model
    # with open(model_filename, 'rb') as file:
    #     model = pickle.load(file)

    # Load the TensorFlow Lite model
    interpreter = tflite.Interpreter(model_path="danger_float.lite")
    interpreter.allocate_tensors()

    audio_processing  = threading.Thread(target=audio_thread, args=(interpreter,))
    audio_processing.start()
    while True:
        TEMP_THRESHOLD = 25.5
        data_frames = collect_data(mlx)

        # Preprocess data frames with Gaussian filtering
        smoothed_frames = [gaussian_filter(frame, sigma=1.5) for frame in data_frames]

        # Thresholding to create binary image
        binary_frames = [(frame > TEMP_THRESHOLD).astype(np.uint8) for frame in smoothed_frames]

        fall, centroids = detect_fall(binary_frames)

        if (fall):
            print("Fall Detected")
            # Set up the plot for the animation
            fig, ax = plt.subplots()
            heatmap = ax.imshow(smoothed_frames[0], cmap='inferno')
            
            
            # Add colorbar indicating temperature range
            cbar = fig.colorbar(heatmap, ax=ax)
            cbar.set_label('Temperature (°C)')
            
            # Initialize centroid plot
            centroid_plot, = ax.plot([], [], 'bo', markersize=15)  # Blue circle marker
            
            # Animation function to update heatmap and centroid
            def update(frame):
                heatmap.set_data(smoothed_frames[frame])    
                # Plot centroids for the current frame
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    try:
                        c = centroids[frame]
                        centroid_plot.set_data(c)
                    except:
                        return [heatmap, centroid_plot]
                
                return [heatmap, centroid_plot]

            # Create animation
            #print("creating animation")
            ani = FuncAnimation(fig, update, frames=len(smoothed_frames), interval=63, blit=True)
            
            # Save animation
            ani.save('heatmap_video.mp4', writer='ffmpeg', fps=15)
            send_to_webapp('fall', 'heatmap_video.mp4')
        else:
            print("No Fall")


        # prediction = classify(model, data_frames)

        # if prediction[0] == 1:
        #     print("Fall Detected")
        #     # Set up the plot for the animation
        #     fig, ax = plt.subplots()
        #     heatmap = ax.imshow(data_frames[0], cmap='inferno')
            
        #     # Animation function to update heatmap
        #     def update(frame):
        #         heatmap.set_data(frame)
        #         return [heatmap]

        #     # Create animation
        #     ani = FuncAnimation(fig, update, frames=data_frames, interval=63, blit=True)
        #     # Save animation
        #     ani.save('heatmap_video.mp4', writer='ffmpeg', fps=15)
        #     plt.show()
        #     send_to_webapp('fall', 'heatmap_video.mp4')

        # else:
        #     print(f'{"No Fall"}')
        

if __name__ == "__main__":
    main()
