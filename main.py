import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import board
import busio
import adafruit_mlx90640
import time
from non_ml import classify
import pickle

model_filename = "finalized_model.pkl"

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
    # Initialize I2C and MLX90640 sensor
    i2c = busio.I2C(board.SCL, board.SDA)
    mlx = adafruit_mlx90640.MLX90640(i2c)
    mlx.refresh_rate = adafruit_mlx90640.RefreshRate.REFRESH_32_HZ
    
    with open(model_filename, 'rb') as file:
        model = pickle.load(file)

    while True:
        # Collect data
        data_frames = collect_data(mlx)

        # start_time = time.time()  # Start timing
        prediction = classify(model, data_frames)
        # end_time = time.time()  # End timing
        # inference_time = end_time - start_time  # Calculate inference time
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
