import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Function to manually read metadata and data
def read_csv_with_metadata(path):
    with open(path, 'r') as file:
        metadata_lines = [next(file) for _ in range(2)]  # Read the first two lines as metadata
        data = pd.read_csv(file, header=None)  # The rest is the data
    
    metadata = [line.strip().split(',') for line in metadata_lines]
    return metadata, data

# Configuration
original_csv_path = 'melexis_csv/modified_melexis-G1-2-f03.csv'  # Path to the original CSV file
fall_start_frame = 485  # Update with the starting frame of the fall
fall_duration = 45  # Duration of the fall in frames
fall_number = 2  # Identifier for the fall
classname = 'fall' 

# Extract the base filename without the extension to use in output filenames
base_filename = os.path.splitext(os.path.basename(original_csv_path))[0]

# Read the original data with metadata
metadata, data = read_csv_with_metadata(original_csv_path)

# Calculate the end frame for the selected fall
fall_end_frame = fall_start_frame + fall_duration

# Extract frames for the fall
fall_data = data.iloc[fall_start_frame - 3 : fall_start_frame + fall_duration - 3]  # Adjust for zero indexing and metadata

# Save the trimmed data to a new CSV, including metadata
trimmed_csv_path = f'melexis_falls/{base_filename}_{classname}_{fall_number}_frames_{fall_start_frame}_{fall_end_frame-1}.csv'
with open(trimmed_csv_path, 'w') as f:
    for line in metadata:  # Write metadata lines
        f.write(','.join(line) + '\n')
    fall_data.to_csv(f, index=False, header=False)  # Write fall data
print(f"Trimmed CSV saved to {trimmed_csv_path}")

video_file_path  = f'melexis_falls/{base_filename}_{classname}_{fall_number}_frames_{fall_start_frame}_{fall_end_frame-1}.mp4'
# Prepare data for video
temperature_data = fall_data.iloc[:, 1:].to_numpy()

vmin = temperature_data.min()
vmax = temperature_data.max()

fig, ax = plt.subplots()
frame_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, color="white")
im = ax.imshow(np.zeros((24, 32)), cmap='hot', interpolation='nearest', vmin=vmin, vmax=vmax)

def init():
    im.set_data(np.zeros((24, 32)))  # Adjusted to 24x32
    frame_text.set_text('')
    return [im, frame_text]

def animate(i):
    heatmap_data = temperature_data[i].reshape(24, 32)  # Adjusted to 24x32
    im.set_data(heatmap_data)
    frame_text.set_text(f'Frame {i+1}')
    return [im, frame_text]

ani = animation.FuncAnimation(fig, animate, init_func=init, frames=len(temperature_data), blit=False)

Writer = animation.writers['ffmpeg']
writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)

ani.save(video_file_path, writer=writer)
plt.close()
