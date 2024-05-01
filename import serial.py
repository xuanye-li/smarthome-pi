import serial
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Set up the serial connection
ser = serial.Serial('COM8', 115200, timeout=1)  # Adjust the COM port as needed

def update_data(frame, img, ser):
    line = ser.readline().decode('utf-8').strip()
    if line:
        try:
            data = np.array([float(val) for val in line.split(',') if val.strip()])
            data = data.reshape((24, 32))  # Reshape the data to match the MLX90640 24x32 array
            img.set_data(data)
            print("Frame updated")  # Optional: Print a confirmation that the frame was updated
        except ValueError:
            print("Data conversion error")  # Error handling for data conversion issues
    return img,

def init():
    img.set_data(np.zeros((24, 32)))  # Initialize the plot with zeros
    return img,

# Set up the plot
fig, ax = plt.subplots()
data = np.zeros((24, 32))
img = ax.imshow(data, cmap='inferno', interpolation='nearest', vmin=20, vmax=40)
plt.colorbar(img)  # Add a color bar to the side

ani = animation.FuncAnimation(fig, update_data, fargs=(img, ser), init_func=init, interval=250, blit=True)

plt.show()