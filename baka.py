import time
import board
import busio
import numpy as np
import adafruit_mlx90640
import matplotlib.pyplot as plt
import pygame
from pygame.locals import QUIT

# Initialize I2C and MLX90640 sensor
i2c = busio.I2C(board.SCL, board.SDA, frequency=800000)
mlx = adafruit_mlx90640.MLX90640(i2c)
mlx.refresh_rate = adafruit_mlx90640.RefreshRate.REFRESH_16_HZ

# Initialize Pygame display
pygame.init()
window_size = (640, 480)  # Adjust to your screen resolution
screen = pygame.display.set_mode(window_size)
pygame.display.set_caption('MLX90640 Temperature Heatmap')

# Function to scale heatmap image to window size
def scale_to_screen(img):
    return pygame.transform.scale(img, window_size)

# Main loop
frame = [0] * 768
try:
    while True:
        stamp = time.monotonic()
        try:
            mlx.getFrame(frame)
        except ValueError:
            continue

        # Convert frame data to 2D array
        data = np.reshape(frame, (24, 32))

        # Generate heatmap
        plt.figure(figsize=(8, 6))
        plt.imshow(data, cmap='inferno', interpolation='none')
        plt.colorbar()
        plt.savefig('/tmp/heatmap.png')
        plt.close()

        # Load heatmap image into pygame
        heatmap = pygame.image.load('/tmp/heatmap.png')
        heatmap = scale_to_screen(heatmap)
        screen.blit(heatmap, (0, 0))
        pygame.display.update()

        # Event handling to quit
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()

        print("Read 2 frames in %0.2f s" % (time.monotonic() - stamp))
except KeyboardInterrupt:
    pygame.quit()
