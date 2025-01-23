import matplotlib.pyplot as plt
import cupy as cp
import numpy as np

def before(poissoned_pixels):
    # Red
    plt.figure(figsize=(10, 5))
    plt.title('Pixel Values (Red Channel)')
    plt.hist(cp.asnumpy(poissoned_pixels[:, :, 0]).flatten(), bins=256, range=(0, 256), color='r', alpha=0.7)
    plt.xlabel('Pixel Index')
    plt.ylabel('Pixel Value')
    plt.show()

    # Green
    plt.figure(figsize=(10, 5))
    plt.title('Pixel Values (Green Channel)')
    plt.hist(cp.asnumpy(poissoned_pixels[:, :, 1]).flatten(), bins=256, range=(0, 256), color='g', alpha=0.7)
    plt.xlabel('Pixel Index')
    plt.ylabel('Pixel Value')
    plt.show()

    # Blue
    plt.figure(figsize=(10, 5))
    plt.title('Pixel Values (Blue Channel)')
    plt.hist(cp.asnumpy(poissoned_pixels[:, :, 2]).flatten(), bins=256, range=(0, 256), color='b', alpha=0.7)
    plt.xlabel('Pixel Index')
    plt.ylabel('Pixel Value')
    plt.show()

def threshold(dct_pixels, quantized_pixels):
    # Побудова графіків значень пікселів до і після
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.hist(cp.asnumpy(dct_pixels).flatten(), bins=256, range=(0, 256), color='blue', alpha=0.7)
    plt.title('Before')
    plt.xlabel('Brightness')
    plt.ylabel('Amount of pixels')

    plt.subplot(1, 2, 2)
    plt.hist(cp.asnumpy(quantized_pixels).flatten(), bins=256, range=(0, 256), color='green', alpha=0.7)
    plt.title('After')
    plt.xlabel('Brightness')
    plt.ylabel('Amount of pixels')

    plt.tight_layout()
    plt.show()