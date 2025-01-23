import cupy as cp
import numpy as np

# Функція для застосування порога
def threshold(dct_pixels, T):
    def apply_threshold(image, threshold):
        binary_image = cp.where(cp.abs(image) > threshold, image, 0)
        return binary_image

    # Застосування межі яскравості
    thresholded_pixels = apply_threshold(dct_pixels, T)

    return cp.transpose(thresholded_pixels)