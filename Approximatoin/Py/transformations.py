import numpy as np
import cupy as cp
from scipy.fftpack import dct, idct
from scipy.linalg import hadamard
import pywt

def ans_matrix(pixels):
    return 2*cp.sqrt(pixels+3/8)

def ians_matrix(pixels):
    return 0.25*cp.square(pixels)-1/8+1/4*((3/2)**(1/2))*1/pixels-11/8*1/cp.square(pixels)+5/8*((3/2)**(1/2))*1/cp.power(pixels,3)

def dct_matrix(pixels):
    def apply_dct(image_channel):
        dct_channel = dct(dct(cp.asnumpy(image_channel).T, norm='ortho').T, norm='ortho')
        return cp.array(dct_channel)

    dct_pixels = cp.zeros_like(pixels, dtype=cp.float64)
    for i in range(3):
        dct_pixels[:, :, i] = apply_dct(pixels[:, :, i])

    return dct_pixels

def idct_matrix(pixels):
    def apply_idct(image_channel):
        idct_channel = idct(idct(cp.asnumpy(image_channel).T, norm='ortho').T, norm='ortho')
        return cp.array(idct_channel)

    idct_pixels = cp.zeros_like(pixels, dtype=cp.float64)
    for i in range(3):
        idct_pixels[:, :, i] = apply_idct(pixels[:, :, i])

    return idct_pixels

def wht_matrix(pixels):
    height, width = pixels.shape[:2]
    new_size = 2 ** int(cp.ceil(cp.log2(max(height, width))))
    padded_data = cp.zeros((new_size, new_size, 3), dtype=cp.float32)
    padded_data[:height, :width] = pixels
    H = cp.array(hadamard(new_size))
    H_norm = H / cp.sqrt(new_size)

    transformed = cp.zeros((new_size, new_size, 3), dtype=cp.float32)
    for channel in range(3):
        transformed[..., channel] = H_norm @ padded_data[..., channel] @ H_norm
    return transformed
def iwht_matrix(pixels):
    height, width = pixels.shape[:2]
    new_size = 2 ** int(cp.ceil(cp.log2(max(height, width))))
    transformed = cp.zeros_like(pixels, dtype=cp.float32)
    H = cp.array(hadamard(new_size))
    H_norm = H / cp.sqrt(new_size)
    for channel in range(3):
        transformed[..., channel] = H_norm @ pixels[..., channel] @ H_norm
    return transformed

def haar_matrix(pixels):
    coeffs = []

    for i in range(pixels.shape[2]):  # Loop through three channels: R, G, B
        LL, (LH, HL, HH) = pywt.dwt2(cp.asnumpy(pixels[:, :, i]), 'haar')
        coeffs.append((cp.array(LL), cp.array(LH), cp.array(HL), cp.array(HH)))

    return coeffs

def ihaar_matrix(coeffs):
    # Determine the dimensions of the reconstructed image
    height = coeffs[0][0].shape[0] * 2
    width = coeffs[0][0].shape[1] * 2
    channels = len(coeffs)

    reconstructed_pixels = cp.zeros((height, width, channels), dtype=cp.float64)

    for i in range(channels):  # Loop through three channels: R, G, B
        LL, LH, HL, HH = coeffs[i]
        reconstructed_channel = pywt.idwt2((cp.asnumpy(LL), (cp.asnumpy(LH), cp.asnumpy(HL), cp.asnumpy(HH))), 'haar')
        reconstructed_pixels[:, :, i] = cp.array(reconstructed_channel)

    return reconstructed_pixels

def bior53_matrix(pixels):
    coeffs = []

    for i in range(pixels.shape[2]):  # Loop through three channels: R, G, B
        LL, (LH, HL, HH) = pywt.dwt2(cp.asnumpy(pixels[:, :, i]), 'bior2.2')
        coeffs.append((cp.array(LL), cp.array(LH), cp.array(HL), cp.array(HH)))

    return coeffs

def ibior53_matrix(coeffs):
    # Determine the dimensions of the reconstructed image
    height = coeffs[0][0].shape[0] * 2
    width = coeffs[0][0].shape[1] * 2
    channels = len(coeffs)

    reconstructed_pixels = cp.zeros((height, width, channels), dtype=cp.float64)

    for i in range(channels):  # Loop through three channels: R, G, B
        LL, LH, HL, HH = coeffs[i]
        reconstructed_channel = pywt.idwt2((cp.asnumpy(LL), (cp.asnumpy(LH), cp.asnumpy(HL), cp.asnumpy(HH))), 'bior2.2')
        reconstructed_pixels[:reconstructed_channel.shape[0], :reconstructed_channel.shape[1], i] = cp.array(reconstructed_channel)

    return reconstructed_pixels

def bior97_matrix(pixels):
    coeffs = []

    for i in range(pixels.shape[2]):  # Loop through three channels: R, G, B
        LL, (LH, HL, HH) = pywt.dwt2(cp.asnumpy(pixels[:, :, i]), 'bior4.4')
        coeffs.append((cp.array(LL), cp.array(LH), cp.array(HL), cp.array(HH)))

    return coeffs

def ibior97_matrix(coeffs):
    # Determine the dimensions of the reconstructed image
    height = coeffs[0][0].shape[0] * 2
    width = coeffs[0][0].shape[1] * 2
    channels = len(coeffs)

    reconstructed_pixels = cp.zeros((height, width, channels), dtype=cp.float64)

    for i in range(channels):  # Loop through three channels: R, G, B
        LL, LH, HL, HH = coeffs[i]
        reconstructed_channel = pywt.idwt2((cp.asnumpy(LL), (cp.asnumpy(LH), cp.asnumpy(HL), cp.asnumpy(HH))), 'bior4.4')
        reconstructed_pixels[:reconstructed_channel.shape[0], :reconstructed_channel.shape[1], i] = cp.array(reconstructed_channel)

    return reconstructed_pixels

# Function to crop the processed image to match the original dimensions
def crop_to_original(original, processed):
    return processed[:original.shape[0], :original.shape[1], :]