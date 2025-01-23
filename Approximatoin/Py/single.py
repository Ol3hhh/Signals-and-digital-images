from PIL import Image
import cupy as cp
import numpy as np
from transformations import *
from threshold import threshold
from quantization import quantization

# Open the image file and convert to CuPy array
image = Image.open('../Source/el.png').convert('RGB')
pixels = cp.array(image)

def generate_poisson_image(lm, image):
    return cp.random.poisson(lam=lm * image).astype(cp.float32)

poissoned_pixels = generate_poisson_image(1, pixels)
poissoned_image = Image.fromarray(cp.asnumpy(poissoned_pixels).astype(np.uint8))
# poissoned_image.show()

# Apply transformations with CuPy arrays
ans_pixels = ans_matrix(poissoned_pixels)
# Ensure the result of each function is a CuPy array before passing it further
bior97_pixels = bior97_matrix(ans_pixels)
T = 19
Q = 9

quantized_coeffs = []
for LL, LH, HL, HH in bior97_pixels:
    LL_q = quantization(threshold(LL, T), 2 ** Q)
    LH_q = quantization(threshold(LH, T), 2 ** Q)
    HL_q = quantization(threshold(HL, T), 2 ** Q)
    HH_q = quantization(threshold(HH, T), 2 ** Q)
    quantized_coeffs.append((LL_q, LH_q, HL_q, HH_q))

ibior97_pixels = ibior97_matrix(quantized_coeffs)
ibior97_pixels = crop_to_original(pixels, ibior97_pixels)
ibior97_pixels = ians_matrix(ibior97_pixels)
ibior97_pixels = cp.clip(ibior97_pixels, 0, 255).astype(cp.uint8)  # Ensure the data type is uint8

new_image = Image.fromarray(cp.asnumpy(ibior97_pixels))
new_image.show()
