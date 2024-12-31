from PIL import Image
import cupy as cp
import numpy as np


original_image_path = "../Source/Fella.jpg"
original_image = Image.open(original_image_path)
original_image = original_image.convert("RGB")


pixels = cp.array(original_image, dtype=cp.uint8)
pixelsR = pixels.copy()
pixelsG = pixels.copy()
pixelsB = pixels.copy()


print("Shape of the image:", pixels.shape)

def R(pixels):
    pixels[::2, 1::2, 1:] =0#Red
    pixels[1::2, ::2] = 0# Blue
    pixels[::2, ::2] = 0 #Green odd
    pixels[1::2, 1::2] = 0
    return pixels

def B(pixels):
    pixels[::2, 1::2] = 0  # Red
    pixels[1::2, ::2, :2] = 0  # Blue
    pixels[::2, ::2] = 0  # Green odd
    pixels[1::2, 1::2] = 0
    return pixels

def G(pixels):
    pixels[::2, 1::2] = 0  # Red
    pixels[1::2, ::2] = 0  # Blue
    pixels[::2, ::2, [0, 2]] = 0  # Green odd
    pixels[1::2, 1::2, [0, 2]] = 0
    return pixels

pixelsR = R(pixelsR)
pixelsG = G(pixelsG)
pixelsB = B(pixels)



pixelsR = cp.asnumpy(pixelsR)
pixelsG = cp.asnumpy(pixelsG)
pixelsB = cp.asnumpy(pixelsB)

pixels = pixelsR + pixelsG + pixelsB
modified_pixels = cp.asnumpy(pixels)
R_image = Image.fromarray(pixelsR.astype(np.uint8))
G_image = Image.fromarray(pixelsG.astype(np.uint8))
B_image = Image.fromarray(pixelsB.astype(np.uint8))

new_image = Image.fromarray(modified_pixels.astype(np.uint8))

R_image.show()
G_image.show()
B_image.show()
new_image.show()

# R_image.save("../Preprocessed/Fella_Bayes_R.png")
# G_image.save("../Preprocessed/Fella_Bayes_G.png")
# B_image.save("../Preprocessed/Fella_Bayes_B.png")
# new_image.save("../Preprocessed/Fella_Bayes.png")

