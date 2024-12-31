from PIL import Image
import cupy as cp
import numpy as np

# Load the original image
original_image_path = "../Source/Fella.jpg"  # Replace with the path to your image
original_image = Image.open(original_image_path)
original_image = original_image.convert("RGB")  # Ensure the image is in RGB format

# Convert the image to a CuPy array (RGB format)
pixels = cp.array(original_image, dtype=cp.uint8)
pixelsR = pixels.copy()
pixelsG = pixels.copy()
pixelsB = pixels.copy()


print("Shape of the image:", pixels.shape)

def R(pixels):
    #Reddd

    #first row
    pixels[::6, 2::6, 1:] =0
    pixels[::6, 4::6, 1:] = 0
    #Second row
    pixels[1::6, ::6, 1:] = 0
    #Third row
    pixels[2::6, 3::6, 1:] = 0
    #Fourth row
    pixels[3::6, 1::6, 1:] = 0
    pixels[3::6, 5::6, 1:] = 0
    #Fifth row
    pixels[4::6, 3::6, 1:] = 0
    #SIxth row
    pixels[5::6, ::6, 1:] = 0


    #Blueeeee
    #First row
    pixels[::6, 1::6] = 0
    pixels[::6, 5::6] = 0
    #Second row
    pixels[1::6, 3::6] = 0
    #Third row
    pixels[2::6, ::6] = 0
    #Fourth row
    pixels[3::6, 2::6] = 0
    pixels[3::6, 4::6] = 0
    #Fifth row
    pixels[4::6, ::6] = 0
    #Sixth row
    pixels[5::6, 3::6] = 0


    #Greennnn
    # First row
    pixels[::3, ::3] = 0
    # Second row
    pixels[1::3, 1::3] = 0
    pixels[1::3, 2::3] = 0
    # # Third row
    pixels[2::3, 1::3] = 0
    pixels[2::3, 2::3] = 0


    return pixels


def G(pixels):
    # Reddd

    # first row
    pixels[::6, 2::6] = 0
    pixels[::6, 4::6] = 0
    # Second row
    pixels[1::6, ::6] = 0
    # Third row
    pixels[2::6, 3::6] = 0
    # Fourth row
    pixels[3::6, 1::6] = 0
    pixels[3::6, 5::6] = 0
    # Fifth row
    pixels[4::6, 3::6] = 0
    # SIxth row
    pixels[5::6, ::6] = 0

    # Blueeeee
    # First row
    pixels[::6, 1::6] = 0
    pixels[::6, 5::6] = 0
    # Second row
    pixels[1::6, 3::6] = 0
    # Third row
    pixels[2::6, ::6] = 0
    # Fourth row
    pixels[3::6, 2::6] = 0
    pixels[3::6, 4::6] = 0
    # Fifth row
    pixels[4::6, ::6] = 0
    # Sixth row
    pixels[5::6, 3::6] = 0

    # Greennnn
    # First row
    pixels[::3, ::3, [0, 2]] = 0
    # Second row
    pixels[1::3, 1::3, [0, 2]] = 0
    pixels[1::3, 2::3, [0, 2]] = 0
    # # Third row
    pixels[2::3, 1::3, [0, 2]] = 0
    pixels[2::3, 2::3, [0, 2]] = 0
    return pixels




def B(pixels):
    # Reddd

    # first row
    pixels[::6, 2::6] = 0
    pixels[::6, 4::6] = 0
    # Second row
    pixels[1::6, ::6] = 0
    # Third row
    pixels[2::6, 3::6] = 0
    # Fourth row
    pixels[3::6, 1::6] = 0
    pixels[3::6, 5::6] = 0
    # Fifth row
    pixels[4::6, 3::6] = 0
    # SIxth row
    pixels[5::6, ::6] = 0

    # Blueeeee
    # First row
    pixels[::6, 1::6, :2] = 0
    pixels[::6, 5::6, :2] = 0
    # Second row
    pixels[1::6, 3::6, :2] = 0
    # Third row
    pixels[2::6, ::6, :2] = 0
    # Fourth row
    pixels[3::6, 2::6, :2] = 0
    pixels[3::6, 4::6, :2] = 0
    # Fifth row
    pixels[4::6, ::6, :2] = 0
    # Sixth row
    pixels[5::6, 3::6, :2] = 0

    # Greennnn
    # First row
    pixels[::3, ::3] = 0
    # Second row
    pixels[1::3, 1::3] = 0
    pixels[1::3, 2::3] = 0
    # # Third row
    pixels[2::3, 1::3] = 0
    pixels[2::3, 2::3] = 0
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

R_image.save("../Preprocessed/Xtrans_Fella_R.png")
G_image.save("../Preprocessed/Xtrans_Fella_G.png")
B_image.save("../Preprocessed/Xtrans_Fella_B.png")
new_image.save("../Preprocessed/Xtrans_Fella.png")