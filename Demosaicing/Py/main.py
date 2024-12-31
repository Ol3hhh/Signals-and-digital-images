from PIL import Image
import cupy as cp
import numpy as np


from Bayes import *
from Xtrans import *



# Load the original image

original_image_path = "../Source/Fella.jpg"  # Replace with the path to your image

original_image = Image.open(original_image_path)
original_image = original_image.convert("RGB")  # Ensure the image is in RGB format

# Convert the image to a CuPy array (RGB format)
pixels = cp.array(original_image, dtype=cp.uint8)

print("Shape of the image:", pixels.shape)

def bayes(pixels):
    pixels[2:-1:2, 3:-2:2, 1:] = ((pixels[2:-1:2, 1:-3:2, 1::] + pixels[2:-1:2, 5::2, 1::]) // 2 + (pixels[:-3:2, 3:-2:2, 1:] + pixels[3::2, 3:-2:2, 1:]) // 2)//2#Red
    pixels[3:-1:2, 2:-1:2, :2] = ((pixels[3:-1:2, :-3:2, :2] + pixels[3:-1:2, 2::2, :2]) // 2 + (pixels[1:-3:2, 2:-1:2, :2] + pixels[2:-2:2, 2:-1:2, :2]) // 2) //2# Blue
    pixels[2:-1:2, 2:-1:2, [0, 2]] = ((pixels[2:-1:2, :-3:2, [0, 2]] + pixels[2:-1:2, 2::2, [0, 2]]) // 2 + (pixels[:-3:2, 2:-1:2, [0, 2]] + pixels[3::2, 2:-1:2, [0, 2]]) // 2)//2 #Green odd
    pixels[3:-1:2, 3:-2:2, [0, 2]] = ((pixels[3:-1:2, 1:-3:2, [0, 2]] + pixels[3:-1:2, 5::2, [0, 2]]) // 2 + (pixels[1:-3:2, 3:-2:2, [0, 2]] + pixels[2:-2:2, 3:-2:2, [0, 2]]) // 2) //2 #Green even
    return pixels

def count(pixels, col_start, row_start, color):
    step = 6
    row_avg = (
        (pixels[col_start + step:col_start - step:step, row_start + 5:row_start - 2 * step:step, color] +
         pixels[col_start + step:col_start - step:step, row_start + 7:row_start - 2 * step + 2:step, color]) // 2
    )
    col_avg = (
        (pixels[col_start + 5:col_start - 2 * step:step, row_start + step:row_start - 2 * step:step, color] +
         pixels[col_start + 7:col_start - 2 * step + 2:step, row_start + step:row_start - 2 * step:step, color]) // 2
    )
    min_shape = (
        min(row_avg.shape[0], col_avg.shape[0]),
        min(row_avg.shape[1], col_avg.shape[1]),
    )
    row_avg = row_avg[:min_shape[0], :min_shape[1], :]
    col_avg = col_avg[:min_shape[0], :min_shape[1], :]
    target_slice = pixels[
        col_start + step:col_start + step + min_shape[0] * step:step,
        row_start + step:row_start + step + min_shape[1] * step:step,
        color,
    ]
    pixels[
        col_start + step:col_start + step + min_shape[0] * step:step,
        row_start + step:row_start + step + min_shape[1] * step:step,
        color,
    ] = (row_avg + col_avg) // 2
    return pixels





def xtrans(pixels):

    color = [1,2]

    for col_start in range(6):
        if col_start == 0:
            row_start = 2
            count(pixels, col_start, row_start, color)
            row_start = 4
        elif col_start == 1:
            row_start = 0

        elif col_start == 2:
            row_start = 3
        elif col_start == 3:
            row_start = 1
            count(pixels, col_start, row_start, color)
            row_start = 5
        elif col_start == 4:
            row_start = 3
        elif col_start == 5:
            row_start = 0

        count(pixels, col_start, row_start, color)



    color = [0,1]
    for col_start in range(6):
        if col_start == 0:
            row_start = 1
            count(pixels, col_start, row_start, color)
            row_start = 5
        elif col_start == 1:
            row_start = 3

        elif col_start == 2:
            row_start = 0
        elif col_start == 3:
            row_start = 2
            count(pixels, col_start, row_start, color)
            row_start = 4
        elif col_start == 4:
            row_start = 0
        elif col_start == 5:
            row_start = 3

        count(pixels, col_start, row_start, color)

    color = [0, 2]
    for col_start in range(6):
        count(pixels, col_start, col_start, color)
        if col_start == 0:
            row_start = 3
        elif col_start == 1:
            row_start = 2
            count(pixels, col_start, row_start, color)
            row_start = 4
            count(pixels, col_start, row_start, color)
            row_start = 5
        elif col_start == 2:
            row_start = 1
            count(pixels, col_start, row_start, color)
            row_start = 4
            count(pixels, col_start, row_start, color)
            row_start = 5
        elif col_start == 3:
            row_start = 0

        elif col_start == 4:
            row_start = 1
            count(pixels, col_start, row_start, color)
            row_start = 2
            count(pixels, col_start, row_start, color)
            row_start = 5
        elif col_start == 5:
            row_start = 1
            count(pixels, col_start, row_start, color)
            row_start = 2
            count(pixels, col_start, row_start, color)
            row_start = 4

        count(pixels, col_start, row_start, color)


    return pixels


xtrans_pixels = pixels.copy()
pixels_bayes = pixels.copy()
xtrans_pixels = cp.asnumpy(xtrans(xtrans_pixels))
pixels_bayes = cp.asnumpy(bayes(pixels_bayes))


bayes_image = Image.fromarray(pixels_bayes.astype(np.uint8))
bayes_image.show()
bayes_image.save("../Preprocessed/Fella_Bayes_interpolation.png")

xtrans_image = Image.fromarray(xtrans_pixels.astype(np.uint8))
xtrans_image.show()
xtrans_image.save("../Preprocessed/Fella_Xtrans_interpolation.png")