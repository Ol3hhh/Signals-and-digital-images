from PIL import Image
import numpy as np

original_image_path = "demosaic_before.jpg"  # Replace with the path to your image
original_image = Image.open(original_image_path)

# Get the size of the original image
width, height = original_image.size

# Initialize a list to store new image pixels
new_pixels = []

# Iterate through each pixel and apply the demosaicing logic
for y in range(height):
    for x in range(width):

        r, g, b = original_image.getpixel((x, y))

        if x < width - 2:
           a1 = 2
           a2 = 1
        else:
            a1 = -2
            a2 = -1

        if y < height - 1:
            b = 1
        else:
            b = -1



        if x % 2 != 0 and y % 2 != 0:
            print(b)
            g = original_image.getpixel((x+a1, y))[1]
            b = original_image.getpixel((x+a2, y+b))[2]
        elif x % 2 == 0 and y % 2 == 0:
            r = original_image.getpixel((x+a2, y+b))[2]
            g = original_image.getpixel((x+a1, y))[1]
        else:
            r = original_image.getpixel((x+a2, y+b))[2]
            b = original_image.getpixel((x+a2, y+b))[2]

        # Append the modified pixel to the new image
        new_pixels.append((r, g, b))

# Create a new image from the new_image list
new_image = Image.new("RGB", (width, height))
new_image.putdata(new_pixels)

# Save or show the new image
new_image.show()  # Show the modified image
