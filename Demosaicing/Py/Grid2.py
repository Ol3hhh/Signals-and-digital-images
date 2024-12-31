from PIL import Image

def create_grid(image_paths, crop_size=(20, 20)):
    # Open the images and crop them to the specified size
    cropped_images = [Image.open(path).crop((0, 0, crop_size[0], crop_size[1])) for path in image_paths]

    # Determine the size of the grid
    grid_width = 2 * crop_size[0]
    grid_height = 2 * crop_size[1]

    # Create a new image with the size of the grid
    grid_image = Image.new('RGB', (grid_width, grid_height))

    # Paste the cropped images into the grid
    grid_image.paste(cropped_images[0], (0, 0))
    grid_image.paste(cropped_images[1], (crop_size[0], 0))
    grid_image.paste(cropped_images[2], (0, crop_size[1]))
    grid_image.paste(cropped_images[3], (crop_size[0], crop_size[1]))

    return grid_image

# Define the paths to your images
image_paths = ['../Preprocessed/Xtrans_R_white.png', '../Preprocessed/Xtrans_G_white.png', '../Preprocessed/Xtrans_B_white.png', '../Preprocessed/Xtrans_white.png']

# Create the grid image
grid_image = create_grid(image_paths)

# Show or save the resulting grid image
grid_image.show()  # To display the image
grid_image.save('../Preprocessed/Xtrans_white_grid.png')  # To save the image
