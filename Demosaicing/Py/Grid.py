from PIL import Image

# Open the images
image1 = Image.open('../Preprocessed/Xtrans_Fella_R.png')
image2 = Image.open('../Preprocessed/Xtrans_Fella_G.png')
image3 = Image.open('../Preprocessed/Xtrans_Fella_B.png')
image4 = Image.open('../Preprocessed/Xtrans_Fella.png')

# Determine the size of the grid (assuming all images are the same size)
width, height = image1.size
grid_width = 2 * width
grid_height = 2 * height

# Create a new image with the size of the grid
grid_image = Image.new('RGB', (grid_width, grid_height))

# Paste the images into the grid
grid_image.paste(image1, (0, 0))
grid_image.paste(image2, (width, 0))
grid_image.paste(image3, (0, height))
grid_image.paste(image4, (width, height))

# Show or save the resulting grid image
grid_image.show()  # To display the image
grid_image.save('../Preprocessed/Fella_Xtrans_Grid.png')  # To save the image
