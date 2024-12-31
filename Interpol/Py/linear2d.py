import cupy as cp
from PIL import Image

def linear_rescale_2d(k, image):
    original_width, original_height = image.size
    # Calculate new dimensions
    n_width = int(k * original_width)
    n_height = int(k * original_height)

    # Convert image to NumPy array
    image_array = cp.array(image)

    # Generate new x and y indices
    new_x_indices = cp.arange(n_width) / k
    new_y_indices = cp.arange(n_height) / k

    # Calculate x interpolation weights and indices
    x1 = cp.floor(new_x_indices).astype(int)
    x2 = cp.minimum(x1 + 1, original_width - 1)
    a = new_x_indices - x1

    # Interpolate along the x-axis for each row
    x_interpolated = (1 - a)[cp.newaxis, :, cp.newaxis] * image_array[:, x1] + a[cp.newaxis, :, cp.newaxis] * image_array[:, x2]

    # Calculate y interpolation weights and indices
    y1 = cp.floor(new_y_indices).astype(int)
    y2 = cp.minimum(y1 + 1, original_height - 1)
    b = new_y_indices - y1

    # Interpolate along the y-axis for each column
    new_image_array = (1 - b)[:, cp.newaxis, cp.newaxis] * x_interpolated[y1] + b[:, cp.newaxis, cp.newaxis] * x_interpolated[y2]

    # Clip the values to be in the valid range [0, 255] and convert to uint8
    new_image_array = cp.clip(new_image_array, 0, 255).astype(cp.uint8)

    # Create a new image with the resized pixels
    new_image = Image.fromarray(new_image_array, "RGB")

    return new_image
