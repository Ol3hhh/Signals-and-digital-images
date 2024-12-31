import cupy as cp
from PIL import Image


def nearest_neighbor_rescale_2d(k, image):
    original_width, original_height = image.size
    n_width = int(k * original_width)
    n_height = int(k * original_height)
    original_array = cp.array(image)
    original_y_indices = (cp.arange(n_height) / k).astype(int)
    original_x_indices = (cp.arange(n_width) / k).astype(int)
    original_y_indices = cp.clip(original_y_indices, 0, original_height - 1)
    original_x_indices = cp.clip(original_x_indices, 0, original_width - 1)
    new_array = original_array[original_y_indices[:, None], original_x_indices]
    new_array_numpy = cp.asnumpy(new_array)
    new_image = Image.fromarray(new_array_numpy.astype('uint8'))
    return new_image
