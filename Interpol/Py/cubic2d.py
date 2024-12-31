import numpy as cp
from PIL import Image

def cubic_spline_weight(t):
    """Calculates cubic weights for interpolation based on Catmull-Rom spline."""
    a = -0.5  # Parameter for Catmull-Rom splines

    abs_t = cp.abs(t)
    abs_t2 = abs_t ** 2
    abs_t3 = abs_t ** 3

    weight = cp.zeros_like(t)
    condition1 = abs_t <= 1
    condition2 = (abs_t > 1) & (abs_t < 2)

    weight[condition1] = (a + 2) * abs_t3[condition1] - (a + 3) * abs_t2[condition1] + 1
    weight[condition2] = a * abs_t3[condition2] - 5 * a * abs_t2[condition2] + 8 * a * abs_t[condition2] - 4 * a
    return weight

def cubic_interpolate(p0, p1, p2, p3, t):
    """Interpolate pixel values with cubic spline using neighboring pixels."""
    return (
        p0 * cubic_spline_weight(t + 1) +
        p1 * cubic_spline_weight(t) +
        p2 * cubic_spline_weight(t - 1) +
        p3 * cubic_spline_weight(t - 2)
    )

def rescale_cubic_spline_2d(k, image):
    original_width, original_height = image.size
    n_width = int(k * original_width)
    n_height = int(k * original_height)

    # Convert image to NumPy array
    image_array = cp.array(image)

    # Temporary storage for x-axis interpolated pixels
    x_interpolated = cp.zeros((original_height, n_width, 3), dtype=cp.float32)

    # Interpolate along the x-axis
    for y in range(original_height):
        original_x = cp.arange(n_width) / k
        x1 = cp.floor(original_x).astype(int)
        t = original_x - x1

        p0 = image_array[y, cp.clip(x1 - 1, 0, original_width - 1)]
        p1 = image_array[y, cp.clip(x1, 0, original_width - 1)]
        p2 = image_array[y, cp.clip(x1 + 1, 0, original_width - 1)]
        p3 = image_array[y, cp.clip(x1 + 2, 0, original_width - 1)]

        x_interpolated[y] = cubic_interpolate(p0, p1, p2, p3, t[:, None])

    # Final storage for y-axis interpolated pixels
    new_pixels = cp.zeros((n_height, n_width, 3), dtype=cp.uint8)

    # Interpolate along the y-axis
    original_y = cp.arange(n_height) / k
    y1 = cp.floor(original_y).astype(int)
    t = original_y - y1

    for new_x in range(n_width):
        p0 = x_interpolated[cp.clip(y1 - 1, 0, original_height - 1), new_x]
        p1 = x_interpolated[cp.clip(y1, 0, original_height - 1), new_x]
        p2 = x_interpolated[cp.clip(y1 + 1, 0, original_height - 1), new_x]
        p3 = x_interpolated[cp.clip(y1 + 2, 0, original_height - 1), new_x]

        new_pixels[:, new_x] = cp.clip(cubic_interpolate(p0, p1, p2, p3, t[:, None]), 0, 255)

    # Create a new image with the resized pixels
    new_image = Image.fromarray(new_pixels, "RGB")
    return new_image
