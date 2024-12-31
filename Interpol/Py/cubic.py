import numpy as np
from PIL import Image, ImageChops
import math


def cubic_weight(t):
    """Compute the weight for cubic interpolation."""
    a = -0.5  # Typical choice for cubic interpolation
    t = abs(t)
    if t <= 1:
        return (a + 2) * t ** 3 - (a + 3) * t ** 2 + 1
    elif t < 2:
        return a * t ** 3 - 5 * a * t ** 2 + 8 * a * t - 4 * a
    else:
        return 0


def cubic_interpolate(x, y, input_pixels):
    x0 = int(x)
    y0 = int(y)

    # Collect surrounding 4x4 pixel values
    values = np.zeros((4, 4, 3), dtype=np.float64)
    for i in range(-1, 3):
        for j in range(-1, 3):
            xi = min(max(x0 + j, 0), input_pixels.shape[1] - 1)
            yi = min(max(y0 + i, 0), input_pixels.shape[0] - 1)
            values[i + 1, j + 1] = input_pixels[yi, xi]

    # Calculate weights for x and y directions
    weights_x = np.array([cubic_weight(x - (x0 + j)) for j in range(-1, 3)])
    weights_y = np.array([cubic_weight(y - (y0 + i)) for i in range(-1, 3)])

    # Interpolate over x for each y, then over y
    interpolated = np.zeros(3)
    for c in range(3):  # For each color channel
        col = np.dot(weights_x, values[:, :, c])
        interpolated[c] = np.dot(weights_y, col)

    return np.clip(interpolated, 0, 255).astype(np.uint8)


def rotate_image_cubic(image, angle):
    input_pixels = np.array(image)
    h, w = input_pixels.shape[:2]

    theta = math.radians(angle)
    cos_theta, sin_theta = math.cos(theta), math.sin(theta)

    center_x, center_y = w / 2, h / 2

    new_w = int(abs(h * sin_theta) + abs(w * cos_theta))
    new_h = int(abs(h * cos_theta) + abs(w * sin_theta))

    rotated_image = np.zeros((new_h, new_w, 3), dtype=np.uint8)
    new_center_x, new_center_y = new_w / 2, new_h / 2

    y_indices, x_indices = np.indices((new_h, new_w))
    rel_x = x_indices - new_center_x
    rel_y = y_indices - new_center_y

    orig_x = rel_x * cos_theta + rel_y * sin_theta + center_x
    orig_y = -rel_x * sin_theta + rel_y * cos_theta + center_y

    for i in range(new_h):
        for j in range(new_w):
            x, y = orig_x[i, j], orig_y[i, j]
            if 0 <= x < w and 0 <= y < h:
                rotated_image[i, j] = cubic_interpolate(x, y, input_pixels)

    return Image.fromarray(rotated_image)


def crop_black_borders(image):
    bg = Image.new(image.mode, image.size, (0, 0, 0))
    diff = ImageChops.difference(image, bg)
    bbox = diff.getbbox()
    if bbox:
        return image.crop(bbox)
    return image


if __name__ == "__main__":
    image_path = "Fella.jpg"
    image = Image.open(image_path)
    angle = 12

    for _ in range(1):
        image = rotate_image_cubic(image, angle)
        image = crop_black_borders(image)
        print(_)

    image.show()
    output_path = "../Source/32d/с.jpg"
    image.save(output_path)
