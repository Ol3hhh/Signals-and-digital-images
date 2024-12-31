import cupy as cp
from PIL import Image, ImageChops
import math


def rotate_image_nearest_neighbor(image, angle, center_x=None, center_y=None):
    input_pixels = cp.array(image)
    h, w = input_pixels.shape[:2]

    if center_x is None:
        center_x, center_y = w / 2, h / 2


    theta = math.radians(angle)
    cos_theta, sin_theta = math.cos(theta), math.sin(theta)


    new_w = int(abs(h * sin_theta) + abs(w * cos_theta))
    new_h = int(abs(h * cos_theta) + abs(w * sin_theta))

    new_center_x, new_center_y = new_w / 2, new_h / 2

    rotated_image = cp.zeros((new_h, new_w, 3), dtype=cp.uint8)
    y_indices, x_indices = cp.indices((new_h, new_w))
    rel_x = x_indices - new_center_x
    rel_y = y_indices - new_center_y

    orig_x = (rel_x * cos_theta + rel_y * sin_theta + center_x).round().astype(int)
    orig_y = (-rel_x * sin_theta + rel_y * cos_theta + center_y).round().astype(int)

    valid_coords = (0 <= orig_x) & (orig_x < w) & (0 <= orig_y) & (orig_y < h)

    rotated_image[y_indices[valid_coords], x_indices[valid_coords]] = input_pixels[
        orig_y[valid_coords], orig_x[valid_coords]]

    return Image.fromarray(cp.asnumpy(rotated_image))


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



    # Поворот зображення
    for i in range(30):
        image = rotate_image_nearest_neighbor(image, angle)
        image = crop_black_borders(image)
        print(i)

    image.show()
    output_path = "../New/nnr.jpg"
    image.save(output_path)
