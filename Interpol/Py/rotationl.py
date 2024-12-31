import cupy as cp
from PIL import Image
import math

def rotate_image_linear_interpolation_cupy(image, angle):
    # Конвертуємо зображення в масив CuPy
    input_pixels = cp.array(image)
    h, w, c = input_pixels.shape

    # Переводимо кут у радіани
    theta = math.radians(angle)
    cos_theta, sin_theta = math.cos(theta), math.sin(theta)

    # Обчислюємо центр вихідного зображення
    center_x, center_y = w / 2, h / 2

    # Ініціалізуємо масив нового зображення з розмірами оригінального
    rotated_image = cp.zeros_like(input_pixels, dtype=cp.uint8)
    new_center_x, new_center_y = w / 2, h / 2

    # Створюємо сітку координат для нового зображення
    y_indices, x_indices = cp.indices((h, w))
    rel_x = x_indices - new_center_x
    rel_y = y_indices - new_center_y

    # Використовуємо зворотну матрицю обертання для обчислення координат у вихідному зображенні
    orig_x = rel_x * cos_theta + rel_y * sin_theta + center_x
    orig_y = -rel_x * sin_theta + rel_y * cos_theta + center_y

    # Маска для координат, що знаходяться в межах вихідного зображення
    valid_coords = (0 <= orig_x) & (orig_x < w) & (0 <= orig_y) & (orig_y < h)

    # Лінійна інтерполяція для всіх пікселів одночасно
    x0 = cp.floor(orig_x).astype(cp.int32)
    y0 = cp.floor(orig_y).astype(cp.int32)
    x1 = cp.clip(x0 + 1, 0, w - 1)
    y1 = cp.clip(y0 + 1, 0, h - 1)

    dx = orig_x - x0
    dy = orig_y - y0

    for channel in range(c):
        rotated_image[..., channel] = input_pixels[..., channel]

        p00 = input_pixels[y0, x0, channel]
        p01 = input_pixels[y0, x1, channel]
        p10 = input_pixels[y1, x0, channel]
        p11 = input_pixels[y1, x1, channel]

        p0 = (1 - dx) * p00 + dx * p01
        p1 = (1 - dy) * p10 + dy * p11

        rotated_image[valid_coords, channel] = (1 - dy) * p0[valid_coords] + dy * p1[valid_coords]

    return Image.fromarray(cp.asnumpy(rotated_image))

# Основний код
if __name__ == "__main__":
    # Відкриття зображення
    image_path = "Fella.jpg"  # Шлях до вашого зображення
    image = Image.open(image_path)
    angle = 12  # Кут обертання

    for i in range(30):
        image = rotate_image_linear_interpolation_cupy(image, angle)
        print(i)

    image.show()
    output_path = "../New/linear_interpolation_cupy_with_black_borders.jpg"
    image.save(output_path)
