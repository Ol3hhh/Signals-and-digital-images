import numpy as np
from PIL import Image


def load_image_as_grayscale(path):
    """Завантаження зображення та перетворення його на градації сірого."""
    with Image.open(path) as img:
        gray_image = img.convert("L")
    return np.array(gray_image)


def save_image(image_array, path):
    """Збереження зображення у форматі PNG."""
    img = Image.fromarray(np.uint8(image_array))
    img.save(path)


import numpy as np
from PIL import Image


def bayesian_demosaicing_enhanced(image):
    """Демозаїкування із врахуванням градієнтів."""
    height, width = image.shape
    demosaiced_image = np.zeros((height, width, 3), dtype=np.float32)

    for i in range(height):
        for j in range(width):
            # Ініціалізація каналів
            red, green, blue = 0, 0, 0

            # Залежно від позиції у фільтрі Байєра
            if (i % 2 == 0) and (j % 2 == 0):  # Червоний піксель
                red = image[i, j]
                green = (image[i - 1, j] if i > 0 else 0 + image[i + 1, j] if i < height - 1 else 0 +
                                                                                                  image[
                                                                                                      i, j - 1] if j > 0 else 0 +
                                                                                                                              image[
                                                                                                                                  i, j + 1] if j < width - 1 else 0) / 4
                blue = interpolate_with_gradient(image, j, i, -1, -1)
            elif (i % 2 == 0) and (j % 2 == 1):  # Зелений піксель (червоний рядок)
                green = image[i, j]
                red = (image[i, j - 1] if j > 0 else 0 + image[i, j + 1] if j < width - 1 else 0) / 2
                blue = (image[i - 1, j] if i > 0 else 0 + image[i + 1, j] if i < height - 1 else 0) / 2
            elif (i % 2 == 1) and (j % 2 == 0):  # Зелений піксель (синій рядок)
                green = image[i, j]
                red = (image[i - 1, j] if i > 0 else 0 + image[i + 1, j] if i < height - 1 else 0) / 2
                blue = (image[i, j - 1] if j > 0 else 0 + image[i, j + 1] if j < width - 1 else 0) / 2
            elif (i % 2 == 1) and (j % 2 == 1):  # Синій піксель
                blue = image[i, j]
                green = (image[i - 1, j] if i > 0 else 0 + image[i + 1, j] if i < height - 1 else 0 +
                                                                                                  image[
                                                                                                      i, j - 1] if j > 0 else 0 +
                                                                                                                              image[
                                                                                                                                  i, j + 1] if j < width - 1 else 0) / 4
                red = interpolate_with_gradient(image, j, i, -1, -1)

            # Встановлення значень
            demosaiced_image[i, j] = [red, green, blue]

    # Нормалізація
    demosaiced_image = np.clip(demosaiced_image, 0, 255)
    return demosaiced_image.astype(np.uint8)


# Функція інтерполяції (згадана раніше)
def interpolate_with_gradient(image, x, y, dx, dy):
    h, w = image.shape
    val1 = image[y + dy, x] if 0 <= y + dy < h else 0
    val2 = image[y, x + dx] if 0 <= x + dx < w else 0
    weight1 = abs(image[y, x] - val1)
    weight2 = abs(image[y, x] - val2)
    if weight1 + weight2 == 0:
        return (val1 + val2) / 2
    return (weight1 * val2 + weight2 * val1) / (weight1 + weight2)


# Шлях до вхідного та вихідного файлів
input_path = "Fella.jpg"
output_path = "Fella1.jpg"

# Завантаження, обробка та збереження
gray_image = load_image_as_grayscale(input_path)
demosaiced_image = bayesian_demosaicing(gray_image)
save_image(demosaiced_image, output_path)

print(f"Демозаїкування завершено. Зображення збережено як {output_path}")
