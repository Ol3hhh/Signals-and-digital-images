from PIL import Image
import numpy as np

def cubic_spline_weight(t):
    a = -0.5
    abs_t = np.abs(t)
    abs_t2 = abs_t ** 2
    abs_t3 = abs_t ** 3
    if abs_t <= 1:
        return (a + 2) * abs_t3 - (a + 3) * abs_t2 + 1
    elif abs_t < 2:
        return a * abs_t3 - 5 * a * abs_t2 + 8 * a * abs_t - 4 * a
    else:
        return 0.0

def cubic_interpolate(p0, p1, p2, p3, t):
    return (
        p0 * cubic_spline_weight(t + 1) +
        p1 * cubic_spline_weight(t) +
        p2 * cubic_spline_weight(t - 1) +
        p3 * cubic_spline_weight(t - 2)
    )

def rescale_cubic_spline_1d(k, image):
    original_width, original_height = image.size
    n_width = int(k * original_width)
    n_height = original_height
    new_pixels = []
    for y in range(original_height):
        row_pixels = [image.getpixel((x, y)) for x in range(original_width)]
        new_row = []
        for new_x in range(n_width):
            original_x = new_x / k
            x1 = int(original_x)
            t = original_x - x1
            p0 = np.array(row_pixels[max(x1 - 1, 0)], dtype=float)
            p1 = np.array(row_pixels[x1], dtype=float)
            p2 = np.array(row_pixels[min(x1 + 1, original_width - 1)], dtype=float)
            p3 = np.array(row_pixels[min(x1 + 2, original_width - 1)], dtype=float)
            new_pixel = cubic_interpolate(p0, p1, p2, p3, t)
            new_row.append(tuple(np.clip(new_pixel, 0, 255).astype(np.uint8)))
        new_pixels.extend(new_row)
    new_image = Image.new("RGB", (n_width, n_height))
    new_image.putdata(new_pixels)
    return new_image

if __name__ == "__main__":
    image = Image.open("Fella.jpg")
    width, height = image.size
    print(f"Original size: {(width, height)}")
    k = 0.5
    new_image = rescale_cubic_spline_1d(k, image)
    new_image.show()
    new_image.save("../New/c1d05.jpg")
