from PIL import Image

def nearest_neighbor_rescale_1d(k, image):
    original_width, height = image.size
    n_width = int(k * original_width)
    new_pixels = []

    for y in range(height):
        for new_x in range(n_width):
            original_x = int(new_x / k)
            original_x = min(original_x, original_width - 1)
            pixel = image.getpixel((original_x, y))
            new_pixels.append(pixel)

    new_image = Image.new("RGB", (n_width, height))
    new_image.putdata(new_pixels)

    return new_image


if __name__ == "__main__":
    image = Image.open("Fella.jpg")
    width, height = image.size
    print(f"Original size: {(width, height)}")
    k = 0.5
    new_image = nearest_neighbor_rescale_1d(k, image)
    new_image.show()
