from PIL import Image
import cupy as cp

def linear_rescale_1d(k, image):
    original_width, original_height = image.size
    n_width = int(k * original_width)
    n_height = original_height

    img_array = cp.array(image, dtype=cp.float32)
    new_x_coords = cp.arange(n_width, dtype=cp.float32) / k
    x1 = cp.floor(new_x_coords).astype(cp.int32)
    x2 = cp.clip(x1 + 1, 0, original_width - 1)
    a = new_x_coords - x1

    new_image_array = cp.zeros((n_height, n_width, 3), dtype=cp.float32)
    pixel1 = img_array[:, x1, :]
    pixel2 = img_array[:, x2, :]
    new_image_array = (1 - a)[:, None] * pixel1 + a[:, None] * pixel2

    new_image = Image.fromarray(cp.asnumpy(new_image_array).astype(cp.uint8), mode="RGB")
    return new_image

if __name__ == "__main__":
    image = Image.open("Fella.jpg")
    width, height = image.size
    print(f"Original size: {(width, height)}")
    k = 2
    new_image = linear_rescale_1d(k, image)
    new_image.show()
    new_image.save("../New/l1d2.jpg")
