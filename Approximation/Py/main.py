import cupy as cp
import numpy as np
from PIL import Image

original_image_path = "../Source/Fella.jpg"
original_image = Image.open(original_image_path).convert("RGB")
pixels = cp.array(original_image, dtype=cp.uint8)


def generate_poisson_image(lm, image):
    #lm - lambda
    return cp.random.poisson(lam=lm * image).astype(cp.uint8)

poissoned_pixels = cp.asnumpy(generate_poisson_image(1, pixels))


poissoned_pixels = poissoned_pixels.transpose((1, 0, 2))
for i in range()

poissoned_image = Image.fromarray(poissoned_pixels.astype(np.uint8))
poissoned_image.show()