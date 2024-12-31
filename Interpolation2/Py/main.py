import cupy as cp
import numpy as np
from PIL import Image
from scipy.special import mathieu_even_coef

from nn import *
from linear import *
from cubic import rescale_cubic_spline_2d


# Wczytaj obraz
original_image_path = "../Source/flower.jpg"
original_image = Image.open(original_image_path).convert("RGB")
pixels = cp.array(original_image, dtype=cp.uint8)

pixels_copy = pixels.copy()
print("Shape of the image:", pixels.shape)


lambdas = [1, 4, 16, 64, 256, 1024]

def generate_poisson_image(lm, image):
    #lm - lambda
    return cp.random.poisson(lam=lm * image).astype(cp.uint8)


def calculate_mse(original, scaled):
    diff = original - scaled
    mse = cp.mean(cp.square(diff))
    return mse


def calculate_mae(original, scaled):
    diff = cp.abs(original - scaled)
    mae = cp.mean(diff)
    return mae


def calculate_rsm(original, scaled):
    diff = cp.square(original - scaled)
    rms = cp.mean(cp.sqrt(diff))
    return rms

def calculate_rcm(original, scaled):
    diff = cp.power((original - scaled), 3)
    rcm = cp.mean(cp.cbrt(diff))
    return rcm

def calculate_geometric_mean(original, scaled):
    return cp.power(cp.prod(original - scaled), original.shape[0])

images = []
msel = 0
msec = 0

mael = 0
maec = 0

rsml = 0
rsmc = 0

rcml = 0
rcmc = 0

gm = 0

for i, lm in enumerate(lambdas):
    poisson_image = generate_poisson_image(lm, pixels_copy)
    poisson_image = poisson_image.get()
    poisson_image = np.clip(poisson_image, 0, 255)
    images.append(poisson_image)
    image = Image.fromarray(images[i].astype(np.uint8))
    image.save(f"../Preprocessed/source_{lm}.png")

    nn_image = nearest_neighbor_rescale_2d(100 / 1024, image)
    l_image = linear_rescale_2d(100 / 1024, image)
    c_image = rescale_cubic_spline_2d(100 / 1024, image)
    nn_image.save(f"../Preprocessed/nn_100_{lm}.png")
    l_image.save(f"../Preprocessed/l_100_{lm}.png")
    c_image.save(f"../Preprocessed/c_100_{lm}.png")

    l_image = linear_rescale_2d(1024 / 100, l_image)
    c_image = rescale_cubic_spline_2d(1024 / 100, c_image)
    l_image.save(f"../Preprocessed/l_1024_{lm}.png")
    c_image.save(f"../Preprocessed/c_1024_{lm}.png")



    original_array = cp.array(np.array(image))
    nn_array = cp.array(np.array(nn_image))
    l_array = cp.array(np.array(l_image))
    c_array = cp.array(np.array(c_image))

    msel += calculate_mse(original_array, l_array)
    msec += calculate_mse(original_array, c_array)

    mael += calculate_mae(original_array, l_array)
    maec += calculate_mae(original_array, c_array)

    rsml += calculate_rsm(original_array, l_array)
    rsmc += calculate_rsm(original_array, c_array)

    rcml += calculate_rcm(original_array, l_array)
    rcmc += calculate_rcm(original_array, c_array)
    print("linear MSE: " + str(calculate_mse(original_array, l_array)))
    print("cubic MSE: " + str(calculate_mse(original_array, c_array)))

    print("linear MAE: " + str(calculate_mae(original_array, l_array)))
    print("cubic MAE: " + str(calculate_mae(original_array, c_array)))

    print("linear RSM: " + str(calculate_rsm(original_array, l_array)))
    print("cubic RSM: " + str(calculate_rsm(original_array, c_array)))

    print("linear RSM: " + str(calculate_rcm(original_array, l_array)))
    print("cubic RSM: " + str(calculate_rcm(original_array, c_array)))


    print("__________")

print("linear MSE: " + str(int(msel)))
print("cubic MSE: " + str(int(msec)))
print("linear MAE: " + str(int(mael)))
print("cubic MAE: " + str(int(maec)))
print("linear RSM: " + str(int(rsml)))
print("cubic RSM: " + str(int(rsmc)))
print("linear RCM: " + str(int(rcml)))
print("cubic RCM: " + str(int(rcmc)))


