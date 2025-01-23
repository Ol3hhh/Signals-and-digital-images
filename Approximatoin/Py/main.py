import numpy as np
import cupy as cp
from PIL import Image

from transformations import *
from threshold import threshold
from quantization import quantization


def crop_to_original(original, processed):
    return processed[:original.shape[0], :original.shape[1], :]

image = Image.open('../Source/el.png')
pixels = cp.array(image)

def generate_poisson_image(lm, image):
    return cp.random.poisson(lam=lm * image).astype(cp.float32)

poissoned_pixels = generate_poisson_image(1, pixels)
poissoned_image = Image.fromarray(cp.asnumpy(poissoned_pixels).astype(np.uint8))
poissoned_image.show()
poissoned_image.save('../Preprocessed/el_poisson.png')
ans_pixels = ans_matrix(poissoned_pixels)

dct_pixels = dct_matrix(ans_pixels)
wht_pixels = wht_matrix(ans_pixels)
haar_pixels = haar_matrix(ans_pixels)
bior53_pixels = bior53_matrix(ans_pixels)
bior97_pixels = bior97_matrix(ans_pixels)


best_dct_coeffs = (0, 0)
best_wht_coeffs = (0, 0)
best_haar_coeffs = (0, 0)
best_bior53_coeffs = (0, 0)
best_bior97_coeffs = (0, 0)

min_dct_deviation = float('inf')
min_wht_deviation = float('inf')
min_haar_deviation = float('inf')
min_bior53_deviation = float('inf')
min_bior97_deviation = float('inf')

T = 1
Q = 1
qh = 0
for i in range(20):
    for j in range(10):
        # Process for DCT
        quantized_pixels = quantization(threshold(dct_pixels, T), Q)
        idct_pixels = idct_matrix(quantized_pixels)
        idct_pixels = crop_to_original(pixels, idct_pixels)
        idct_pixels = ians_matrix(idct_pixels)
        idct_pixels = cp.clip(idct_pixels, 0, 255).astype(cp.uint8)  # Ensure the data type is uint8
        deviation_dct = cp.mean(cp.abs(pixels - idct_pixels))
        if deviation_dct < min_dct_deviation:
            min_dct_deviation = deviation_dct
            best_dct_coeffs = (T, Q)
            best_dct_pixels = idct_pixels
        print(1)

        # Process for WHT
        quantized_pixels = quantization(threshold(wht_pixels, T), Q)
        iwht_pixels = iwht_matrix(quantized_pixels)
        iwht_pixels = crop_to_original(pixels, iwht_pixels)
        iwht_pixels = ians_matrix(iwht_pixels)
        iwht_pixels = cp.clip(iwht_pixels, 0, 255).astype(cp.uint8)  # Ensure the data type is uint8
        iwht_pixels = cp.nan_to_num(iwht_pixels)

        deviation_wht = cp.mean(cp.abs(pixels - iwht_pixels))
        if deviation_wht < min_wht_deviation:
            min_wht_deviation = deviation_wht
            best_wht_coeffs = (T, Q)
            best_wht_pixels = iwht_pixels
        print(2)

        # Process for Haar
        quantized_coeffs = []
        for LL, LH, HL, HH in haar_pixels:
            LL_q = quantization(threshold(LL, T), 2 ** qh)
            LH_q = quantization(threshold(LH, T), 2 ** qh)
            HL_q = quantization(threshold(HL, T), 2 ** qh)
            HH_q = quantization(threshold(HH, T), 2 ** qh)
            quantized_coeffs.append((LL_q, LH_q, HL_q, HH_q))

        ihaar_pixels = ihaar_matrix(quantized_coeffs)
        ihaar_pixels = crop_to_original(pixels, ihaar_pixels)
        ihaar_pixels = ians_matrix(ihaar_pixels)
        ihaar_pixels = cp.clip(ihaar_pixels, 0, 255).astype(cp.uint8)  # Ensure the data type is uint8
        deviation_haar = cp.mean(cp.abs(pixels - ihaar_pixels))
        if deviation_haar < min_haar_deviation:
            min_haar_deviation = deviation_haar
            best_haar_coeffs = (T, 2 ** qh)
            best_haar_pixels = ihaar_pixels
        print(3)

        # Process for Bior53
        quantized_coeffs = []
        for LL, LH, HL, HH in bior53_pixels:
            LL_q = quantization(threshold(LL, T), 2 ** qh)
            LH_q = quantization(threshold(LH, T), 2 ** qh)
            HL_q = quantization(threshold(HL, T), 2 ** qh)
            HH_q = quantization(threshold(HH, T), 2 ** qh)
            quantized_coeffs.append((LL_q, LH_q, HL_q, HH_q))

        ibior53_pixels = ibior53_matrix(quantized_coeffs)
        ibior53_pixels = crop_to_original(pixels, ibior53_pixels)
        ibior53_pixels = ians_matrix(ibior53_pixels)
        ibior53_pixels = cp.clip(ibior53_pixels, 0, 255).astype(cp.uint8)  # Ensure the data type is uint8
        deviation_bior53 = cp.mean(cp.abs(pixels - ibior53_pixels))
        if deviation_bior53 < min_bior53_deviation:
            min_bior53_deviation = deviation_bior53
            best_bior53_coeffs = (T, 2 ** qh)
            best_bior53_pixels = ibior53_pixels
        print(4)

        # Process for Bior97
        quantized_coeffs = []
        for LL, LH, HL, HH in bior97_pixels:
            LL_q = quantization(threshold(LL, 37), 2 ** 2)
            LH_q = quantization(threshold(LH, 37), 2 ** 2)
            HL_q = quantization(threshold(HL, 37), 2 ** 2)
            HH_q = quantization(threshold(HH, 37), 2 ** 2)
            quantized_coeffs.append((LL_q, LH_q, HL_q, HH_q))

        ibior97_pixels = ibior97_matrix(quantized_coeffs)
        ibior97_pixels = crop_to_original(pixels, ibior97_pixels)
        ibior97_pixels = ians_matrix(ibior97_pixels)
        ibior97_pixels = cp.clip(ibior97_pixels, 0, 255).astype(cp.uint8)  # Ensure the data type is uint8
        deviation_bior97 = cp.mean(cp.abs(pixels - ibior97_pixels))
        if deviation_bior97 < min_bior97_deviation:
            min_bior97_deviation = deviation_bior97
            best_bior97_coeffs = (37, 2 ** 2)
            best_bior97_pixels = ibior97_pixels
        print(5)
        T += 2
    T = 1
    Q += 2
    print(Q)
    qh += 1

print(f"Best DCT coefficients: {best_dct_coeffs} with deviation {min_dct_deviation}")
print(f"Best WHT coefficients: {best_wht_coeffs} with deviation {min_wht_deviation}")
print(f"Best Haar coefficients: {best_haar_coeffs} with deviation {min_haar_deviation}")
print(f"Best Bior53 coefficients: {best_bior53_coeffs} with deviation {min_bior53_deviation}")
print(f"Best Bior97 coefficients: {best_bior97_coeffs} with deviation {min_bior97_deviation}")

# Display images with the best coefficients
best_dct_image = Image.fromarray(cp.asnumpy(best_dct_pixels).astype(np.uint8))
best_dct_image.save("../Preprocessed/Best/el_Best_DCT_Image.png")

best_wht_image = Image.fromarray(cp.asnumpy(best_wht_pixels).astype(np.uint8))
best_wht_image.save("../Preprocessed/Best/el_Best_WHT_Image.png")

best_haar_image = Image.fromarray(cp.asnumpy(best_haar_pixels).astype(np.uint8))
best_haar_image.save("../Preprocessed/Best/el_Best_Haar_Image.png")

best_bior53_image = Image.fromarray(cp.asnumpy(best_bior53_pixels).astype(np.uint8))
best_bior53_image.save("../Preprocessed/Best/el_Best_Bior53_Image.png")

best_bior97_image = Image.fromarray(cp.asnumpy(best_bior97_pixels).astype(np.uint8))
best_bior97_image.save("../Preprocessed/Best/el_Best_Bior97_Image.png")


# Function to count nonzero coefficients
def count_nonzero_coefficients(matrix):
    return cp.count_nonzero(matrix)

# Count nonzero coefficients for the best DCT image
dct_nonzero = count_nonzero_coefficients(quantization(threshold(dct_pixels, best_dct_coeffs[0]), best_dct_coeffs[1]))

# Count nonzero coefficients for the best WHT image
wht_nonzero = count_nonzero_coefficients(quantization(threshold(wht_pixels, best_wht_coeffs[0]), best_wht_coeffs[1]))

# Count nonzero coefficients for the best Haar image
haar_nonzero = sum(
    count_nonzero_coefficients(LL) + count_nonzero_coefficients(LH) +
    count_nonzero_coefficients(HL) + count_nonzero_coefficients(HH)
    for LL, LH, HL, HH in haar_pixels
)


original_nonzero = cp.count_nonzero(ans_matrix(pixels))
# Count nonzero coefficients for the best Bior53 image
bior53_nonzero = sum(
    count_nonzero_coefficients(LL) + count_nonzero_coefficients(LH) +
    count_nonzero_coefficients(HL) + count_nonzero_coefficients(HH)
    for LL, LH, HL, HH in bior53_pixels
)

# Count nonzero coefficients for the best Bior97 image
bior97_nonzero = sum(
    count_nonzero_coefficients(LL) + count_nonzero_coefficients(LH) +
    count_nonzero_coefficients(HL) + count_nonzero_coefficients(HH)
    for LL, LH, HL, HH in bior97_pixels
)
print(f"Nonzero coefficients for the original image: {original_nonzero}")
# Display the results
print(f"Nonzero coefficients for DCT: {dct_nonzero}")
print(f"Nonzero coefficients for WHT: {wht_nonzero}")
print(f"Nonzero coefficients for Haar: {haar_nonzero}")
print(f"Nonzero coefficients for Bior53: {bior53_nonzero}")
print(f"Nonzero coefficients for Bior97: {bior97_nonzero}")
