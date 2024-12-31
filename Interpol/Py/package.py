import cv2

# Path to the image
path = 'Fella.jpg'
image = cv2.imread(path, -1)

# Get the original dimensions
height, width, channels = image.shape



def scale1d(a):
    return cv2.resize(image, (int(width * a), int(height * a)), interpolation=cv2.INTER_LINEAR)


def rotate(image, center, alpha):
    height, width, channels = image.shape
    M = cv2.getRotationMatrix2D(center, alpha, 1.0)
    return cv2.warpAffine(image, M, (width, height))


image = scale1d(0.1)
image = rotate(image, (0, 0), 15)
# Display the resized image
cv2.imshow("Resized Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
