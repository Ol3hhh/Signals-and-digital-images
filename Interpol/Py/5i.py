from PIL import Image
from nn2d import nearest_neighbor_rescale_2d as nn

image = Image.open("el.png")


step = 0.1
k = 1
for i in range(5):
    k += k * step + k
    image = nn(k, image)
    print(i)

k = 1/k
image = nn(k, image)
image.show()
image.save("../New/21/nn2d.jpg")