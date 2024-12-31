import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from PIL import Image
import os
import re

full_dir = 'Py/photos/full'
cropped_dir = 'Py/photos/cropped'


def clear_directory(directory):
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)

clear_directory(full_dir)
clear_directory(cropped_dir)

n = 5
m = -32
l = 32
size = 512
part = size / l

def f(a):
    return np.sin(n * a + (m * np.pi / 10))

a = np.linspace(0, 2 * np.pi, 90)

fig, ax1 = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(5.12, 5.12), dpi=100)
ax1.set_yticklabels([])

r = f(a)
animated_frame, = ax1.plot(a, r)

i = 0
y = l

def update(frame):
    global m, i, y

    r = f(a)
    animated_frame.set_data(a, r)
    m += 1
    
    if m == 33:
        m = -32
        isDone = True

    if round(size / l) > i:
        plt.savefig(f'{full_dir}/main_screen{i}.png')
        image = Image.open(f'{full_dir}/main_screen{i}.png')
        crop_box = (0, y - l, 512, y)
        cropped_image = image.crop(crop_box) 
        cropped_image.save(f'{cropped_dir}/cropped_screen{i}.png')
        print(y)
    else:
        print("READY!!!")
    i += 1 
    y += l
    return animated_frame,

animation = FuncAnimation(fig, update, frames=64, interval=100)

plt.show()

output_image_path = 'combined_image.png'

cropped_images = [f for f in os.listdir(cropped_dir) if f.endswith('.png')]

def sort_key(filename):
    match = re.search(r'(\d+)', filename)
    return int(match.group(1)) if match else float('inf')

cropped_images.sort(key=sort_key)

if len(cropped_images) > 0:
    first_image = Image.open(os.path.join(cropped_dir, cropped_images[0]))
    width, height = first_image.size
else:
    exit()

total_width = width
total_height = height * len(cropped_images)

combined_image = Image.new('RGB', (total_width, total_height))

for index, image_file in enumerate(cropped_images):
    image = Image.open(os.path.join(cropped_dir, image_file))
    combined_image.paste(image, (0, index * height))

combined_image.save(output_image_path)

clear_directory(full_dir)
clear_directory(cropped_dir)