import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

# Set initial parameters
n = 3
m = -32
l = 16
size = 256 

# Function to calculate the sine wave based on a
def f(a):
    return np.sin(n * a + (m * np.pi / 10))

# Generate initial angle values with a lower sampling rate
a = np.linspace(0, 2 * np.pi, 90)  # Reduced number of points to create aliasing

# Initialize the figure and axis
fig, (ax1, ax2) = plt.subplots(1, 2, subplot_kw={'projection': 'polar'})
ax1.set_yticklabels([])  # Remove radial tick labels

r = f(a)
animated_frame, = ax1.plot(a, r)

ax2.set_yticklabels([])

# Update function to animate the plot
def update(frame):
    global m

    r = f(a)
    animated_frame.set_data(a, r)
    m += 1
    
    if m == 33:
        m = -32
    
    return animated_frame,

# Create the animation
animation = FuncAnimation(fig, update, frames=64, interval=100)

# Draw the part of the given function on the second plot in [0.6;1]
a2 = []
r2 = []
m = -32

step = 0

for _ in range(l):
    b = np.linspace(0 + step, 2*np.pi/l + step, 50)
    a2.append(b)
    d = f(b)
    r2.append(d)
    m += round(64/l)
    step += 2*np.pi/l

a2 = np.concatenate(a2)  # Concatenate the list of arrays into a single array
r2 = np.concatenate(r2)  # Concatenate the list of arrays into a single array

ax2.plot(a2, r2)

# Display the plot
plt.show()
