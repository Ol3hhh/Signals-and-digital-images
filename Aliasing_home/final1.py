import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

n = 3
m = -32

# Function to calculate the radius based on the angle
def f(a):
    return np.sin(n * a + (m * np.pi / 10))

# Define the angles from 0 to 2π
a = np.linspace(0, 2 * np.pi, 90)

# Create a figure with two polar subplots
fig, (ax1, ax2) = plt.subplots(1, 2, subplot_kw={'projection': 'polar'})
ax1.set_yticklabels([])  # Hide the radial tick labels

# Initial plot on the first subplot
r = f(a)
animated_frame, = ax1.plot(a, r)

# Set theta offset for proper orientation
ax1.set_theta_offset(np.pi / 2)
ax2.set_theta_offset(np.pi / 2)

# Update function for animation
def update(frame):
    global m
    r = f(a)
    animated_frame.set_data(a, r)
    m += 1
    if m == 33:
        m = -32
    return animated_frame,

# Create the animation
anim = FuncAnimation(fig, update, frames=64, interval=100)

# Lists to hold aliased data
a2 = []
r2 = []
l = 16
part = 1 / l 

min_a = 0
max_a = np.pi
min_r = part * (l-1)
max_r = 1

# Function to perform aliasing
def aliasing():
    global r2, a2, min_r, max_r
    
    for _ in range(l):
        # Filter angles and calculate corresponding radii
        filtred_a = a[(a > min_a) & (a < max_a)]
        d = f(filtred_a)  
        
        # Filter radii based on minimum and maximum
        filtred_r = d[(d > min_r) & (d < max_r)]
        
        # Store filtered angles and radii
        a2.append(filtred_a)
        r2.append(filtred_r)
        print(len(a2), ' ', len(r2))
        # Update min_r and max_r for next iteration
        min_r -= part
        max_r -= part

# Call the aliasing function
aliasing()
a2 = np.concatenate(a2)  # Convert list to a single arra
r2 = np.concatenate(r2)  # Convert list to a single array
print(len(a2), ' ', len(r2))
# Plot the aliased data on the second subplot
ax2.plot(a2, r2)  # Use markers for clarity



# Show the final plot
plt.show()
