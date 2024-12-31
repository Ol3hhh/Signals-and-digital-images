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

step_t = 0
i = 0
step_r = l-1

min_t = 0
max_t = 2*np.pi

min_r = (1/l)*step_r
max_r = 1
# Set max and min t dynamically for filtering b
for _ in range(l):
    b = np.linspace(0 + step_t, 2*np.pi/l + step_t, 50)
    
    # Filter b based on the condition (b between min_t and max_t)
    
    a2.append(np.where((b > min_t) & (b < max_t), b, np.nan))
    
    d = f(b)
    
    # Filter d values between (1/l)*(l-1) and 1
    r2.append(np.where((d > min_r) & (d < max_r), d, np.nan))
    
    m += round(64/l)
    step_t += 2*np.pi/l
    step_r -= 1
    print(min_r, ' ', max_r)
    print(r2)
    min_r -= 1/l*step_r 
    max_r -= 1/l*step_r 
    

# Concatenate the list of arrays into a single array
a2 = np.concatenate(a2)
r2 = np.concatenate(r2)

# Plot the static part on the second polar plot
ax2.plot(a2, r2)
print((1/l)*(l-1))

# Display the plot
plt.show()
