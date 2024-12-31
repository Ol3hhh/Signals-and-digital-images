import numpy as np

a = np.ones((1, 2, 3))

a = a.transpose((1, 0, 2))

print(a.shape)