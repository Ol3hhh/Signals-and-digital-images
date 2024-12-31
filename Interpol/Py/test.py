import numpy as np

x = np.array([1, 3, 3])
y = np.array([5, 2, 0])

valid = (x > 2) & (y > 0)
print(valid)
# Вибираємо індекси для валідних пікселів
valid_y_indices = y[valid]

# Виводимо результат
print(valid_y_indices)


