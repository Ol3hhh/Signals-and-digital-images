import numpy as np
from scipy.special import factorial

np.set_printoptions(precision=6, suppress=True)

index = np.array([2, 8, 0, 0, 6, 5])

def P(lm, k):
    # lambda - lm
    return (lm**k * np.exp(-lm)) / factorial(k, exact=False)

# Zadanie 1: P(X = k_i) dla λ = 1
print("P(X = k_i) dla λ = 1:")
print(P(1, index))

print("______________")

# Zadanie 2: P(X = 0) dla różnych λ = k_i
print("P(X = 0) dla różnych λ = k_i:")
print(P(index, 0))

print("______________")

# Zadanie 3: P(X ≥ Σk_i)
Lambda_sum = index.sum()  # Suma k_i
ps = sum(P(Lambda_sum, i) for i in range(int(Lambda_sum)))  # P(X < Σk_i) ps - probability sum

print("P(X ≥ Σk_i):")
print(1 - ps)

print("______________")

# Zadanie 4: P(X ≤ Σk_i) dla efektywności QE = π/4
QE = np.pi / 4
Lambda_eff = QE * Lambda_sum

P_le = sum(P(Lambda_eff, i) for i in range(int(Lambda_sum) + 1))  # P(X ≤ Σk_i)
print("P(X ≤ Σk_i) dla QE = π/4:")
print(P_le)
