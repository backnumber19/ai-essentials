import numpy as np
import time
import random


def gaussian_jordan_inverse(A):
    n = A.shape[0]
    A = A.astype(float)
    augmented = np.hstack([A, np.eye(n)])
    for i in range(n):
        max_row = i + np.argmax(np.abs(augmented[i:, i]))
        if max_row != i:
            augmented[[i, max_row]] = augmented[[max_row, i]]
        pivot = augmented[i, i]
        augmented[i] = augmented[i] / pivot
        for j in range(n):
            if i != j:
                factor = augmented[j, i]
                augmented[j] = augmented[j] - factor * augmented[i]
    return augmented[:, n:]


def lissa_inverse(A, num_lissa_iter, batch_ratio, base_stepsize=1e-4):
    n = A.shape[0]
    hessian_batch_size = max(1, int(n * batch_ratio))
    A_norm = np.linalg.norm(A, 2)
    stepsize = base_stepsize / (A_norm * A_norm)
    X = np.zeros((n, n))
    I = np.eye(n)

    for _ in range(num_lissa_iter):
        if hessian_batch_size < n:
            rand_indices = random.sample(range(n), hessian_batch_size)
            A_batch = A[rand_indices, :]
            I_batch = I[rand_indices, :]
        else:
            A_batch = A
            I_batch = I
        AX = A_batch @ X
        residual = I_batch - AX
        gradient = A_batch.T @ residual
        X = X + stepsize * gradient
    return X


sizes = [100, 500, 1000]

for n in sizes:
    print(f"\nMatrix size: {n}x{n}")

    np.random.seed(0)
    random.seed(0)

    A = np.random.randn(n, n)
    A = A @ A.T + 0.1 * np.eye(n)

    start_time = time.time()
    inv_gaussian = gaussian_jordan_inverse(A)
    time_gaussian = time.time() - start_time

    start_time = time.time()
    inv_lissa = lissa_inverse(A, num_lissa_iter=150, batch_ratio=0.5)
    time_lissa = time.time() - start_time

    error_gaussian = np.linalg.norm(A @ inv_gaussian - np.eye(n)) / n
    error_lissa = np.linalg.norm(A @ inv_lissa - np.eye(n)) / n

    print(f"Gaussian-Jordan: {time_gaussian:.2f}s (error: {error_gaussian:.2f})")
    print(f"LiSSA: {time_lissa:.2f}s (error: {error_lissa:.2f})")
    print(f"LiSSA Speedup: {time_gaussian/time_lissa:.2f}x\n")
