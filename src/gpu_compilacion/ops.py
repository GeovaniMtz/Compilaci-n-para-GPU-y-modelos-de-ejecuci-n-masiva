import numpy as np
import time

def matrix_mult_cpu(A, B):
    """
    Multiplicación de matrices (C = A x B) de forma secuencial.
    """
    N = A.shape[0]  
    M = B.shape[1]  
    K = A.shape[1]  

    C = np.zeros((N, M), dtype=np.float32)

    # Multiplicación de matrices secuencial
    start = time.time()
    for i in range(N):
        for j in range(M):
            suma = 0.0
            for k in range(K):
                suma += A[i, k] * B[k, j]
            C[i, j] = suma
    tiempo = time.time() - start
    return C, tiempo

def matrix_add_cpu(A, B):
    """
    Suma de matrices (C = A + B) de forma secuencial.
    """
    N, M = A.shape[0], A.shape[1]
    C = np.zeros((N, M), dtype=np.float32)

    start = time.time()
    for i in range(N):
        for j in range(M):
            C[i, j] = A[i, j] + B[i, j]
    tiempo = time.time() - start
    return C, tiempo