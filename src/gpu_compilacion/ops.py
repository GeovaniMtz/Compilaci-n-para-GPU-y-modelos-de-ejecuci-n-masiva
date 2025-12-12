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

    # Lógica iterativa/secuencial (3 bucles anidados)
    start = time.time()
    for i in range(N):
        for j in range(M):
            suma = 0.0
            for k in range(K):
                suma += A[i, k] * B[k, j]
            C[i, j] = suma
    tiempo = time.time() - start
    return C, tiempo