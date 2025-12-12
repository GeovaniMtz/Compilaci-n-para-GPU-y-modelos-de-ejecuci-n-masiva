from numba import cuda
import numpy as np
import math
import time

@cuda.jit
def matrix_mult_kernel(A, B, C):
    """ 
    En esta función realiza la multiplicación de matrices A y B
    a nivel de kernel en GPU, escribiendo el resultado en la matriz C.
    """
    # Transformación Loop-to-Grid
    i, j = cuda.grid(2) 
    
    N, M = C.shape[0], C.shape[1]
    K = A.shape[1]

    if i < N and j < M:
        suma = 0.0
        for k in range(K):
            suma += A[i, k] * B[k, j]
        C[i, j] = suma

def run_matrix_mult_gpu(A, B, threads_per_block=(32, 32)):
    """
    Ejecuta la multiplicación de matrices en GPU.
    Realiza la transferencia de datos, ejecución del kernel y
    transferencia de resultados de vuelta al host.
    """
    N, M = A.shape[0], B.shape[1]
    C_host = np.zeros((N, M), dtype=np.float32)
    
    # Transferencia Host -> Device
    start_transfer_htod = time.time()
    A_device = cuda.to_device(A)
    B_device = cuda.to_device(B)
    C_device = cuda.to_device(C_host.copy())
    tiempo_transfer_htod = time.time() - start_transfer_htod
    
    # Ejecución del Kernel
    start_kernel = time.time()
    
    # Cálculo de la Jerarquía de Hilos
    blocks_x = math.ceil(N / threads_per_block[0])
    blocks_y = math.ceil(M / threads_per_block[1])
    blocks_per_grid = (blocks_x, blocks_y)

    # Ejecución de código PTX compilado por JIT.
    matrix_mult_kernel[blocks_per_grid, threads_per_block](A_device, B_device, C_device)
    cuda.synchronize()
    tiempo_gpu_kernel = time.time() - start_kernel

    # Transferencia Device -> Host
    start_dtoh = time.time()
    C_device.copy_to_host(C_host)
    tiempo_transfer_dtoh = time.time() - start_dtoh
    
    tiempo_gpu_total = tiempo_transfer_htod + tiempo_gpu_kernel + tiempo_transfer_dtoh
    
    return C_host, tiempo_gpu_total, tiempo_gpu_kernel, tiempo_transfer_htod + tiempo_transfer_dtoh