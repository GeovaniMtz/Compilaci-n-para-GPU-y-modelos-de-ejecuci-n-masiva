from numba import cuda, float32

# Tamaño del tile (debe coincidir con BLOCK_DIM en config.py)
TPB = 16 

@cuda.jit
def matmul_kernel_optimized(A, B, C):
    """
    Multiplicación de matrices optimizada usando MEMORIA COMPARTIDA (Tiling).
    
    Demuestra:
    - Transformación Loop→Grid (bucles externos desaparecen)
    - Uso de memoria compartida para eficiencia
    - Sincronización de hilos
    """
    
    # Arrays en memoria compartida (rápida, on-chip)
    sA = cuda.shared.array(shape=(TPB, TPB), dtype=float32)
    sB = cuda.shared.array(shape=(TPB, TPB), dtype=float32)

    # TRANSFORMACIÓN CLAVE: Los bucles 'for i' y 'for j' desaparecen
    # Se reemplazan por coordenadas calculadas del hilo
    x, y = cuda.grid(2)  # Coordenada espacial en la grilla
    tx = cuda.threadIdx.x  # Posición dentro del bloque
    ty = cuda.threadIdx.y
    
    tmp = 0.0

    # Número de tiles necesarios para cubrir el ancho
    num_tiles = (A.shape[1] + TPB - 1) // TPB
    
    for i in range(num_tiles):
        # --- FASE 1: CARGA A MEMORIA COMPARTIDA ---
        idx_k = i * TPB + ty
        idx_k_B = i * TPB + tx
        
        if x < A.shape[0] and idx_k_B < A.shape[1]:
            sA[tx, ty] = A[x, idx_k_B]
        else:
            sA[tx, ty] = 0.0

        if idx_k < B.shape[0] and y < B.shape[1]:
            sB[tx, ty] = B[idx_k, y]
        else:
            sB[tx, ty] = 0.0

        cuda.syncthreads()  # Esperar carga cooperativa

        # --- FASE 2: CÓMPUTO ---
        for k in range(TPB):
            tmp += sA[tx, k] * sB[k, ty]

        cuda.syncthreads()  # Esperar cómputo antes de siguiente tile

    # --- FASE 3: ESCRITURA ---
    if x < C.shape[0] and y < C.shape[1]:
        C[x, y] = tmp