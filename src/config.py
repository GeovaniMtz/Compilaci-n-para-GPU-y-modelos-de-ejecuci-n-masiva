# Configuración de Hardware
THREADS_PER_BLOCK_X = 16
THREADS_PER_BLOCK_Y = 16
BLOCK_DIM = (THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y)

# Configuración de Simulación
MATRIX_SIZE = 8192  # Tamaño para demostrar paralelismo masivo
DTYPE = 'float32'   # Precisión simple (estándar en GPUs)
TOLERANCE = 1e-2    # Tolerancia para validación