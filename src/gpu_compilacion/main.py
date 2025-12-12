import numpy as np
import time
from numba import cuda

from .ops import matrix_mult_cpu
from .kernels import run_matrix_mult_gpu

def crear_matrices(N):
    """
    Función auxiliar para crear matrices aleatorias de tamaño NxN.
    """
    np.random.seed(42)
    A = np.random.rand(N, N).astype(np.float32)
    B = np.random.rand(N, N).astype(np.float32)
    return A, B

def run_compilacion():
    """
    Función principal para ejecutar el proceso de compilación
    """
    
    # Configuración de tamaño de matriz y bloques/hilos
    MATRIX_SIZE = 512 
    THREADS_PER_BLOCK = (32, 32)
    
    print("="*80)
    print("Proceso de compilación para GPU: Multiplicación de Matrices")
    print("="*80)

    # Inicialización de datos: Matrices A y B
    A, B = crear_matrices(MATRIX_SIZE)
    total_threads_needed = MATRIX_SIZE * MATRIX_SIZE
    
    print(f"\nConfiguración:")
    print(f"Matriz de Cómputo (N): {MATRIX_SIZE}x{MATRIX_SIZE}")
    print(f"Lógica Secuencial (CPU): 3 bucles anidados (O(N³))")
    print(f"Lógica Paralela (GPU): {total_threads_needed:,} hilos de ejecución.")
    
    # Se ejecuta la fase secuencial (CPU)
    print("\n" + "="*80)
    print("Fase secuencial de compilación y ejecución (CPU)")
    print("="*80)
    C_cpu, tiempo_cpu_seq = matrix_mult_cpu(A, B)
    print(f"Tiempo de Ejecución (CPU): {tiempo_cpu_seq:.4f} segundos")

    # Se ejecuta la fase de compilación y ejecución en GPU
    try:
        if not cuda.is_available():
            print("\nNo se detectó una GPU compatible con CUDA")
            return

        print("\n" + "="*80)
        print("Fase secuencial de compilación y ejecución (GPU)")
        print("="*80)

        # Compilación JIT (Warm-up)
        print("A) COMPILACIÓN JIT (Python -> PTX):")
        start_jit = time.time()
        
        # Se manda a llamar una vez para forzar la compilación JIT y cacheo
        run_matrix_mult_gpu(A[:1,:1], B[:1,:1], threads_per_block=(1, 1)) 
        tiempo_jit_warmup = time.time() - start_jit
        print(f"Compilador Numba JIT activado, PTX (IR) generado y cacheado. (Tiempo: {tiempo_jit_warmup:.4f}s)")
        
        # Se transfiere data HOST -> DEVICE
        print("\nB) TRANSFERENCIA HOST -> DEVICE:")

        # La función run_matrix_mult_gpu ya mide esto internamente.
        print("Se mueven los datos de la memoria HOST (CPU) a la memoria DEVICE (GPU)...")
        
        # Ejecución del Kernel (Mapeo Loop-to-Grid)
        print("\nC) MAPEO ESPACIAL Y EJECUCIÓN (SIMT):")
        print(f"El compilador rompe los ciclos y asigna la tarea a {total_threads_needed:,} hilos concurrentes...")
        
        C_gpu, tiempo_gpu_total, tiempo_gpu_kernel, tiempo_transfer = run_matrix_mult_gpu(A, B, threads_per_block=THREADS_PER_BLOCK)

        print(f"Tiempo Cómputo Kernel (Puro): {tiempo_gpu_kernel:.4f} segundos")
        print(f"Tiempo Transferencias (Overhead): {tiempo_transfer:.4f} segundos")
        print(f"Tiempo TOTAL GPU: {tiempo_gpu_total:.4f} segundos")
        
        # Resultados y Análisis
        print("\n" + "="*80)
        print("Resultados y Análisis de la Transformación")
        print("="*80)
        
        speedup_kernel_vs_seq = tiempo_cpu_seq / tiempo_gpu_kernel
        
        print(f"\n| {'Métrica':<35} | {'Valor':<10} | {'Análisis':<30} |")
        print("|" + "-"*36 + "|" + "-"*12 + "|" + "-"*32 + "|")
        print(f"| {'Tiempo CPU (Lógica Secuencial)':<35} | {tiempo_cpu_seq:<12.4f} | {'Base de comparación'}")
        print(f"| {'Tiempo GPU (Kernel Puro)':<35} | {tiempo_gpu_kernel:<12.4f} | {'Mapeo Loop-to-Grid'}")
        print(f"| {'Speedup (Kernel vs CPU)':<35} | {f'{speedup_kernel_vs_seq:.2f}x':<12} | {f'Aceleración por gpu'}")

        # Validación
        valid = np.allclose(C_cpu, C_gpu, atol=1e-5)
        print(f"\nValidación (CPU vs GPU): {'COOINCIDEN' if valid else 'FALLO'} - Demuestra la precisión de la traducción JIT.")
        
    except Exception as e:
        print(f"\n[ERROR en GPU]: {e}")
        print("Verifique su entorno CUDA y la instalación de Numba.")
