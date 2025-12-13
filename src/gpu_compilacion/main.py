import numpy as np
import time
from numba import cuda

from .ops import matrix_mult_cpu, matrix_add_cpu 
from .kernels import run_matrix_mult_gpu, run_matrix_add_gpu 

def crear_matrices(N):
    """Función auxiliar para crear matrices aleatorias de tamaño NxN."""
    np.random.seed(42)
    A = np.random.rand(N, N).astype(np.float32)
    B = np.random.rand(N, N).astype(np.float32)
    return A, B

# AHORA ACEPTA EL TAMAÑO COMO PARÁMETRO
def run_compilacion(matrix_size):
    """
    Función principal para ejecutar el proceso de compilación
    """
    
    # Se usa el tamaño recibido desde run.py
    MATRIX_SIZE = matrix_size
    THREADS_PER_BLOCK = (32, 32)
    
    # ... (El resto del código sigue igual, pero usando MATRIX_SIZE)
    
    print("="*80)
    print("Proceso de compilación para GPU: Multiplicación y Suma de Matrices")
    print("="*80)

    # Inicialización de datos: Matrices A y B
    A, B = crear_matrices(MATRIX_SIZE)
    total_threads_needed = MATRIX_SIZE * MATRIX_SIZE
    
    print(f"\nConfiguración:")
    print(f"Matriz de Cómputo (N): {MATRIX_SIZE}x{MATRIX_SIZE}")
    print(f"Lógica Secuencial (CPU): 3 bucles anidados (O(N³)) para Mult. | 2 bucles (O(N²)) para Suma.")
    print(f"Lógica Paralela (GPU): {total_threads_needed:,} hilos de ejecución.")
    
    # Se ejecuta la fase secuencial primero para multiplicación
    print("\n" + "="*80)
    print("Fase secuencial de compilación y ejecución (CPU) - Multiplicación")
    print("="*80)
    C_cpu_mult, tiempo_cpu_mult_seq = matrix_mult_cpu(A, B)
    print(f"Tiempo de Ejecución (CPU Multiplicación): {tiempo_cpu_mult_seq:.4f} segundos")

    # Se ejecuta la fase de compilación y ejecución en GPU para multiplicación
    try:
        if not cuda.is_available():
            print("\nNo se detectó una GPU compatible con CUDA")
            return

        print("\n" + "="*80)
        print("Fase de compilación y ejecución (GPU) - Multiplicación")
        print("="*80)

        # Compilación JIT (Warm-up)
        print("A) COMPILACIÓN JIT (Python -> PTX):")
        start_jit = time.time()
        # Se manda a llamar una vez para forzar la compilación JIT de la multiplicación
        run_matrix_mult_gpu(A[:1,:1], B[:1,:1], threads_per_block=(1, 1)) 
        tiempo_jit_warmup_mult = time.time() - start_jit
        print(f"Compilador Numba JIT activado para Multiplicación. (Tiempo: {tiempo_jit_warmup_mult:.4f}s)")
        
        C_gpu_mult, tiempo_gpu_total_mult, tiempo_gpu_kernel_mult, tiempo_transfer_mult = run_matrix_mult_gpu(A, B, threads_per_block=THREADS_PER_BLOCK)

        print(f"\nRESULTADOS MULTIPLICACIÓN:")
        print(f"Tiempo Cómputo Kernel (Puro): {tiempo_gpu_kernel_mult:.4f} segundos")
        print(f"Tiempo Transferencias (Overhead): {tiempo_transfer_mult:.4f} segundos")
        print(f"Tiempo TOTAL GPU: {tiempo_gpu_total_mult:.4f} segundos")
        
        # Se ejecuta la fase secuencial para suma
        print("\n" + "="*80)
        print("Fase secuencial de compilación y ejecución (CPU) - Suma")
        print("="*80)
        C_cpu_add, tiempo_cpu_add_seq = matrix_add_cpu(A, B)
        print(f"Tiempo de Ejecución (CPU Suma): {tiempo_cpu_add_seq:.4f} segundos")

        # Fase de compilación y ejecución en GPU para suma
        print("\n" + "="*80)
        print("Fase de compilación y ejecución (GPU) - Suma")
        print("="*80)
        
        # ACompilación JIT (Warm-up) para SUMA
        print("A) COMPILACIÓN JIT (Python -> PTX):")
        start_jit = time.time()
        # Se manda a llamar una vez para forzar la compilación JIT de la SUMA
        run_matrix_add_gpu(A[:1,:1], B[:1,:1], threads_per_block=(1, 1)) 
        tiempo_jit_warmup_add = time.time() - start_jit
        print(f"Compilador Numba JIT activado para Suma. (Tiempo: {tiempo_jit_warmup_add:.4f}s)")

        C_gpu_add, tiempo_gpu_total_add, tiempo_gpu_kernel_add, tiempo_transfer_add = run_matrix_add_gpu(A, B, threads_per_block=THREADS_PER_BLOCK)
        
        print(f"\nRESULTADOS SUMA:")
        print(f"Tiempo Cómputo Kernel (Puro): {tiempo_gpu_kernel_add:.4f} segundos")
        print(f"Tiempo Transferencias (Overhead): {tiempo_transfer_add:.4f} segundos")
        print(f"Tiempo TOTAL GPU: {tiempo_gpu_total_add:.4f} segundos")


        # Resultados y Análisis
        print("\n" + "="*80)
        print("Resultados y Análisis de la Transformación (COMPLETOS)")
        print("="*80)
        
        speedup_mult = tiempo_cpu_mult_seq / tiempo_gpu_kernel_mult
        speedup_add = tiempo_cpu_add_seq / tiempo_gpu_kernel_add

        print(f"\n| {'Operación':<15} | {'CPU Seq (s)':<12} | {'GPU Kernel (s)':<15} | {'Speedup (x)':<12} |")
        print("|" + "-"*16 + "|" + "-"*14 + "|" + "-"*17 + "|" + "-"*14 + "|")
        print(f"| {'Multiplicación':<15} | {tiempo_cpu_mult_seq:<12.4f} | {tiempo_gpu_kernel_mult:<15.4f} | {f'{speedup_mult:.2f}x':<12} |")
        print(f"| {'Suma':<15} | {tiempo_cpu_add_seq:<12.4f} | {tiempo_gpu_kernel_add:<15.4f} | {f'{speedup_add:.2f}x':<12} |")

        # Validación
        valid_mult = np.allclose(C_cpu_mult, C_gpu_mult, atol=1e-5)
        valid_add = np.allclose(C_cpu_add, C_gpu_add, atol=1e-5)
        print(f"\nValidación Multiplicación: {'COINCIDEN' if valid_mult else 'FALLO'}")
        print(f"Validación Suma: {'COINCIDEN' if valid_add else 'FALLO'}")
        
    except Exception as e:
        print(f"\n[ERROR en GPU]: {e}")
        print("Verifique su entorno CUDA y la instalación de Numba.")