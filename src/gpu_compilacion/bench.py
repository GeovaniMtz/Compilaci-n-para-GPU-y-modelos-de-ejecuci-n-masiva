import numpy as np
import time
from .ops import matrix_mult_cpu
from .kernels import run_matrix_mult_gpu
from numba import cuda

def run_benchmark(matrix_size=1024):
    """Ejecuta las pruebas CPU y GPU y compara el rendimiento."""
    print(f"--- Prueba de Compilación Masiva: Matriz {matrix_size}x{matrix_size} ---")

    np.random.seed(42)
    A = np.random.rand(matrix_size, matrix_size).astype(np.float32)
    B = np.random.rand(matrix_size, matrix_size).astype(np.float32)
    
    times = {}

    # --- Ejecución CPU (Secuencial) ---
    print("1. Ejecutando Lógica Secuencial (CPU)...")
    start_cpu = time.time()
    C_cpu = matrix_mult_cpu(A, B)
    end_cpu = time.time()
    times['cpu'] = end_cpu - start_cpu
    print(f"   Tiempo CPU: {times['cpu']:.4f} segundos")

    # --- Ejecución GPU (Paralela, con JIT y Kernel) ---
    try:
        if not cuda.is_available():
             print("\nAdvertencia: No se detectó una GPU CUDA. Skippeando prueba GPU.")
             return times

        print("2. Ejecutando Compilación Paralela (GPU - Numba JIT)...")
        start_gpu = time.time()
        C_gpu = run_matrix_mult_gpu(A, B)
        end_gpu = time.time()
        times['gpu'] = end_gpu - start_gpu
        print(f"   Tiempo GPU: {times['gpu']:.4f} segundos")
        
        # --- Análisis de Resultados (Evidencia Observable) ---
        if times['cpu'] > 0 and times['gpu'] > 0:
            speedup = times['cpu'] / times['gpu']
            print("\n==============================================")
            print("🚀 Análisis del Desafío de Compilación:")
            print(f"   Aceleración (Speedup) GPU/CPU: {speedup:.2f}x")
            print("   (Demuestra cómo la ruptura de la secuencialidad y el mapeo al Kernel (SIMT) produce ganancia de rendimiento)[cite: 82].")
            print("==============================================")

        # Validación de exactitud
        if np.allclose(C_cpu, C_gpu, atol=1e-5):
             print("   Validación: Resultados de CPU y GPU coinciden. ✅")
        else:
             print("   Validación: Los resultados de CPU y GPU NO coinciden.")


    except Exception as e:
        print(f"\nError al ejecutar la prueba en GPU: {e}")
        print("Verifique la instalación de CUDA y Numba.")

    return times