import numpy as np
import time
from numba import cuda

# Importaciones de la Lógica (NO se modifica la lógica de cálculo aquí)
from .ops import matrix_mult_cpu
from .kernels import run_matrix_mult_gpu

def crear_matrices(N):
    """Utilidad simple para crear matrices."""
    np.random.seed(42)
    A = np.random.rand(N, N).astype(np.float32)
    B = np.random.rand(N, N).astype(np.float32)
    return A, B

def run_compilacion():
    """
    Ejecuta la demostración, mostrando explícitamente el proceso de compilación 
    (JIT, Transferencia y Mapeo Loop-to-Grid) y la comparación de rendimiento.
    """
    
    # Configuración de tamaño de matriz (grande para forzar la diferencia)
    MATRIX_SIZE = 512 
    THREADS_PER_BLOCK = (32, 32)
    
    print("="*80)
    print("  PROYECTO: COMPILACIÓN PARA GPU Y MODELOS DE EJECUCIÓN MASIVA")
    print("  Demostración del Proceso de Traducción y Paralelismo Explícito")
    print("="*80)

    # Inicialización de datos
    A, B = crear_matrices(MATRIX_SIZE)
    total_threads_needed = MATRIX_SIZE * MATRIX_SIZE
    
    print(f"\n📊 Configuración:")
    print(f"  - Matriz de Cómputo (N): {MATRIX_SIZE}x{MATRIX_SIZE}")
    print(f"  - Lógica Secuencial (CPU): 3 ciclos anidados (O(N³))")
    print(f"  - Lógica Paralela (GPU): {total_threads_needed:,} hilos de ejecución.")
    
    # --- 1. EJECUCIÓN SECUENCIAL (CPU) ---
    print("\n" + "="*80)
    print("--- 1. Fase Secuencial (CPU) ---")
    print("="*80)
    print("-> Ejecutando Lógica de Alto Nivel (matrix_mult_cpu)...")
    C_cpu, tiempo_cpu_seq = matrix_mult_cpu(A, B)
    print(f"⏱️  Tiempo de Ejecución (CPU Pura): {tiempo_cpu_seq:.4f} segundos")
    print("   (Esto demuestra la ineficiencia que el compilador debe resolver.)")

    # --- 2. EJECUCIÓN PARALELA (GPU) - PROCESO DE COMPILACIÓN ---
    try:
        if not cuda.is_available():
            print("\n❌ ERROR: No se detectó una GPU compatible con CUDA. No se puede continuar.")
            return

        print("\n" + "="*80)
        print("--- 2. Fase de Compilación y Ejecución GPU ---")
        print("="*80)

        # A) COMPILACIÓN JIT (Warm-up)
        print("A) COMPILACIÓN JIT (Python -> PTX):")
        start_jit = time.time()
        # Se invoca el Kernel por primera vez. Numba traduce el código Python a PTX (IR).
        run_matrix_mult_gpu(A[:1,:1], B[:1,:1], threads_per_block=(1, 1)) 
        tiempo_jit_warmup = time.time() - start_jit
        print(f"   ✓ Compilador Numba JIT activado, PTX (IR) generado y cacheado. (Tiempo: {tiempo_jit_warmup:.4f}s)")
        
        # B) TRANSFERENCIA (Desafío de Arquitectura Heterogénea)
        print("\nB) TRANSFERENCIA HOST -> DEVICE:")
        # La función run_matrix_mult_gpu ya mide esto internamente.
        print("   -> Se mueven los datos de la memoria HOST (CPU) a la memoria DEVICE (GPU)...")
        
        # C) EJECUCIÓN DEL KERNEL (Mapeo Loop-to-Grid)
        print("\nC) MAPEO ESPACIAL Y EJECUCIÓN (SIMT):")
        print(f"   -> El compilador rompe los bucles y asigna la tarea a {total_threads_needed:,} hilos concurrentes...")
        
        C_gpu, tiempo_gpu_total, tiempo_gpu_kernel, tiempo_transfer = run_matrix_mult_gpu(A, B, threads_per_block=THREADS_PER_BLOCK)

        print(f"   ⏱️  Tiempo Cómputo Kernel (Puro): {tiempo_gpu_kernel:.4f} segundos")
        print(f"   ⏱️  Tiempo Transferencias (Overhead): {tiempo_transfer:.4f} segundos")
        print(f"   ⏱️  Tiempo TOTAL GPU: {tiempo_gpu_total:.4f} segundos")
        
        # --- 3. ANÁLISIS DE RESULTADOS ---
        print("\n" + "="*80)
        print("--- 3. Resultados y Análisis de la Transformación ---")
        print("="*80)
        
        speedup_kernel_vs_seq = tiempo_cpu_seq / tiempo_gpu_kernel
        
        print(f"\n| {'Métrica':<35} | {'Valor':<10} | {'Análisis':<30} |")
        print("|" + "-"*36 + "|" + "-"*12 + "|" + "-"*32 + "|")
        print(f"| {'Tiempo CPU (Lógica Secuencial)':<35} | {tiempo_cpu_seq:<12.4f} | {'Base de comparación'}")
        print(f"| {'Tiempo GPU (Kernel Puro)':<35} | {tiempo_gpu_kernel:<12.4f} | {'Cómputo tras Mapeo Loop-to-Grid'}")
        print(f"| {'Speedup (Kernel vs CPU)':<35} | {f'{speedup_kernel_vs_seq:.2f}x':<12} | {f'Aceleración por Paralelismo Explícito'}")

        # Validación
        valid = np.allclose(C_cpu, C_gpu, atol=1e-5)
        print(f"\n✓ Validación (CPU vs GPU): {'COINCIDEN' if valid else 'FALLA'} - Demuestra la precisión de la traducción JIT.")
        
    except Exception as e:
        print(f"\n❌ [ERROR en GPU]: {e}")
        print("Verifique su entorno CUDA y la instalación de Numba.")
    
    print("\n" + "="*80)
    print("  FIN DE LA DEMOSTRACIÓN EXPLÍCITA")
    print("="*80)