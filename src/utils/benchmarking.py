import time
from numba import cuda

def measure_execution_time(func, *args, **kwargs):
    """
    Ejecuta una función y mide su tiempo de ejecución.
    Si es una función de GPU, fuerza la sincronización.
    
    Retorna: (resultado, tiempo_en_segundos)
    """
    if cuda.is_available():
        cuda.synchronize()
        
    start_time = time.time()
    result = func(*args, **kwargs)
    
    if cuda.is_available():
        cuda.synchronize()
        
    end_time = time.time()
    execution_time = end_time - start_time
    return result, execution_time

class PerformanceTracker:
    """Clase para guardar y comparar tiempos entre CPU y GPU."""
    def __init__(self):
        self.gpu_time = 0
        self.cpu_time = 0
        
    def print_stats(self):
        print("\n" + "="*40)
        print("     REPORTE DE RENDIMIENTO")
        print("="*40)
        print(f"⏱️  Tiempo CPU (NumPy): {self.cpu_time:.6f} s")
        print(f"🚀 Tiempo GPU (Numba): {self.gpu_time:.6f} s")
        
        if self.gpu_time > 0:
            speedup = self.cpu_time / self.gpu_time
            print(f"⚡ Aceleración (Speedup): {speedup:.2f}x")
        else:
            print("⚠️ No se pudo calcular el Speedup")
        print("="*40 + "\n")