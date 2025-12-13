import sys
import os

# --- Ajuste de PATH para importación ---
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from gpu_compilacion.main import run_compilacion

if __name__ == "__main__":
    
    # Leer el tamaño de la matriz desde los argumentos de la terminal
    if len(sys.argv) < 2:
        print("\n[ERROR]")
        print("Uso: python run.py [N]")
        print("Ejemplo: python run.py 1024")
        sys.exit(1)
        
    try:
        matrix_size = int(sys.argv[1])
        if matrix_size <= 0:
            raise ValueError
    except ValueError:
        print("\n[ERROR]")
        print("El tamaño de la matriz (N) debe ser un número entero positivo.")
        sys.exit(1)

    run_compilacion(matrix_size)