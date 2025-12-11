import sys
import os

# Agregamos el directorio actual al path para poder importar 'src'
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.main import run_simulation

if __name__ == "__main__":
    try:
        run_simulation()
    except Exception as e:
        print(f"ERROR CRÍTICO: {e}")
        print("Asegúrate de tener una GPU NVIDIA y CUDA Toolkit instalado.")