import sys
import os

# --- Ajuste de PATH para importación ---
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from gpu_compilacion.main import run_compilacion

if __name__ == "__main__":
  run_compilacion()