import numpy as np
from src.config import DTYPE

def generate_matrices(n):
    """Genera matrices aleatorias en el Host."""
    print(f"[HOST] Generando matrices {n}x{n}...")
    A = np.random.rand(n, n).astype(DTYPE)
    B = np.random.rand(n, n).astype(DTYPE)
    C = np.zeros((n, n), dtype=DTYPE)
    return A, B, C