import numpy as np
from src.config import TOLERANCE

def verify_results(A, B, C_gpu):
    """Compara resultado GPU vs CPU (NumPy) para asegurar corrección."""
    print("[VALIDACIÓN] Calculando referencia en CPU...")
    C_ref = np.dot(A, B)
    
    is_correct = np.allclose(C_gpu, C_ref, rtol=TOLERANCE, atol=TOLERANCE)
    
    if is_correct:
        print("✅ PASÓ: El resultado de la GPU coincide con la CPU.")
    else:
        diff = np.abs(C_gpu - C_ref).max()
        print(f"❌ FALLÓ: Diferencia máxima encontrada: {diff}")