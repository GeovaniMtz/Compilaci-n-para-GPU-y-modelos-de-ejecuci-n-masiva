# Proyecto 02 — Compilación para GPU (Numba CUDA)

Este repositorio contiene un **programa demostrativo ejecutable** que toma operaciones matriciales comunes (suma, escalamiento y multiplicación de matrices)
y las ejecuta como **kernels paralelos** en GPU usando **Numba CUDA (JIT)**.

La idea central es mostrar **cómo se traduce la abstracción de bucles (CPU)** a **coordenadas de hilos (GPU)** mediante `cuda.grid(2)`,
y cómo **la primera ejecución** incurre en el costo de **compilación JIT a PTX**, mientras que las siguientes reutilizan el kernel ya compilado.

## Requisitos

- Python 3.10–3.12 (probado con 3.11)
- GPU NVIDIA con drivers instalados (CUDA Driver).  
  > Si no hay GPU disponible, el script corre en modo CPU y avisa.

Dependencias (pip):

```bash
pip install -r requirements.txt
```

## Ejecución rápida

### 1) Element-wise: `C = (A + B) * alpha`

```bash
python run.py elemwise --m 4096 --n 4096 --alpha 2.0 --bench --verify
```

### 2) Multiplicación de matrices: `C = A @ B`

```bash
python run.py matmul --m 1024 --k 1024 --n 1024 --bench --verify
```

### 3) Imprimir PTX (evidencia de compilación)

```bash
python run.py matmul --m 256 --k 256 --n 256 --print-ptx
```

## ¿Qué muestra este demo?

- **Loop-to-Grid:** el par `(i, j)` del doble bucle externo se convierte en `(row, col)` calculado por cada hilo con `cuda.grid(2)`.
- **JIT:** Numba compila el kernel la primera vez que se llama (según tipos/tamaños), generando PTX.
- **Overhead vs throughput:** con tamaños pequeños la GPU puede perder por el costo fijo (transferencias + launch); con tamaños grandes tiende a ganar.
- **Kernel Fusion (opcional):** se incluyen dos versiones de `C=(A+B)*alpha`:
  - 2 kernels: `add_kernel` + `scale_kernel`
  - 1 kernel fusionado: `fused_add_scale_kernel`

## Estructura

```
proyecto_gpu_compilacion/
├─ run.py
├─ requirements.txt
└─ src/
   └─ gpu_compilacion/
      ├─ kernels.py        # kernels CUDA (Numba)
      ├─ ops.py            # envolturas CPU/GPU (to_device, launch, etc.)
      └─ bench.py          # medición de tiempos (CPU y GPU)
```

## Notas de instalación (Windows)

Si `pip` te instala un paquete raro como `numba_cuda` o hay conflictos, crea un venv limpio:

```powershell
py -m venv .venv
.\.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

Si `cuda.is_available()` da `False`, normalmente es por drivers NVIDIA faltantes o por ejecutar en una VM sin passthrough.
