# Proyecto 02 — Compilación para GPU con Numba CUDA (JIT → PTX)

Este repositorio contiene un **programa demostrativo ejecutable** que toma operaciones matriciales básicas (**multiplicación** y **suma**) y las ejecuta como **kernels paralelos** en GPU usando **Numba CUDA (JIT)**.

El objetivo del demo es hacer explícito:

- **CPU (secuencial):** el cálculo se expresa como bucles `for` (por ejemplo, 2–3 bucles anidados).
- **GPU (paralelo):** esos índices del bucle se mapean a **coordenadas de hilos** con `cuda.grid(2)`.
- **Compilación JIT:** la **primera** ejecución en GPU paga el costo de **compilar** el kernel (generando código PTX). Las siguientes ejecuciones reutilizan lo ya compilado para la misma firma de tipos.

---

## Estructura del repositorio

```text
Compilador_GPU/
├── README.md
├── requerimientos.txt
├── scripts/
│   ├── set_path.ps1
│   └── set_path.sh
└── src/
    ├── run.py
    └── gpu_compilacion/
        ├── __init__.py
        ├── bench.py      # utilidades de medición/benchmark
        ├── kernels.py    # kernels CUDA (Numba @cuda.jit)
        ├── main.py       # flujo principal del demo (CPU vs GPU, prints)
        └── ops.py        # operaciones CPU (referencia secuencial)
```

---

## Requisitos

- **Python 3.10–3.12** (probado con 3.11)
- **GPU NVIDIA** con **drivers** instalados (CUDA Driver)
  - Si no hay GPU disponible, el programa **corre en CPU** y lo indica.

### Dependencias

El archivo `requerimientos.txt` incluye:

- `numba`
- `numpy`

Instalación:

```bash
pip install -r requerimientos.txt
````

> En Debian/Ubuntu puede aparecer el error `externally-managed-environment`. En ese caso usa un entorno virtual (abajo hay ejemplos).

---

## Ejecución rápida

Desde la raíz del repo:

```bash
python src/run.py 1024
```

Donde `1024` es el tamaño `N` para matrices `N×N`.

### ¿Qué imprime el programa?

Normalmente verás:

* tiempos de **CPU secuencial**
* tiempos de **GPU** separados en:

  * **tiempo de kernel** (ejecución en GPU)
  * **tiempo de transferencias** (Host↔Device)
  * **tiempo total GPU**
* **speedup** aproximado
* **validación** (`CPU` vs `GPU`) con tolerancia numérica

---

## Cómo evidenciar el costo de compilación JIT

Ejecuta el programa **dos veces** con el mismo `N`:

```bash
python src/run.py 1024
python src/run.py 1024
```

La **primera corrida** suele ser más lenta del lado GPU por el costo de **compilación JIT** (además de transferencias/launch). En la segunda ya se reutiliza el kernel compilado (misma firma de tipos).

---

## Instalación recomendada (Linux/macOS)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requerimientos.txt
python src/run.py 1024
```

---

## Instalación recomendada (Windows PowerShell)

```powershell
py -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requerimientos.txt
python src\run.py 1024
```

---

## Si `cuda.is_available()` sale en `False`

Causas típicas:

* Drivers NVIDIA no instalados / desactualizados
* Estás en una VM sin passthrough de GPU
* Estás en WSL sin configuración de CUDA para WSL

Alternativa rápida: ejecutar en **Google Colab** con runtime de GPU (solo instalas `numba` y `numpy`).

---

## (Opcional) Ver PTX generado por Numba

Numba permite inspeccionar el PTX del kernel ya compilado. Una forma práctica (después de forzar al menos una compilación/ejecución) es:

```python
from gpu_compilacion.kernels import matrix_mult_kernel

# después de que el kernel se haya compilado (por ejemplo, tras correr el programa)
sig = matrix_mult_kernel.signatures[0]
print(matrix_mult_kernel.inspect_ptx(sig))
```

> Nota: este snippet asume que el kernel ya generó al menos una firma en `signatures`.

---

## Referencias (documentación)

```text
Numba CUDA (docs): https://numba.pydata.org/numba-doc/latest/cuda/index.html
NVIDIA PTX ISA:    https://docs.nvidia.com/cuda/parallel-thread-execution/index.html
```

