# Proyecto 02 — Compilación para GPU con Numba CUDA (JIT → PTX)

Este repositorio contiene un **programa demostrativo ejecutable** que toma operaciones matriciales básicas (**multiplicación** y **suma**) y las ejecuta como **kernels paralelos** en GPU usando **Numba CUDA (JIT)**.

El objetivo del programa es hacer explícito:

- **CPU (secuencial):** el cálculo se expresa como bucles `for` (por ejemplo, 2–3 bucles anidados).
- **GPU (paralelo):** esos índices del bucle se mapean a **coordenadas de hilos** con `cuda.grid(2)`.
- **Compilación JIT:** la **primera** ejecución en GPU tiene un costo de **compilar** el kernel alto (generando código PTX). Las siguientes ejecuciones reutilizan lo ya compilado para la misma firma de tipos.


---
# ¡ IMPORTANTE !
> Si no se cuenta con una GPU NVIDIA local, se puede ejecutar el proyecto en **Google Colab**: descargar y comprimir el proyecto como **ZIP**, súbirlo a Colab, **descomprímelo** y ejecutar el script activando un entorno con **GPU (T4)** (Instrucciones complestas abajo).
---

## Estructura del repositorio

```text
CompilacionGPU/
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
````

---

## Requisitos

* **Python 3.10–3.12** (probado con 3.11)
* **GPU NVIDIA** con **drivers** instalados (CUDA Driver)

  * Si no hay GPU disponible, el programa **corre en CPU** y lanza un **error para de GPU**.

### Entorno virtual (Linux Debian/Ubuntu y derivados)

En estos sistemas, `pip` puede bloquear instalaciones globales por **PEP 668** (`externally-managed-environment`).
Por lo tanto, este proyecto se instala y ejecuta **solo dentro de un `venv`**.

---

## Dependencias

El archivo `requerimientos.txt` incluye:

* `numba`
* `numpy`

> No se recomienda instalar dependencias “globales” con `pip` en Linux.

---

## Instalación

### Linux/macOS (venv obligatorio)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requerimientos.txt
```

### Windows PowerShell

```powershell
py -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requerimientos.txt
```

---

## Ejecución rápida

Desde la raíz del repo (con el `venv` activo):

```bash
python src/run.py 1024
```

Donde `1024` es el tamaño `N` para matrices `N×N`.

### ¿Qué imprime el programa?

* tiempos de **CPU secuencial**
* tiempos de **GPU** separados en:

  * **tiempo de kernel** (ejecución en GPU)
  * **tiempo de transferencias** (Host↔Device)
  * **tiempo total GPU**
* **speedup** aproximado
* **validación** (`CPU` vs `GPU`) con tolerancia numérica

---

## Costo de compilación JIT

Ejecuta el programa **dos veces** con el mismo `N`:

```bash
python src/run.py 1024
python src/run.py 1024
```

La **primera ejecución** puede ser más lenta del lado GPU por el costo de **compilación JIT** (además de transferencias/launch). En la segunda ya se reutiliza el kernel compilado (misma firma de tipos).

---

## Ejecutar en Google Colab (sin GPU NVIDIA local)

1. En Colab abrir: `Entorno de ejecución > Cambiar tipo de entorno de ejecución`

   * **Acelerador por hardware:** `GPU T4`

2. Sube el proyecto como `.zip` (desde el panel de archivos) y descomprímelo.
   Luego, en una celda:

```bash
!unzip -q CompilacionGPU.zip -d CompilacionGPU
%cd CompilacionGPU
!pip -q install -r requerimientos.txt
!python src/run.py 1024
```

> Si el nombre del zip/carpeta cambia, se debe ajsutar `CompilacionGPU.zip` y/o la ruta del `cd`.


---


## Referencias (documentación)

```text
Numba CUDA (docs): https://numba.readthedocs.io/en/stable/
NVIDIA PTX ISA:    https://docs.nvidia.com/cuda/parallel-thread-execution/index.html
```
