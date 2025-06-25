# -*- coding: utf-8 -*-
"""
Python 3
05 / 07 / 2024
@author: z_tjona

"I find that I don't understand things unless I try to program them."
-Donald E. Knuth
"""

# ----------------------------- logging --------------------------
import logging
from sys import stdout
from datetime import datetime
import time
import numpy as np
import matplotlib.pyplot as plt

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][%(levelname)s] %(message)s",
    stream=stdout,
    datefmt="%m-%d %H:%M:%S",
)
logging.info(datetime.now())

# ----------------------------- variables globales para graficar --------------------------
_gauss_data = []
_jordan_data = []

# ####################################################################
def eliminacion_gaussiana(A: np.ndarray) -> np.ndarray:
    global _gauss_data
    if not isinstance(A, np.ndarray):
        A = np.array(A, dtype=float)
    assert A.shape[0] == A.shape[1] - 1
    A = A.copy()
    n = A.shape[0]

    suma = mult = div = resta = 0
    t0 = time.time()

    for i in range(0, n - 1):
        p = None
        for pi in range(i, n):
            if A[pi, i] == 0:
                continue
            if p is None or abs(A[pi, i]) < abs(A[p, i]):
                p = pi

        if p is None:
            raise ValueError("No existe solución única.")

        if p != i:
            A[[i, p]] = A[[p, i]]

        for j in range(i + 1, n):
            m = A[j, i] / A[i, i]
            div += 1
            A[j, i:] = A[j, i:] - m * A[i, i:]
            mult += len(A[i, i:])
            resta += len(A[i, i:])

    if A[n - 1, n - 1] == 0:
        raise ValueError("No existe solución única.")

    solucion = np.zeros(n)
    solucion[n - 1] = A[n - 1, n] / A[n - 1, n - 1]
    div += 1

    for i in range(n - 2, -1, -1):
        suma_i = 0
        for j in range(i + 1, n):
            suma_i += A[i, j] * solucion[j]
            mult += 1
            suma += 1
        solucion[i] = (A[i, n] - suma_i) / A[i, i]
        resta += 1
        div += 1

    t1 = time.time()
    total_op = suma + resta + mult + div
    tiempo = t1 - t0

    _gauss_data.append((tiempo, total_op))

    logging.info(f"[GAUSSIANA] Multiplicaciones: {mult}, Divisiones: {div}, Sumas: {suma}, Restas: {resta}")
    logging.info(f"[GAUSSIANA] Tiempo de ejecución: {tiempo:.6f} segundos")
    return solucion


# ####################################################################
def gauss_jordan(Ab: np.ndarray) -> np.ndarray:
    global _jordan_data
    if not isinstance(Ab, np.ndarray):
        Ab = np.array(Ab, dtype=float)
    assert Ab.shape[0] == Ab.shape[1] - 1
    Ab = Ab.copy()
    n = Ab.shape[0]

    suma = mult = div = resta = 0
    t0 = time.time()

    for i in range(n):
        p = None
        for pi in range(i, n):
            if Ab[pi, i] == 0:
                continue
            if p is None or abs(Ab[pi, i]) < abs(Ab[p, i]):
                p = pi

        if p is None:
            raise ValueError("No existe solución única.")

        if p != i:
            Ab[[i, p]] = Ab[[p, i]]

        for j in range(n):
            if i == j:
                continue
            m = Ab[j, i] / Ab[i, i]
            div += 1
            Ab[j, i:] = Ab[j, i:] - m * Ab[i, i:]
            mult += len(Ab[i, i:])
            resta += len(Ab[i, i:])

    if Ab[n - 1, n - 1] == 0:
        raise ValueError("No existe solución única.")

    solucion = np.zeros(n)
    for i in range(n - 1, -1, -1):
        solucion[i] = Ab[i, -1] / Ab[i, i]
        div += 1

    t1 = time.time()
    total_op = suma + resta + mult + div
    tiempo = t1 - t0

    _jordan_data.append((tiempo, total_op))

    logging.info(f"[GAUSS-JORDAN] Multiplicaciones: {mult}, Divisiones: {div}, Sumas: {suma}, Restas: {resta}")
    logging.info(f"[GAUSS-JORDAN] Tiempo de ejecución: {tiempo:.6f} segundos")
    return solucion


# ####################################################################
def graficar_comparacion():
    if not _gauss_data or not _jordan_data:
        logging.warning("No hay datos suficientes para graficar.")
        return

    tiempos_gauss, ops_gauss = zip(*_gauss_data)
    tiempos_jordan, ops_jordan = zip(*_jordan_data)

    plt.figure(figsize=(10, 6))
    plt.plot(tiempos_gauss, ops_gauss, marker='o', linestyle='-', color='blue', label="Gauss")
    plt.plot(tiempos_jordan, ops_jordan, marker='o', linestyle='-', color='red', label="Jordan")
    plt.title("Tiempo vs Operaciones - Comparación Gauss vs Gauss-Jordan")
    plt.xlabel("Tiempo de ejecución (s)")
    plt.ylabel("Total de operaciones")
    plt.grid(True)
    plt.legend()
    plt.show()
