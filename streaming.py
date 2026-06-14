"""
streaming.py
============

Módulo con las estructuras de datos y algoritmos de minería de flujos
(Data Stream Mining) implementados DESDE CERO para la Parte V del
Proyecto 2 — Data Mining UTEC.

Contiene:
    - CountMinSketch : conteo aproximado de frecuencias (Sección 2).
    - DGIM           : conteo aproximado de 1s en ventana deslizante
                        (Sección 3, técnica adicional).
    - Funciones auxiliares de ventanas deslizantes y métricas por
      ventana (Sección 1).

Todas las clases/funciones están escritas sin depender de librerías
externas de hashing probabilístico ni de implementaciones "listas"
de streaming (solo se usan estructuras estándar: hashlib, math,
collections, numpy, pandas).
"""

from __future__ import annotations

import hashlib
import math
import random
from collections import deque

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 1. VENTANAS DESLIZANTES: estadísticas y métricas por ventana
# ---------------------------------------------------------------------------

def estadisticas_por_ventana(df: pd.DataFrame, freq: str,
                              value_col: str = "stars") -> pd.DataFrame:
    """
    Calcula count, sum, average (y métricas adicionales) de `value_col`
    agrupando el stream en ventanas de tiempo de tamaño `freq`
    (ej: '1h', '4h', '1D').

    Simula un procesador de streams que mantiene acumuladores
    (count, sum, sum_sq, min, max) por ventana y los "emite" (resetea)
    al cerrarse la ventana -- memoria O(1) por ventana activa.

    Returns
    -------
    pd.DataFrame indexado por el inicio de cada ventana, con columnas:
        count, sum, average, std, min, max
    """
    g = df.set_index("date")[value_col].resample(freq)
    out = g.agg(["count", "sum", "mean", "std", "min", "max"])
    out = out.rename(columns={"mean": "average"})
    return out


def metricas_globales_por_ventana(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    """
    Calcula un panel más completo de métricas por ventana de tiempo,
    combinando varias columnas del stream de reseñas:

        - n_reviews       : count de reseñas (volumen / tendencia de popularidad)
        - avg_stars       : promedio de calificación
        - sum_useful      : suma de votos "useful" recibidos
        - pct_useful      : fracción de reseñas con useful > 0
        - avg_text_length : longitud promedio del texto
        - n_business_unicos / n_users_unicos : actividad distinta por ventana

    Estas métricas son las que un sistema de monitoreo en tiempo real
    mantendría por ventana (acumuladores incrementales), sin almacenar
    cada evento individual una vez cerrada la ventana.
    """
    gdf = df.set_index("date")

    out = pd.DataFrame({
        "n_reviews":       gdf["stars"].resample(freq).count(),
        "avg_stars":       gdf["stars"].resample(freq).mean(),
        "sum_useful":      gdf["useful"].resample(freq).sum(),
        "pct_useful":      gdf["useful"].resample(freq).apply(lambda s: (s > 0).mean() if len(s) else np.nan),
        "avg_text_length": gdf["text_length"].resample(freq).mean() if "text_length" in df.columns else np.nan,
        "n_business_unicos": gdf["business_id"].resample(freq).nunique(),
        "n_users_unicos":    gdf["user_id"].resample(freq).nunique(),
    })
    return out


# ---------------------------------------------------------------------------
# 2. COUNT-MIN SKETCH
# ---------------------------------------------------------------------------

class CountMinSketch:
    """
    Count-Min Sketch implementado desde cero (sin librerías externas
    de hashing probabilístico).

    Memoria: O(k * w) contadores, independiente del número de
    elementos distintos del stream.

    Operaciones:
        update(x)   : procesa un elemento del stream (inserción), O(k)
        estimate(x) : f_hat(x) = min_i C[i, h_i(x)],                O(k)

    Garantías teóricas:
        - f_hat(x) >= f(x)                (nunca subestima)
        - f_hat(x) <= f(x) + eps * N       con prob >= 1 - delta
          donde  eps = e / w   y   delta = e^{-k}
    """

    def __init__(self, k: int = 4, w: int = 2000, seed: int = 42):
        self.k = k
        self.w = w
        self.C = np.zeros((k, w), dtype=np.int64)
        self.N = 0  # total de elementos procesados (tamaño del stream)

        # k funciones hash independientes, generadas con "sal" distinta
        # por fila usando md5 (evita depender de hash() de Python, que
        # varía entre ejecuciones por PYTHONHASHSEED)
        rng = random.Random(seed)
        self.salts = [rng.randint(0, 2**32 - 1) for _ in range(k)]

    def _hash(self, x, i: int) -> int:
        """Función hash i-ésima: retorna un índice en [0, w)."""
        s = f"{self.salts[i]}_{x}".encode("utf-8")
        digest = hashlib.md5(s).hexdigest()
        return int(digest, 16) % self.w

    def update(self, x, count: int = 1) -> None:
        """Procesa un elemento del stream (inserción)."""
        for i in range(self.k):
            j = self._hash(x, i)
            self.C[i, j] += count
        self.N += count

    def estimate(self, x) -> int:
        """Consulta: f_hat(x) = min_i C[i, h_i(x)]"""
        return int(min(self.C[i, self._hash(x, i)] for i in range(self.k)))

    def error_bound(self) -> float:
        """Cota de error eps*N con eps = e/w (cota estándar de CMS)."""
        eps = math.e / self.w
        return eps * self.N

    def delta(self) -> float:
        """Probabilidad de exceder la cota de error: delta = e^{-k}."""
        return math.exp(-self.k)

    def memory_bytes(self) -> int:
        """Memoria real usada por la matriz de contadores (bytes)."""
        return self.C.nbytes


# ---------------------------------------------------------------------------
# 3. DGIM (Datar, Gionis, Indyk, Motwani, 2002)
# ---------------------------------------------------------------------------

class DGIM:
    """
    Algoritmo DGIM para conteo aproximado de '1's en los últimos N bits
    de un stream binario, usando memoria O(log^2 N).

    Estructura: lista de 'buckets', cada uno con:
        - size      : número de 1s que representa (potencia de 2)
        - timestamp : posición (tiempo) del 1 más reciente del bucket

    Reglas:
        1. El tamaño de cada bucket es potencia de 2.
        2. A lo sumo 2 buckets de cualquier tamaño dado.
        3. Los buckets están ordenados por timestamp (más antiguo -> más reciente).
    """

    def __init__(self, N: int):
        self.N = N
        self.t = 0
        self.buckets: deque[list[int]] = deque()  # cada bucket: [size, timestamp]

    def add_bit(self, bit: int) -> None:
        self.t += 1

        # 1) Eliminar buckets que ya quedaron fuera de la ventana
        while self.buckets and self.buckets[0][1] <= self.t - self.N:
            self.buckets.popleft()

        if bit == 0:
            return  # caso 1: bit=0, no se crea bucket

        # caso 2: bit=1
        # (a) crear nuevo bucket de tamaño 1
        self.buckets.append([1, self.t])

        # (b)-(d) fusión en cascada si hay 3 buckets del mismo tamaño
        self._merge_cascade()

    def _merge_cascade(self) -> None:
        i = 0
        while i <= len(self.buckets) - 3:
            s0 = self.buckets[i][0]
            s1 = self.buckets[i + 1][0]
            s2 = self.buckets[i + 2][0]
            if s0 == s1 == s2:
                merged_size = s0 + s1
                merged_ts = self.buckets[i + 1][1]
                del self.buckets[i]
                del self.buckets[i]
                self.buckets.insert(i, [merged_size, merged_ts])
                i = max(i - 1, 0)
            else:
                i += 1

    def estimate_count(self) -> float:
        """
        Estima el número de 1s en los últimos N bits:
            suma de todos los buckets excepto el más antiguo
            + 1/2 del tamaño del bucket más antiguo
        """
        if not self.buckets:
            return 0.0
        total = sum(b[0] for b in self.buckets) - self.buckets[0][0]
        total += self.buckets[0][0] / 2
        return total

    def memory_bits(self) -> float:
        """
        Memoria usada: por cada bucket, O(log N) bits para el timestamp
        + O(log log N) bits para el tamaño (se guarda el exponente).
        Total: O(log^2 N).
        """
        log2N = max(1, math.log2(self.N))
        bits_por_bucket = 2 * math.log2(log2N + 1) + math.log2(self.N)
        return len(self.buckets) * bits_por_bucket


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------

def cargar_json_lineas_robusto(ruta_archivo: str) -> pd.DataFrame:
    """
    Carga un archivo JSON-lines tolerando líneas corruptas
    (igual al usado en el EDA de la Parte I).
    """
    import json

    datos = []
    lineas_corruptas = 0
    lineas_totales = 0

    with open(ruta_archivo, "r", encoding="utf-8") as f:
        for linea in f:
            lineas_totales += 1
            linea = linea.strip()
            if not linea:
                continue
            try:
                datos.append(json.loads(linea))
            except json.JSONDecodeError:
                lineas_corruptas += 1
                continue

    if lineas_corruptas > 0:
        print(f" Advertencia: se saltaron {lineas_corruptas} líneas corruptas "
              f"de {lineas_totales} en {ruta_archivo}.")

    return pd.DataFrame(datos)
