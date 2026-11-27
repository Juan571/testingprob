# ================================================================
# V8 - PREDICTOR DE ANIMALES (logit condicional + pruebas de hipótesis)
# ================================================================
#
# QUÉ HACE
#
# 1. Lee resultados.csv (La Granjita, Lotto Activo e Internacional), limpia y quita
#    duplicados.
# 2. Contrasta hipótesis sobre los sorteos (¿se cumplen o no?):
#       - uniformidad y "rachas" por periodos
#       - repetición / anti-repetición y curva de riesgo
#       - pares de animales que salen (o se evitan) el mismo día
#       - transiciones (qué animal sigue a cuál), paso numérico ±1
#       - efectos entre loterías
#       - hora del día y día de la semana
#    con corrección por comparaciones múltiples y replicación
#    (primera mitad del histórico frente a la segunda).
# 3. Ajusta un LOGIT CONDICIONAL (modelo de elección discreta de
#    McFadden) por lotería con los patrones que SÍ se cumplen:
#       P(animal a | pasado) = exp(x_a · β) / Σ_b exp(x_b · β)
# 4. Backtest walk-forward de 365 días SIN información futura y
#    comparación contra el azar (IC 95 %, prueba z) y entre modelos
#    anidados (qué aporta cada patrón).
# 5. Predice el próximo sorteo (probabilidades calibradas) y, por
#    Monte Carlo, la probabilidad de aparecer en los próximos 11 turnos.
# 6. Genera CSVs y el dashboard HTML (v8_output/dashboard.html).
#
# Solo requiere numpy y pandas.
#
# CSV esperado:
#
#     date,lottery,number,animal,time
#     2026-08-26,lottoactivo,11,gato,04:00 PM
#
# ================================================================

import os
import sys
import math
import time
import html as _html
import json
import warnings
from collections import defaultdict
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ================================================================
# CONFIGURACIÓN
# ================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(BASE_DIR, "resultados.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "v8_output")

LOTTERIES = ["lagranjita", "lottoactivo", "lottoactivoint"]
LOTTERY_LABEL = {
    "lagranjita": "La Granjita",
    "lottoactivo": "Lotto Activo",
    "lottoactivoint": "Lotto Activo Int",
}

# Otras loterías que usa el modelo de cada una, en orden: [hermana, tercera].
# La "hermana" es la que más la influye (Lotto Activo <-> La Granjita;
# Lotto Activo Int, que sortea a las hh:30, con Lotto Activo).
OTHERS = {
    "lagranjita": ["lottoactivo", "lottoactivoint"],
    "lottoactivo": ["lagranjita", "lottoactivoint"],
    "lottoactivoint": ["lottoactivo", "lagranjita"],
}

# Loterías que sortean a la misma hora (métrica "consolidada" por turno).
SAME_TURN = ["lagranjita", "lottoactivo"]

# Loterías con las que se estiman las matrices de pares y transiciones.
# Lotto Activo Int comparte menos esa estructura (correlación de pares ~0,2
# frente a ~0,4 entre Granjita y Activo) y añadirla empeora el ajuste.
PAIR_LOTTERIES = ["lagranjita", "lottoactivo"]

# Pago por acierto: el premio es 30 veces lo apostado.
PAYOUT = {"lagranjita": 30.0, "lottoactivo": 30.0, "lottoactivoint": 30.0}

N_ANIMALS = 38
CHANCE = 1.0 / N_ANIMALS

# Rankings mostrados y evaluados
TOP_N = 10
TOP_PER_LOTTERY = 7
TOP_KS = (1, 3, 5, 10)

# Modelo (logit condicional)
HALF_LIFE_DAYS = 180      # peso de un sorteo: 0.5 ** (antigüedad_días / 180)
MIN_TRAIN_WEIGHT = 0.01   # sorteos con menos peso no se usan (velocidad)
L2_PENALTY = 3.0          # regularización ridge de los coeficientes
PAIR_SHRINK = 30.0        # contracción de la matriz de pares hacia 0
TRANSITION_SHRINK = 40.0  # contracción de la matriz de transiciones hacia 0
MIN_HISTORY_DRAWS = 300   # calentamiento: sorteos previos mínimos por lotería

# Backtest walk-forward (sin información futura)
EVAL_DAYS = 365           # ventana evaluada fuera de muestra
REFIT_DAYS = 14           # cada cuántos días se re-entrena el modelo
ABLATION_REFIT_DAYS = 91  # re-entrenamiento de la comparación de modelos

# Monte Carlo (próximos 11 turnos)
ELEVEN_TURNS = 11
ELEVEN_TOP = 5
ELEVEN_HISTORY_STARTS = 30
N_SIMULATIONS = 4000
N_SIMULATIONS_HISTORY = 1000
RANDOM_SEED = 12345

# Cubetas de "antigüedad" (sorteos desde la última aparición)
OWN_AGE_EDGES = [1, 2, 3, 4, 6, 9, 13, 19, 26, 36, 51, 76, 101, 151]
OWN_AGE_REF = 26          # cubeta de referencia [26-35] (coeficiente 0)
OTHER_AGE_EDGES = [1, 2, 3, 4, 6, 9, 13, 26, 51, 101]
OTHER_AGE_REF = 26        # cubeta de referencia [26-50]
OWN_PREV = 12             # sorteos previos de la misma lotería (pares)
OTHER_PREV = 6            # sorteos previos de la otra lotería (pares)
COUNT_WINDOW = 152        # ventana para el conteo reciente (~2 semanas)
NEVER_SEEN_AGE = 100000


# ================================================================
# TABLA OFICIAL (animal <-> número)
# ================================================================

ANIMAL_NUM = {
    "delfin": 0, "carnero": 1, "toro": 2, "ciempies": 3, "alacran": 4,
    "leon": 5, "rana": 6, "perico": 7, "raton": 8, "aguila": 9,
    "tigre": 10, "gato": 11, "caballo": 12, "mono": 13, "paloma": 14,
    "zorro": 15, "oso": 16, "pavo": 17, "burro": 18, "chivo": 19,
    "cochino": 20, "gallo": 21, "camello": 22, "cebra": 23, "iguana": 24,
    "gallina": 25, "vaca": 26, "perro": 27, "zamuro": 28, "elefante": 29,
    "caiman": 30, "lapa": 31, "ardilla": 32, "pescado": 33, "venado": 34,
    "jirafa": 35, "culebra": 36, "ballena": 37,
}

NUM_ANIMAL = {v: k for k, v in ANIMAL_NUM.items()}

ANIMAL_ALIASES = {"zebra": "cebra", "loro": "perico"}


def animal_label(animal):
    """'toro (2)'. La ballena se muestra con su número oficial '00'."""

    num = ANIMAL_NUM.get(animal)

    if num is None:
        return str(animal)

    return f"{animal} ({'00' if num == 37 else num})"


# ================================================================
# DISTRIBUCIONES (sin scipy)
# ================================================================

def norm_sf(z):
    """P(Z > z) para Z ~ N(0, 1)."""
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def norm_two_sided(z):
    return min(1.0, 2.0 * norm_sf(abs(z)))


def _gamma_series(a, x):
    """P(a, x) regularizada por serie (x < a + 1)."""
    term = 1.0 / a
    total = term
    ap = a
    for _ in range(100000):
        ap += 1.0
        term *= x / ap
        total += term
        if abs(term) < abs(total) * 1e-15:
            break
    return total * math.exp(-x + a * math.log(x) - math.lgamma(a))


def _gamma_cont_frac(a, x):
    """Q(a, x) regularizada por fracción continua (x >= a + 1)."""
    tiny = 1e-300
    b = x + 1.0 - a
    c = 1.0 / tiny
    d = 1.0 / b
    h = d
    for i in range(1, 100000):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        d = tiny if abs(d) < tiny else d
        c = b + an / c
        c = tiny if abs(c) < tiny else c
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-15:
            break
    return math.exp(-x + a * math.log(x) - math.lgamma(a)) * h


def chi2_sf(x, k):
    """P(X > x) para X ~ chi-cuadrado(k)."""
    if x <= 0:
        return 1.0
    a, y = k / 2.0, x / 2.0
    if y < a + 1.0:
        return max(0.0, 1.0 - _gamma_series(a, y))
    return _gamma_cont_frac(a, y)


def chi2_cdf(x, k):
    """P(X <= x) para X ~ chi-cuadrado(k) (cola izquierda precisa)."""
    if x <= 0:
        return 0.0
    a, y = k / 2.0, x / 2.0
    if y < a + 1.0:
        return _gamma_series(a, y)
    return max(0.0, 1.0 - _gamma_cont_frac(a, y))


def format_p(p):
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return "—"
    if p < 1e-16:
        return "< 1e-16"
    if p < 0.001:
        return f"{p:.1e}"
    return f"{p:.3f}"


# ================================================================
# CARGA DE DATOS
# ================================================================

def load_data(path=INPUT_FILE):

    if not os.path.exists(path):
        raise FileNotFoundError(f"No se encontró {path}")

    df = pd.read_csv(path)
    df.columns = [str(c).strip().lower() for c in df.columns]

    missing = [c for c in ("date", "lottery", "animal", "time") if c not in df.columns]

    if missing:
        raise ValueError("Faltan columnas en resultados.csv: " + ", ".join(missing))

    df["lottery"] = df["lottery"].astype(str).str.strip().str.lower()
    df["animal"] = (
        df["animal"].astype(str).str.strip().str.lower().replace(ANIMAL_ALIASES)
    )

    unknown = sorted(set(df["animal"]) - set(ANIMAL_NUM))

    if unknown:
        print(f"Aviso: se ignoran animales desconocidos: {unknown}")

    df = df[df["animal"].isin(ANIMAL_NUM) & df["lottery"].isin(LOTTERIES)].copy()

    df["number"] = df["animal"].map(ANIMAL_NUM).astype(int)

    time_txt = df["time"].astype(str).str.strip()
    parsed = pd.to_datetime(time_txt, format="%I:%M %p", errors="coerce")
    retry = parsed.isna()

    if retry.any():
        parsed.loc[retry] = pd.to_datetime(time_txt[retry], format="%H:%M", errors="coerce")

    df["hour"] = parsed.dt.strftime("%H:%M")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["datetime"] = pd.to_datetime(
        df["date"].dt.strftime("%Y-%m-%d") + " " + df["hour"], errors="coerce"
    )

    df = df.dropna(subset=["date", "datetime", "hour"])

    # Un único resultado por (turno, lotería): el último publicado gana.
    before = len(df)
    df = df.drop_duplicates(subset=["datetime", "lottery"], keep="last")

    if len(df) < before:
        print(f"Aviso: se eliminaron {before - len(df)} resultados duplicados.")

    return df.sort_values(["datetime", "lottery"]).reset_index(drop=True)


def build_turn_table(df):
    """Tabla turno x lotería con el número ganador (NaN si no hubo sorteo)."""
    return (
        df.pivot_table(index="datetime", columns="lottery", values="number", aggfunc="last")
        .reindex(columns=LOTTERIES)
        .sort_index()
    )


# ================================================================
# ESTADO DEL SORTEO
# ================================================================
#
# El estado resume TODO lo que el modelo puede saber antes de un
# sorteo. Tiene una primera dimensión S (simulaciones); el recorrido
# histórico usa S = 1 y el Monte Carlo usa S = miles.
#
# ================================================================

class DrawState:

    def __init__(self, S=1):
        self.S = S
        self.age = {L: np.full((S, N_ANIMALS), NEVER_SEEN_AGE, np.int64) for L in LOTTERIES}
        self.prev = {L: np.full((S, OWN_PREV), -1, np.int64) for L in LOTTERIES}
        self.today = {L: np.zeros((S, N_ANIMALS), bool) for L in LOTTERIES}
        self.count = {L: np.zeros((S, N_ANIMALS), np.int64) for L in LOTTERIES}
        self.ring = {L: np.full((S, COUNT_WINDOW), -1, np.int64) for L in LOTTERIES}
        self.ring_pos = {L: 0 for L in LOTTERIES}
        self.n_draws = {L: 0 for L in LOTTERIES}
        self.current_date = None

    def copy_tiled(self, S):
        """Replica un estado S=1 en S simulaciones independientes."""
        new = DrawState(S)
        for L in LOTTERIES:
            new.age[L] = np.repeat(self.age[L], S, axis=0)
            new.prev[L] = np.repeat(self.prev[L], S, axis=0)
            new.today[L] = np.repeat(self.today[L], S, axis=0)
            new.count[L] = np.repeat(self.count[L], S, axis=0)
            new.ring[L] = np.repeat(self.ring[L], S, axis=0)
            new.ring_pos[L] = self.ring_pos[L]
            new.n_draws[L] = self.n_draws[L]
        new.current_date = self.current_date
        return new

    def start_turn(self, date):
        """Al cambiar de día se reinician los indicadores de 'hoy'."""
        if date != self.current_date:
            for L in LOTTERIES:
                self.today[L][:] = False
            self.current_date = date

    def inputs(self, L):
        """
        Entradas del modelo para predecir el próximo sorteo de L.
        o0_* = lotería hermana, o1_* = tercera lotería (ver OTHERS).
        """
        inp = {
            "own_age": self.age[L],
            "own_today": self.today[L],
            "own_prev": self.prev[L][:, :OWN_PREV],
            "count": self.count[L],
        }
        for j, O in enumerate(OTHERS[L]):
            inp[f"o{j}_age"] = self.age[O]
            inp[f"o{j}_today"] = self.today[O]
            inp[f"o{j}_prev"] = self.prev[O][:, :OTHER_PREV]
        return inp

    def update(self, L, winners):
        """Registra los ganadores (array de tamaño S) de la lotería L."""
        rows = np.arange(self.S)
        winners = np.asarray(winners, np.int64)
        self.age[L] += 1
        self.age[L][rows, winners] = 1
        self.prev[L][:, 1:] = self.prev[L][:, :-1]
        self.prev[L][:, 0] = winners
        self.today[L][rows, winners] = True
        pos = self.ring_pos[L]
        old = self.ring[L][:, pos]
        has_old = old >= 0
        self.count[L][rows[has_old], old[has_old]] -= 1
        self.count[L][rows, winners] += 1
        self.ring[L][:, pos] = winners
        self.ring_pos[L] = (pos + 1) % COUNT_WINDOW
        self.n_draws[L] += 1


def replay_history(table, snapshot_turns=()):
    """
    Recorre todos los turnos en orden. Para cada sorteo de cada lotería
    guarda las entradas del modelo ANTES del sorteo (sin fuga de datos).

    Devuelve:
        rows[L]   -> dict de arrays (un elemento por sorteo de L)
        state     -> estado final (tras el último turno conocido)
        snapshots -> {momento: copia del estado justo antes de ese momento}
    """

    state = DrawState(1)
    pending = sorted(set(snapshot_turns))
    snapshots = {}

    acc = {L: defaultdict(list) for L in LOTTERIES}

    values = table.to_numpy()
    turns = table.index

    for i, ts in enumerate(turns):

        while pending and pending[0] <= ts:
            snap = state.copy_tiled(1)
            snap.start_turn(pending[0].normalize())
            snapshots[pending.pop(0)] = snap

        state.start_turn(ts.normalize())

        winners = {}

        for j, L in enumerate(LOTTERIES):
            v = values[i, j]
            if np.isnan(v):
                continue
            winners[L] = int(v)
            if state.n_draws[L] >= MIN_HISTORY_DRAWS:
                inp = state.inputs(L)
                a = acc[L]
                for key, arr in inp.items():
                    a[key].append(arr[0].copy())
                a["y"].append(int(v))
                a["datetime"].append(ts)

        # Los sorteos del mismo turno son simultáneos: se actualiza el
        # estado DESPUÉS de leer las entradas de todas las loterías.
        for L, w in winners.items():
            state.update(L, [w])

    for ts in pending:
        snap = state.copy_tiled(1)
        snap.start_turn(ts.normalize())
        snapshots[ts] = snap

    rows = {}

    for L in LOTTERIES:
        a = acc[L]
        rows[L] = {k: np.array(v) for k, v in a.items()}
        rows[L]["datetime"] = pd.DatetimeIndex(a["datetime"])

    return rows, state, snapshots


def subset_rows(rows, mask):
    return {k: v[mask] for k, v in rows.items()}


# ================================================================
# MATRICES DE PARES Y DE TRANSICIONES (aprendidas de los datos)
# ================================================================

class PairData:

    """
    Dos matrices aprendidas de los datos:

    W (pares del mismo día)   W[a, b] = log((C_ab + λ) / (E_ab + λ))
        C_ab = días (por lotería) en que salieron a y b
        E_ab = n_a · n_b / D   (esperado si los días fueran independientes)

    T (transiciones)          T[a, b] = log((M_ab + λ) / (E_ab + λ))
        M_ab = veces que b salió justo después de a (o a después de b)
        E_ab = esperado según la frecuencia de cada animal

    Valores < 0 -> la pareja se "evita"; > 0 -> se "atrae". λ contrae
    hacia 0 las parejas con poca evidencia.
    """

    def __init__(self, df, lotteries=PAIR_LOTTERIES):
        df = df[df["lottery"].isin(lotteries)]
        days = (
            df.groupby(["lottery", df["datetime"].dt.normalize()])["number"]
            .apply(lambda s: np.bincount(s.to_numpy(), minlength=N_ANIMALS) > 0)
        )
        self.dates = pd.DatetimeIndex(days.index.get_level_values(1))
        self.X = np.vstack(days.to_numpy()).astype(np.float64)
        self.years = self.dates.year.to_numpy()

        src, dst, when = [], [], []
        for L in lotteries:
            g = df[df["lottery"] == L].sort_values("datetime")
            s = g["number"].to_numpy()
            src.append(s[:-1])
            dst.append(s[1:])
            when.append(g["datetime"].to_numpy()[1:])
        self.t_src = np.concatenate(src)
        self.t_dst = np.concatenate(dst)
        self.t_dates = pd.DatetimeIndex(np.concatenate(when))
        self.t_years = self.t_dates.year.to_numpy()

    def counts(self, mask):
        X = self.X[mask]
        D = len(X)
        if D == 0:
            return np.zeros((N_ANIMALS, N_ANIMALS)), np.zeros((N_ANIMALS, N_ANIMALS)), 0
        n = X.sum(0)
        return X.T @ X, np.outer(n, n) / D, D

    def pair_matrix(self, mask, lam=PAIR_SHRINK):
        """W a partir de los días seleccionados (cada año se agrega aparte)."""
        C = np.zeros((N_ANIMALS, N_ANIMALS))
        E = np.zeros((N_ANIMALS, N_ANIMALS))
        for y in np.unique(self.years[mask]):
            c, e, _ = self.counts(mask & (self.years == y))
            C += c
            E += e
        W = np.log((C + lam) / (E + lam))
        np.fill_diagonal(W, 0.0)
        return W

    def transition_matrix(self, mask, lam=TRANSITION_SHRINK):
        """T simétrica a partir de las transiciones seleccionadas."""
        M = np.zeros((N_ANIMALS, N_ANIMALS))
        np.add.at(M, (self.t_src[mask], self.t_dst[mask]), 1)
        M = M + M.T
        r = M.sum(1)
        E = np.outer(r, r) / max(M.sum(), 1.0)
        T = np.log((M + lam) / (E + lam))
        np.fill_diagonal(T, 0.0)
        return T

    def matrices(self, day_mask, trans_mask):
        return {"W": self.pair_matrix(day_mask), "T": self.transition_matrix(trans_mask)}

    def cross_fitted(self, cutoff):
        """
        Matrices para entrenar con datos < cutoff:
            por_año[y] -> sin el año y (para las filas de entrenamiento de y)
            completa   -> con todo lo anterior a cutoff (para predecir)
        Así los coeficientes de pares/transiciones no se inflan por
        sobreajuste (validación cruzada por años).
        """
        d_before = np.asarray(self.dates < cutoff)
        t_before = np.asarray(self.t_dates < cutoff)
        full = self.matrices(d_before, t_before)
        by_year = {
            int(y): self.matrices(d_before & (self.years != y), t_before & (self.t_years != y))
            for y in np.unique(self.years[d_before])
        }
        return by_year, full


# ================================================================
# VARIABLES DEL MODELO
# ================================================================

def _bucket_names(edges, ref, prefix):
    names = []
    for i, lo in enumerate(edges):
        if lo == ref:
            continue
        hi = edges[i + 1] - 1 if i + 1 < len(edges) else None
        if hi is None:
            label = f"{lo}+"
        elif hi == lo:
            label = f"{lo}"
        else:
            label = f"{lo}-{hi}"
        names.append(f"{prefix}_{label}")
    return names


# Grupos de variables. "o0" = lotería hermana y "o1" = tercera lotería de
# la lotería que se predice (ver OTHERS). El orden de GROUP_ORDER es el
# orden de las columnas de X.
FEATURE_GROUPS = {
    "own_age": _bucket_names(OWN_AGE_EDGES, OWN_AGE_REF, "edad_propia"),
    "own_today": ["hoy_propia"],
    "own_pairs": ["par_propia_lag1", "par_propia_lag2_12"],
    "trans": ["transicion_lag1"],
    "count": ["conteo_152"],
    "o0_age": _bucket_names(OTHER_AGE_EDGES, OTHER_AGE_REF, "edad_o0") + ["hoy_o0"],
    "o0_pairs": ["par_o0_lag1", "par_o0_lag2_6"],
    "o1_age": _bucket_names(OTHER_AGE_EDGES, OTHER_AGE_REF, "edad_o1") + ["hoy_o1"],
    "o1_pairs": ["par_o1_lag1", "par_o1_lag2_6"],
}

GROUP_ORDER = list(FEATURE_GROUPS)

# Variables continuas (afinidad aprendida): su efecto se explica con un ejemplo.
CONTINUOUS_FEATURES = {
    "par_propia_lag1": "W", "par_propia_lag2_12": "W",
    "par_o0_lag1": "W", "par_o0_lag2_6": "W",
    "par_o1_lag1": "W", "par_o1_lag2_6": "W",
    "transicion_lag1": "T", "conteo_152": None,
}


def _age_text(name):
    k = name.split("_")[-1]
    return f"{k} sorteo" if k == "1" else f"{k} sorteos"


def feature_description(name, L):
    """Descripción legible de una variable para el modelo de la lotería L."""
    if name.startswith("edad_propia"):
        return f"Salió hace {_age_text(name)} en {LOTTERY_LABEL[L]} (ref. 26-35)"
    for j, O in enumerate(OTHERS[L]):
        lab = LOTTERY_LABEL[O]
        if name.startswith(f"edad_o{j}"):
            return f"Salió hace {_age_text(name)} en {lab} (ref. 26-50)"
        if name == f"hoy_o{j}":
            return f"Ya salió hoy en {lab}"
        if name == f"par_o{j}_lag1":
            return f"Afinidad de par con el último animal de {lab}"
        if name == f"par_o{j}_lag2_6":
            return f"Afinidad de par con los 5 sorteos anteriores de {lab}"
    return {
        "hoy_propia": f"Ya salió hoy en {LOTTERY_LABEL[L]}",
        "par_propia_lag1": "Afinidad de par con el último animal (misma lotería)",
        "par_propia_lag2_12": "Afinidad de par con los 11 sorteos anteriores (misma lotería)",
        "transicion_lag1": "Afinidad de transición con el último animal (misma lotería)",
        "conteo_152": "Veces que salió en los últimos 152 sorteos (centrado en 4)",
    }.get(name, name)


def feature_label(name, L):
    """Nombre de la variable con la lotería real en lugar de o0/o1."""
    for j, O in enumerate(OTHERS[L]):
        name = name.replace(f"o{j}", O)
    return name


# Modelos anidados para medir qué aporta cada patrón
MODEL_SPECS = {
    "anti_repeticion": ["own_age", "own_today"],
    "mas_hermana": ["own_age", "own_today", "o0_age"],
    "mas_pares": ["own_age", "own_today", "o0_age", "own_pairs", "o0_pairs"],
    "mas_transiciones": ["own_age", "own_today", "o0_age", "own_pairs", "o0_pairs", "trans"],
    "mas_tercera": ["own_age", "own_today", "o0_age", "own_pairs", "o0_pairs", "trans",
                    "o1_age", "o1_pairs"],
    "mas_conteo": ["own_age", "own_today", "o0_age", "own_pairs", "o0_pairs", "trans",
                   "o1_age", "o1_pairs", "count"],
}

MODEL_LABELS = {
    "azar": "Azar (elegir al azar)",
    "anti_repeticion": "Solo anti-repetición",
    "mas_hermana": "+ lotería hermana (Activo ↔ Granjita, Int ↔ Activo)",
    "mas_pares": "+ pares que se evitan el mismo día",
    "mas_transiciones": "+ transiciones (qué sigue a qué)",
    "mas_tercera": "+ tercera lotería",
    "mas_conteo": "+ conteo reciente (probado, no mejora)",
}

# Modelo de producción (el conteo reciente no mejora fuera de muestra).
FINAL_MODEL = "mas_tercera"
FINAL_GROUPS = MODEL_SPECS[FINAL_MODEL]


def _one_hot_ages(age, edges, ref):
    b = np.searchsorted(edges, age, side="right") - 1
    keep = [i for i, lo in enumerate(edges) if lo != ref]
    return np.eye(len(edges), dtype=np.float64)[b][..., keep]


def _pair_sum(W, prev):
    """Suma de W[prev_k, :] sobre las posiciones válidas (prev >= 0)."""
    Wp = np.vstack([W, np.zeros((1, N_ANIMALS))])
    idx = np.where(prev < 0, N_ANIMALS, prev)
    return Wp[idx].sum(axis=1)


def make_features(inp, mats, groups=None):
    """
    inp:  entradas de DrawState (o filas del histórico), con primera
          dimensión n.
    mats: {"W": matriz de pares, "T": matriz de transiciones} (38 x 38).
    Devuelve X con forma (n, 38, F).
    """

    groups = set(groups or GROUP_ORDER)
    W = mats["W"]
    parts = []

    def col(x):
        return np.asarray(x, dtype=np.float64)[..., None]

    if "own_age" in groups:
        parts.append(_one_hot_ages(inp["own_age"], OWN_AGE_EDGES, OWN_AGE_REF))

    if "own_today" in groups:
        parts.append(col(inp["own_today"]))

    if "own_pairs" in groups:
        parts.append(col(_pair_sum(W, inp["own_prev"][:, :1])))
        parts.append(col(_pair_sum(W, inp["own_prev"][:, 1:])))

    if "trans" in groups:
        parts.append(col(_pair_sum(mats["T"], inp["own_prev"][:, :1])))

    if "count" in groups:
        parts.append(col((inp["count"] - 4.0) / 4.0))

    for j in (0, 1):
        if f"o{j}_age" in groups:
            parts.append(_one_hot_ages(inp[f"o{j}_age"], OTHER_AGE_EDGES, OTHER_AGE_REF))
            parts.append(col(inp[f"o{j}_today"]))
        if f"o{j}_pairs" in groups:
            parts.append(col(_pair_sum(W, inp[f"o{j}_prev"][:, :1])))
            parts.append(col(_pair_sum(W, inp[f"o{j}_prev"][:, 1:])))

    return np.concatenate(parts, axis=2)


def feature_names(groups):
    """Nombres de las columnas de X, en el mismo orden que make_features."""
    return [n for g in GROUP_ORDER if g in groups for n in FEATURE_GROUPS[g]]


def make_features_by_year(rows, mats_by_year, mats_default, groups):
    """Variables de filas históricas usando las matrices cruzadas de su año."""
    years = rows["datetime"].year.to_numpy()
    X = None
    for y in np.unique(years):
        m = years == y
        mats = mats_by_year.get(int(y), mats_default)
        Xy = make_features(subset_rows(rows, m), mats, groups)
        if X is None:
            X = np.empty((len(years),) + Xy.shape[1:])
        X[m] = Xy
    return X


# ================================================================
# LOGIT CONDICIONAL (McFadden) - Newton-Raphson
# ================================================================
#
#   P(gana a | x) = exp(x_a · β) / Σ_b exp(x_b · β)
#
# Se maximiza la log-verosimilitud ponderada con penalización L2.
# Es un problema cóncavo: Newton converge en pocas iteraciones.
#
# ================================================================

def _softmax_rows(s):
    s = s - s.max(axis=1, keepdims=True)
    p = np.exp(s)
    return p / p.sum(axis=1, keepdims=True)


def _objective(beta, X, y, w, l2):
    n, A, F = X.shape
    p = _softmax_rows(X @ beta)
    idx = np.arange(n)
    ll = float((w * np.log(p[idx, y] + 1e-300)).sum())
    xbar = np.einsum("na,naf->nf", p, X)
    grad = (w[:, None] * (X[idx, y] - xbar)).sum(0)
    Z = (X * np.sqrt(w[:, None, None] * p[..., None])).reshape(n * A, F)
    hess = Z.T @ Z - (w[:, None] * xbar).T @ xbar
    f = -ll + 0.5 * l2 * float(beta @ beta)
    return f, -grad + l2 * beta, hess + l2 * np.eye(F), ll


def fit_conditional_logit(X, y, w, l2=L2_PENALTY, beta0=None, max_iter=50):
    F = X.shape[2]
    beta = np.zeros(F) if beta0 is None or len(beta0) != F else beta0.copy()
    f, g, H, ll = _objective(beta, X, y, w, l2)
    for _ in range(max_iter):
        step = np.linalg.solve(H, g)
        t = 1.0
        while True:
            cand = beta - t * step
            f2, g2, H2, ll2 = _objective(cand, X, y, w, l2)
            if f2 <= f + 1e-12 or t < 1e-6:
                break
            t *= 0.5
        converged = abs(f - f2) <= 1e-9 * max(1.0, abs(f))
        beta, f, g, H, ll = cand, f2, g2, H2, ll2
        if converged:
            break
    se = np.sqrt(np.clip(np.diag(np.linalg.inv(H)), 0, None))
    return beta, se, ll


def predict_proba(beta, X):
    return _softmax_rows(X @ beta)


class FittedModel:
    """Modelo ajustado para una lotería con datos anteriores a `cutoff`."""

    def __init__(self, lottery, cutoff, groups, beta, se, mats, n_train, ll):
        self.lottery = lottery
        self.cutoff = cutoff
        self.groups = groups
        self.beta = beta
        self.se = se
        self.mats = mats
        self.n_train = n_train
        self.loglik = ll

    def proba(self, inp):
        return predict_proba(self.beta, make_features(inp, self.mats, self.groups))


def train_weights(times, cutoff):
    age_days = np.asarray((cutoff - times).total_seconds(), dtype=float) / 86400.0
    return 0.5 ** (age_days / HALF_LIFE_DAYS)


def fit_model(rows, pairs, lottery, cutoff, groups, beta0=None):
    """Entrena con las filas de `lottery` estrictamente anteriores a cutoff."""
    r = rows[lottery]
    w = train_weights(r["datetime"], cutoff)
    m = (r["datetime"] < cutoff) & (w >= MIN_TRAIN_WEIGHT)
    tr = subset_rows(r, m)
    mats_by_year, mats_full = pairs.cross_fitted(cutoff)
    X = make_features_by_year(tr, mats_by_year, mats_full, groups)
    beta, se, ll = fit_conditional_logit(X, tr["y"], w[m], beta0=beta0)
    return FittedModel(lottery, cutoff, groups, beta, se, mats_full, int(m.sum()), ll)


# ================================================================
# BACKTEST WALK-FORWARD
# ================================================================


# Las fechas de re-entrenamiento siguen una rejilla fija (cada `refit_days`
# desde este lunes), así una predicción ya hecha no cambia al pasar los días.
CUTOFF_EPOCH = pd.Timestamp("2021-01-04")


def refit_cutoffs(last_time, eval_days, refit_days):
    """
    Fechas de re-entrenamiento (medianoche) que cubren los últimos
    eval_days. Devuelve (cutoffs, inicio_de_la_evaluación, fin).
    """
    end = last_time.normalize() + timedelta(days=1)
    start = end - timedelta(days=eval_days)
    k = (start - CUTOFF_EPOCH).days // refit_days
    c = CUTOFF_EPOCH + timedelta(days=k * refit_days)
    cutoffs = []
    while c < end:
        cutoffs.append(c)
        c += timedelta(days=refit_days)
    return cutoffs, start, end


def walk_forward(rows, pairs, last_time, groups, eval_days, refit_days, keep_models=False):
    """
    Para cada bloque [c, c + refit_days):
        1. entrena SOLO con sorteos anteriores a c
        2. predice todos los sorteos del bloque
    Devuelve un DataFrame con una fila por (sorteo, lotería) evaluado y,
    opcionalmente, los modelos de cada bloque.
    """

    cutoffs, start, end = refit_cutoffs(last_time, eval_days, refit_days)
    bounds = cutoffs[1:] + [end + timedelta(days=1)]

    out = []
    models = {}

    for L in LOTTERIES:
        r = rows[L]
        beta = None
        for c, nxt in zip(cutoffs, bounds):
            test = (r["datetime"] >= max(c, start)) & (r["datetime"] < nxt)
            if not test.any():
                continue
            model = fit_model(rows, pairs, L, c, groups, beta0=beta)
            beta = model.beta
            te = subset_rows(r, test)
            P = model.proba(te)
            if keep_models:
                models[(L, c)] = model
            for i in range(len(te["y"])):
                out.append((te["datetime"][i], L, int(te["y"][i]), P[i], c))

    bt = pd.DataFrame(out, columns=["datetime", "lottery", "winner", "probs", "cutoff"])
    bt = bt.sort_values(["datetime", "lottery"]).reset_index(drop=True)
    return (bt, models) if keep_models else bt


def add_rank_columns(bt):
    """
    Rango del ganador y aciertos top-K. Si el ganador empata en
    probabilidad con otros animales, el acierto se cuenta como la
    probabilidad de que un desempate al azar lo deje dentro del top-K.
    """
    P = np.vstack(bt["probs"].to_numpy())
    y = bt["winner"].to_numpy()
    py = P[np.arange(len(y)), y]
    tol = 1e-12
    greater = (P > py[:, None] + tol).sum(1)
    tied = (np.abs(P - py[:, None]) <= tol).sum(1)
    bt = bt.copy()
    bt["p_winner"] = py
    bt["rank"] = greater + 1
    for K in TOP_KS:
        bt[f"hit_top{K}"] = np.clip((K - greater) / tied, 0.0, 1.0)
    order = np.argsort(-P, axis=1)
    bt["top10"] = [",".join(NUM_ANIMAL[a] for a in o[:TOP_N]) for o in order]
    bt["top1"] = [NUM_ANIMAL[o[0]] for o in order]
    bt["top1_probability"] = P[np.arange(len(y)), order[:, 0]]
    bt["real_animal"] = [NUM_ANIMAL[a] for a in y]
    return bt


def binom_ci(k, n, z=1.96):
    """Intervalo de Wilson para una proporción."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return centre - half, centre + half


def topk_summary(bt):
    """Tasa de acierto top-K por lotería frente al azar (K/38), con IC y prueba z."""
    rows_out = []
    for L in LOTTERIES + ["todas"]:
        sub = bt if L == "todas" else bt[bt["lottery"] == L]
        n = len(sub)
        if n == 0:
            continue
        for K in TOP_KS:
            k = float(sub[f"hit_top{K}"].sum())
            rate = k / n
            base = K / N_ANIMALS
            lo, hi = binom_ci(k, n)
            z = (rate - base) / math.sqrt(base * (1 - base) / n)
            rows_out.append({
                "lottery": L, "K": K, "n": n, "hits": k, "rate": rate,
                "ci_low": lo, "ci_high": hi, "chance": base, "lift": rate / base,
                "z": z, "p_value": norm_sf(z),
            })
    return pd.DataFrame(rows_out)


def info_gain_bits(bt):
    """Ganancia media de información frente al azar (bits por sorteo)."""
    return float(np.mean(np.log2(bt["p_winner"].to_numpy() * N_ANIMALS)))


def chance_hit_any(n_picks, n_real):
    """P(al menos uno de n_picks animales elegidos al azar esté entre n_real ganadores distintos)."""
    if n_real <= 0:
        return 0.0
    return 1.0 - math.comb(N_ANIMALS - n_real, n_picks) / math.comb(N_ANIMALS, n_picks)


def union_probability(prob_list):
    """P(el animal sale en al menos una lotería) = 1 - Π (1 - p_L)."""
    q = np.ones(N_ANIMALS)
    for p in prob_list:
        q *= 1.0 - p
    return 1.0 - q


def consolidated_backtest(bt):
    """
    Métrica por TURNO de La Granjita + Lotto Activo (sortean a la misma
    hora), como el dashboard original:
        - top10: los 10 animales con mayor P(salir en al menos una lotería)
        - top7x2: 7 mejores de cada lotería, sin repetir (hasta 14 animales)
    Se compara con la probabilidad exacta de acertar eligiendo al azar
    la misma cantidad de animales.
    """
    out = []
    for ts, g in bt[bt["lottery"].isin(SAME_TURN)].groupby("datetime"):
        probs = {r.lottery: r.probs for r in g.itertuples()}
        real = sorted({NUM_ANIMAL[w] for w in g["winner"]})
        q = union_probability(probs.values())
        top10 = [NUM_ANIMAL[a] for a in np.argsort(-q)[:TOP_N]]
        merged = []
        for L in LOTTERIES:
            if L not in probs:
                continue
            taken = 0
            for a in np.argsort(-probs[L]):
                name = NUM_ANIMAL[a]
                if name in merged:
                    continue
                merged.append(name)
                taken += 1
                if taken >= TOP_PER_LOTTERY:
                    break
        hits10 = [a for a in top10 if a in real]
        hits7 = [a for a in merged if a in real]
        out.append({
            "datetime": ts,
            "date": ts.strftime("%Y-%m-%d"),
            "time": ts.strftime("%H:%M"),
            "lotteries": ",".join(sorted(probs)),
            "top10": ",".join(top10),
            "top_list": ",".join(merged),
            "real_animals": ",".join(real),
            "hit_animals": ",".join(hits10),
            "hit_animals_top7": ",".join(hits7),
            "hit": bool(hits10),
            "hit_top7": bool(hits7),
            "chance_top10": chance_hit_any(TOP_N, len(real)),
            "chance_top7": chance_hit_any(len(merged), len(real)),
            "q_top10": float(1.0 - np.prod([1.0 - probs[L][[ANIMAL_NUM[a] for a in top10]].sum()
                                             for L in probs])),
        })
    return pd.DataFrame(out)


# ================================================================
# PRUEBAS DE HIPÓTESIS (¿se cumple el patrón?)
# ================================================================
#
# Cada prueba compara lo observado con lo que pasaría si cada sorteo
# fuera independiente y uniforme (1/38 por animal). Para no "encontrar"
# patrones por casualidad:
#   - se corrige por comparaciones múltiples (Bonferroni)
#   - se comprueba la REPLICACIÓN: la primera mitad del histórico
#     contra la segunda mitad (un patrón real se repite).
#
# ================================================================


ALPHA = 0.05


def verdict(p, alpha=ALPHA):
    return "Se cumple" if p < alpha else "No se cumple"


def _pc(x):
    """Porcentaje con coma decimal: 0.0265 -> '2,65 %'."""
    return f"{x * 100:.2f}".replace(".", ",") + " %"


def _chi2_contingency(table):
    O = np.asarray(table, dtype=float)
    E = O.sum(1, keepdims=True) * O.sum(0, keepdims=True) / O.sum()
    chi = float(((O - E) ** 2 / np.where(E > 0, E, 1)).sum())
    dof = (O.shape[0] - 1) * (O.shape[1] - 1)
    return chi, dof


def _halves(df):
    mid = df["datetime"].min() + (df["datetime"].max() - df["datetime"].min()) / 2
    return df[df["datetime"] < mid], df[df["datetime"] >= mid], mid


# ----------------------------------------------------------------
# 1. Uniformidad
# ----------------------------------------------------------------

def test_uniformity(df):
    out = []
    for L in LOTTERIES:
        c = np.bincount(df.loc[df["lottery"] == L, "number"], minlength=N_ANIMALS)
        e = c.sum() / N_ANIMALS
        chi = float(((c - e) ** 2 / e).sum())
        p = chi2_sf(chi, N_ANIMALS - 1)
        lo, hi = int(c.argmin()), int(c.argmax())
        out.append({
            "categoria": "Frecuencia", "prueba": "Uniformidad (χ² de bondad de ajuste)",
            "loteria": L, "estadistico": chi, "gl": N_ANIMALS - 1, "p_valor": p,
            "efecto": f"min {NUM_ANIMAL[lo]} {c[lo]} · max {NUM_ANIMAL[hi]} {c[hi]} · esperado {e:.0f}",
            "veredicto": "No uniforme" if p < ALPHA else "Compatible con uniforme",
        })
    return out


# ----------------------------------------------------------------
# 2. Repetición (anti-repetición) y curva de riesgo
# ----------------------------------------------------------------

def repetition_by_lag(df, max_lag=15):
    out = []
    for L in LOTTERIES:
        seq = df.loc[df["lottery"] == L].sort_values("datetime")["number"].to_numpy()
        for k in range(1, max_lag + 1):
            n = len(seq) - k
            rate = float((seq[:-k] == seq[k:]).mean())
            z = (rate - CHANCE) / math.sqrt(CHANCE * (1 - CHANCE) / n)
            out.append({"lottery": L, "lag": k, "n": n, "rate": rate, "z": z})
    return pd.DataFrame(out)


def test_repetition(df):
    rep = repetition_by_lag(df, 1)
    a, b, _ = _halves(df)
    ra, rb = repetition_by_lag(a, 1), repetition_by_lag(b, 1)
    out = []
    for L in LOTTERIES:
        r = rep[rep["lottery"] == L].iloc[0]
        out.append({
            "categoria": "Repetición", "prueba": "Repetir el animal del sorteo anterior",
            "loteria": L, "estadistico": r["z"], "gl": None, "p_valor": norm_two_sided(r["z"]),
            "efecto": (f"{_pc(r['rate'])} vs {_pc(CHANCE)} esperado · 1ª mitad "
                       f"{_pc(ra[ra.lottery == L].iloc[0]['rate'])} · 2ª mitad "
                       f"{_pc(rb[rb.lottery == L].iloc[0]['rate'])}"),
            "veredicto": ("Se cumple: casi nunca repite" if r["z"] < 0 and norm_two_sided(r["z"]) < ALPHA
                          else verdict(norm_two_sided(r["z"]))),
        })
    return out


HAZARD_EDGES = [1, 2, 3, 4, 6, 9, 13, 19, 26, 36, 51, 76, 101, 151]


def hazard_curve(rows, key="own_age", edges=HAZARD_EDGES):
    """P(sale | lleva k sorteos sin salir), agrupado en cubetas de k."""
    out = []
    for L in LOTTERIES:
        age = rows[L][key]
        y = rows[L]["y"]
        b = np.searchsorted(edges, age, side="right") - 1
        expo = np.bincount(b.ravel(), minlength=len(edges))
        hits = np.bincount(b[np.arange(len(y)), y], minlength=len(edges))
        for i, lo in enumerate(edges):
            hi = edges[i + 1] - 1 if i + 1 < len(edges) else None
            label = f"{lo}+" if hi is None else (f"{lo}" if hi == lo else f"{lo}-{hi}")
            rate = hits[i] / expo[i] if expo[i] else float("nan")
            z = ((rate - CHANCE) / math.sqrt(CHANCE * (1 - CHANCE) / expo[i])) if expo[i] else float("nan")
            out.append({"lottery": L, "bucket": label, "exposures": int(expo[i]),
                        "hits": int(hits[i]), "rate": rate, "z": z})
    return pd.DataFrame(out)


# ----------------------------------------------------------------
# 3. Pares en el mismo día
# ----------------------------------------------------------------

def sameday_pair_stats(pairs, mask=None):
    """C, E y Var (nula hipergeométrica por año) para los 703 pares."""
    mask = np.ones(len(pairs.X), bool) if mask is None else mask
    C = np.zeros((N_ANIMALS, N_ANIMALS))
    E = np.zeros_like(C)
    V = np.zeros_like(C)
    for y in np.unique(pairs.years[mask]):
        X = pairs.X[mask & (pairs.years == y)]
        D = len(X)
        if D < 2:
            continue
        n = X.sum(0)
        C += X.T @ X
        E += np.outer(n, n) / D
        V += np.outer(n * (D - n), n * (D - n)) / (D * D * (D - 1))
    return C, E, V


def test_sameday_pairs(pairs):
    C, E, V = sameday_pair_stats(pairs)
    mid = pairs.dates.min() + (pairs.dates.max() - pairs.dates.min()) / 2
    first = np.asarray(pairs.dates < mid)
    Ca, Ea, Va = sameday_pair_stats(pairs, first)
    Cb, Eb, Vb = sameday_pair_stats(pairs, ~first)
    iu = np.triu_indices(N_ANIMALS, 1)
    z = (C - E) / np.sqrt(np.where(V > 0, V, 1))
    za = (Ca - Ea) / np.sqrt(np.where(Va > 0, Va, 1))
    zb = (Cb - Eb) / np.sqrt(np.where(Vb > 0, Vb, 1))
    n_pairs = len(iu[0])
    res = pd.DataFrame({
        "a": [NUM_ANIMAL[i] for i in iu[0]],
        "b": [NUM_ANIMAL[j] for j in iu[1]],
        "observado": C[iu], "esperado": E[iu],
        "ratio": C[iu] / E[iu], "z": z[iu],
        "z_1a_mitad": za[iu], "z_2a_mitad": zb[iu],
    })
    res["p_valor"] = [norm_two_sided(v) for v in res["z"]]
    res["p_bonferroni"] = np.minimum(1.0, res["p_valor"] * n_pairs)
    res["replica"] = (np.sign(res["z_1a_mitad"]) == np.sign(res["z_2a_mitad"])) & (
        np.minimum(abs(res["z_1a_mitad"]), abs(res["z_2a_mitad"])) > 2)
    res["par"] = res["a"].map(animal_label) + " ~ " + res["b"].map(animal_label)
    replication_corr = float(np.corrcoef(za[iu], zb[iu])[0, 1])
    return res.sort_values("z").reset_index(drop=True), replication_corr


def pair_summary_rows(pair_res, replication_corr):
    avoided = pair_res[(pair_res["p_bonferroni"] < ALPHA) & (pair_res["z"] < 0)]
    attracted = pair_res[(pair_res["p_bonferroni"] < ALPHA) & (pair_res["z"] > 0)]
    top = pair_res.iloc[0]
    best_pos = pair_res.iloc[-1]
    return [
        {
            "categoria": "Pares mismo día", "prueba": "Pares que se evitan el mismo día (703 pares, Bonferroni)",
            "loteria": "todas", "estadistico": float(top["z"]), "gl": None,
            "p_valor": float(top["p_bonferroni"]),
            "efecto": (f"{len(avoided)} pares significativos; el más fuerte {top['par']} "
                       f"sale junto el {top['ratio']*100:.0f} % de lo esperado"),
            "veredicto": "Se cumple" if len(avoided) else "No se cumple",
        },
        {
            "categoria": "Pares mismo día", "prueba": "Pares que salen juntos el mismo día (703 pares, Bonferroni)",
            "loteria": "todas", "estadistico": float(best_pos["z"]), "gl": None,
            "p_valor": float(best_pos["p_bonferroni"]),
            "efecto": (f"{len(attracted)} pares significativos; el mayor exceso es {best_pos['par']} "
                       f"({best_pos['ratio']*100:.0f} % de lo esperado)"),
            "veredicto": "Se cumple" if len(attracted) else "No se cumple",
        },
        {
            "categoria": "Pares mismo día", "prueba": "Replicación de los pares (1ª vs 2ª mitad del histórico)",
            "loteria": "todas", "estadistico": replication_corr, "gl": None, "p_valor": None,
            "efecto": f"correlación de los z de los 703 pares entre mitades = {replication_corr:.2f}".replace(".", ","),
            "veredicto": "Se replica" if replication_corr > 0.2 else "No se replica",
        },
    ]


# ----------------------------------------------------------------
# 4. Transiciones (qué animal sigue a cuál)
# ----------------------------------------------------------------

def _transition_counts(df):
    M = np.zeros((N_ANIMALS, N_ANIMALS))
    for L in LOTTERIES:
        s = df.loc[df["lottery"] == L].sort_values("datetime")["number"].to_numpy()
        np.add.at(M, (s[:-1], s[1:]), 1)
    return M


def _quasi_independence(M):
    """χ² de cuasi-independencia (ignora la diagonal = repeticiones)."""
    mask = ~np.eye(N_ANIMALS, dtype=bool)
    E = mask.astype(float)
    for _ in range(300):
        E *= (M.sum(1) / np.where(E.sum(1) > 0, E.sum(1), 1))[:, None]
        E *= (M.sum(0) / np.where(E.sum(0) > 0, E.sum(0), 1))[None, :]
    chi = float(((M - E) ** 2 / np.where(E > 0, E, 1))[mask].sum())
    dof = (N_ANIMALS - 1) ** 2 - N_ANIMALS
    return chi, dof, E


def test_transitions(df):
    a, b, _ = _halves(df)
    out = []
    M = _transition_counts(df)
    chi, dof, E = _quasi_independence(M)
    Ma, Mb = _transition_counts(a), _transition_counts(b)
    _, _, Ea = _quasi_independence(Ma)
    _, _, Eb = _quasi_independence(Mb)
    mask = ~np.eye(N_ANIMALS, dtype=bool)
    za = ((Ma - Ea) / np.sqrt(np.where(Ea > 0, Ea, 1)))[mask]
    zb = ((Mb - Eb) / np.sqrt(np.where(Eb > 0, Eb, 1)))[mask]
    corr = float(np.corrcoef(za, zb)[0, 1])
    out.append({
        "categoria": "Transiciones", "prueba": "¿El animal siguiente depende del anterior? (χ² cuasi-independencia)",
        "loteria": "todas", "estadistico": chi, "gl": dof, "p_valor": chi2_sf(chi, dof),
        "efecto": f"replicación entre mitades: correlación {corr:.2f}".replace(".", ","),
        "veredicto": verdict(chi2_sf(chi, dof)),
    })
    # Paso numérico ±1 (vecino en la tabla)
    for L in LOTTERIES:
        s = df.loc[df["lottery"] == L].sort_values("datetime")["number"].to_numpy()
        step = (s[1:] - s[:-1]) % N_ANIMALS
        nz = step[step != 0]
        n = len(nz)
        k = int(np.isin(nz, (1, N_ANIMALS - 1)).sum())
        base = 2.0 / (N_ANIMALS - 1)
        z = (k / n - base) / math.sqrt(base * (1 - base) / n)
        out.append({
            "categoria": "Transiciones", "prueba": "Paso numérico ±1 (el número vecino)",
            "loteria": L, "estadistico": z, "gl": None, "p_valor": norm_two_sided(z),
            "efecto": f"{_pc(k / n)} vs {_pc(base)} esperado (sin contar repeticiones)",
            "veredicto": verdict(norm_two_sided(z)),
        })
    return out


# ----------------------------------------------------------------
# 5. Efectos entre loterías
# ----------------------------------------------------------------

# (origen, destino): ¿el destino repite lo que salió antes en el origen?
CROSS_PAIRS = [
    ("lottoactivo", "lagranjita"), ("lagranjita", "lottoactivo"),
    ("lottoactivoint", "lottoactivo"), ("lottoactivo", "lottoactivoint"),
    ("lottoactivoint", "lagranjita"), ("lagranjita", "lottoactivoint"),
]


def cross_lottery_lags(df, max_lag=7):
    """
    Para cada sorteo de `destino`, ¿coincide con el k-ésimo sorteo más
    reciente de `origen` anterior a él, el mismo día? (k = 1..max_lag).
    k = 0 es la coincidencia a la misma hora (solo si sortean a la vez).
    El orden es temporal real: Lotto Activo Int sortea a las hh:30.
    """
    out = []
    by = {L: df[df["lottery"] == L].sort_values("datetime") for L in LOTTERIES}
    for src, dst in CROSS_PAIRS:
        s, d = by[src], by[dst]
        if len(s) < 500 or len(d) < 500:
            continue
        s_t = s["datetime"].to_numpy()
        s_n = s["number"].to_numpy()
        s_day = s["datetime"].dt.normalize().to_numpy()
        d_t = d["datetime"].to_numpy()
        d_n = d["number"].to_numpy()
        d_day = d["datetime"].dt.normalize().to_numpy()
        pa = np.bincount(s_n, minlength=N_ANIMALS) / len(s_n)
        pb = np.bincount(d_n, minlength=N_ANIMALS) / len(d_n)
        base = float((pa * pb).sum())
        before = np.searchsorted(s_t, d_t, side="left") - 1
        for k in range(0, max_lag + 1):
            if k == 0:
                j = np.minimum(before + 1, len(s_t) - 1)
                valid = s_t[j] == d_t
            else:
                j = np.maximum(before - (k - 1), 0)
                valid = (before - (k - 1) >= 0) & (s_day[j] == d_day)
            n = int(valid.sum())
            if n < 200:
                continue
            rate = float((s_n[j[valid]] == d_n[valid]).mean())
            z = (rate - base) / math.sqrt(base * (1 - base) / n)
            out.append({"source": src, "target": dst, "lag": k, "n": n,
                        "rate": rate, "chance": base, "z": z})
    return pd.DataFrame(out)


def _pooled_lag_test(lags, src, dst, k_lo, k_hi, label):
    sub = lags[(lags["source"] == src) & (lags["target"] == dst) & lags["lag"].between(k_lo, k_hi)]
    if sub.empty:
        return None
    k = (sub["rate"] * sub["n"]).sum()
    n = sub["n"].sum()
    base = float(sub["chance"].iloc[0])
    z = (k / n - base) / math.sqrt(base * (1 - base) / n)
    p = norm_two_sided(z)
    sentido = "más" if z > 0 else "menos"
    return {
        "categoria": "Entre loterías", "prueba": label, "loteria": dst, "estadistico": z, "gl": None,
        "p_valor": p, "efecto": f"{_pc(k / n)} vs {_pc(base)} esperado ({sentido} de lo normal)",
        "veredicto": verdict(p),
    }


def _same_day_overlap(df, a, b):
    """¿Sale el mismo animal en las loterías a y b el mismo día más de lo esperado?"""
    days = {}
    for L in (a, b):
        sub = df[df["lottery"] == L]
        days[L] = sub.groupby(sub["datetime"].dt.normalize())["number"].apply(
            lambda s: np.bincount(s.to_numpy(), minlength=N_ANIMALS) > 0)
    common = days[a].index.intersection(days[b].index)
    A = np.vstack(days[a].loc[common].to_numpy()).astype(float)
    B = np.vstack(days[b].loc[common].to_numpy()).astype(float)
    D = len(common)
    obs = float((A * B).sum())
    na, nb = A.sum(0), B.sum(0)
    exp = float((na * nb / D).sum())
    var = float((na * nb * (D - na) * (D - nb) / (D * D * (D - 1))).sum())
    z = (obs - exp) / math.sqrt(var)
    return {
        "categoria": "Entre loterías",
        "prueba": f"El mismo animal sale en {LOTTERY_LABEL[a]} y {LOTTERY_LABEL[b]} el mismo día",
        "loteria": "todas", "estadistico": z, "gl": None, "p_valor": norm_two_sided(z),
        "efecto": f"{obs:.0f} veces vs {exp:.0f} esperadas ({(obs / exp - 1) * 100:+.1f} %)".replace(".", ","),
        "veredicto": verdict(norm_two_sided(z)),
    }


def test_cross_lottery(df):
    lags = cross_lottery_lags(df)
    out = []
    same = lags[(lags["lag"] == 0) & (lags["source"] == "lottoactivo") & (lags["target"] == "lagranjita")]
    if not same.empty:
        s = same.iloc[0]
        out.append({
            "categoria": "Entre loterías",
            "prueba": "Mismo animal en La Granjita y Lotto Activo a la misma hora",
            "loteria": "todas", "estadistico": s["z"], "gl": None, "p_valor": norm_two_sided(s["z"]),
            "efecto": f"{_pc(s['rate'])} vs {_pc(s['chance'])} esperado",
            "veredicto": verdict(norm_two_sided(s["z"])),
        })
    for src, dst in CROSS_PAIRS:
        a, b = LOTTERY_LABEL[src], LOTTERY_LABEL[dst]
        for lo, hi, txt in [(1, 1, "el animal que acaba de salir en"), (2, 6, "lo que salió 2-6 sorteos antes en")]:
            r = _pooled_lag_test(lags, src, dst, lo, hi, f"{b} repite {txt} {a}")
            if r:
                out.append(r)
    for a, b in [("lagranjita", "lottoactivo"), ("lottoactivo", "lottoactivoint")]:
        if (df["lottery"] == a).sum() > 500 and (df["lottery"] == b).sum() > 500:
            out.append(_same_day_overlap(df, a, b))
    return out, lags


# ----------------------------------------------------------------
# 6. Hora, día de la semana y dispersión
# ----------------------------------------------------------------

def test_independence_calendar(df):
    out = []
    for L in LOTTERIES:
        sub = df[df["lottery"] == L]
        ct = pd.crosstab(sub["number"], sub["hour"])
        ct = ct.loc[:, ct.sum(axis=0) >= 300]
        chi, dof = _chi2_contingency(ct.to_numpy())
        p = chi2_sf(chi, dof)
        out.append({
            "categoria": "Calendario", "prueba": "¿Ciertos animales salen más a ciertas horas? (χ²)",
            "loteria": L, "estadistico": chi, "gl": dof, "p_valor": p,
            "efecto": f"χ²/gl = {chi / dof:.2f} (≈1 si no hay efecto)".replace(".", ","), "veredicto": verdict(p),
        })
        ct = pd.crosstab(sub["number"], sub["datetime"].dt.dayofweek)
        chi, dof = _chi2_contingency(ct.to_numpy())
        p = chi2_sf(chi, dof)
        out.append({
            "categoria": "Calendario", "prueba": "¿Ciertos animales salen más ciertos días de la semana? (χ²)",
            "loteria": L, "estadistico": chi, "gl": dof, "p_valor": p,
            "efecto": f"χ²/gl = {chi / dof:.2f} (≈1 si no hay efecto)".replace(".", ","), "veredicto": verdict(p),
        })
    return out


def test_dispersion(df, block_days=30):
    out = []
    for L in LOTTERIES:
        sub = df[df["lottery"] == L]
        blk = ((sub["datetime"] - sub["datetime"].min()).dt.days // block_days).to_numpy()
        ct = pd.crosstab(blk, sub["number"])
        ct = ct[ct.sum(axis=1) >= 250]
        chi, dof = _chi2_contingency(ct.to_numpy())
        p_low = chi2_cdf(chi, dof)
        p_high = chi2_sf(chi, dof)
        ratio = chi / dof
        if p_low < ALPHA:
            v = "Se cumple: reparto MÁS parejo que el azar"
        elif p_high < ALPHA:
            v = "Se cumple: hay rachas por periodos"
        else:
            v = "No se cumple"
        out.append({
            "categoria": "Frecuencia", "prueba": f"Rachas por periodos de {block_days} días (dispersión χ²/gl)",
            "loteria": L, "estadistico": chi, "gl": dof, "p_valor": min(1.0, 2 * min(p_low, p_high)),
            "efecto": f"χ²/gl = {ratio:.2f} (<1 reparto parejo, >1 rachas)".replace(".", ","), "veredicto": v,
        })
    return out


def run_pattern_tests(df, table, pairs):
    rows_out = []
    rows_out += test_uniformity(df)
    rows_out += test_dispersion(df)
    rows_out += test_repetition(df)
    pair_res, rep_corr = test_sameday_pairs(pairs)
    rows_out += pair_summary_rows(pair_res, rep_corr)
    rows_out += test_transitions(df)
    cross_rows, lags = test_cross_lottery(df)
    rows_out += cross_rows
    rows_out += test_independence_calendar(df)
    return pd.DataFrame(rows_out), pair_res, lags


# ================================================================
# PREDICCIONES: próximo turno y Monte Carlo de 11 turnos
# ================================================================


# ----------------------------------------------------------------
# Horarios reales (inferidos de los datos recientes)
# ----------------------------------------------------------------

def infer_schedule(df, days=21, min_share=0.4):
    """Horas en que cada lotería sorteó en al menos el 40% de los últimos días."""
    last = df["datetime"].max().normalize()
    recent = df[df["datetime"] >= last - timedelta(days=days - 1)]
    schedule = {}
    for L in LOTTERIES:
        sub = recent[recent["lottery"] == L]
        n_days = max(sub["datetime"].dt.normalize().nunique(), 1)
        share = sub.groupby("hour")["datetime"].nunique() / n_days
        hours = sorted(share[share >= min_share].index)
        schedule[L] = hours or sorted(df.loc[df["lottery"] == L, "hour"].unique())
    return schedule


def upcoming_turns(after, schedule, n):
    """Los próximos n turnos (fecha-hora, loterías) estrictamente posteriores a `after`."""
    by_hour = defaultdict(list)
    for L, hours in schedule.items():
        for h in hours:
            by_hour[h].append(L)
    hours = sorted(by_hour)
    out = []
    day = after.normalize()
    while len(out) < n:
        for h in hours:
            ts = pd.Timestamp(f"{day.strftime('%Y-%m-%d')} {h}")
            if ts > after:
                out.append((ts, sorted(by_hour[h])))
                if len(out) == n:
                    break
        day += timedelta(days=1)
    return out


# ----------------------------------------------------------------
# Modelos finales (entrenados con TODO el histórico)
# ----------------------------------------------------------------

def fit_final_models(rows, pairs, last_time):
    cutoff = last_time + timedelta(minutes=1)
    return {L: fit_model(rows, pairs, L, cutoff, FINAL_GROUPS) for L in LOTTERIES}


def predict_next_turn(df, table, state, models, schedule):
    """
    Predice el próximo sorteo de cada lotería (cada una con su hora).

    Si a una lotería le falta un resultado anterior al último conocido
    (p. ej. ya publicó Lotto Activo pero aún no La Granjita), se predice
    con el estado JUSTO ANTES de ese sorteo, igual que en el entrenamiento.
    """
    last_turn = table.index.max()
    nxt = {}
    for L in LOTTERIES:
        last_L = df.loc[df["lottery"] == L, "datetime"].max()
        nxt[L] = upcoming_turns(last_L, {L: schedule[L]}, 1)[0][0]
    past = sorted({t for t in nxt.values() if t <= last_turn})
    _, _, snaps = replay_history(table, snapshot_turns=past) if past else (None, None, {})

    out = {}
    for L in LOTTERIES:
        t = nxt[L]
        st = (snaps[t] if t in snaps else state).copy_tiled(1)
        st.start_turn(t.normalize())
        out[L] = {"datetime": t, "probs": models[L].proba(st.inputs(L))[0]}
    return out


def next_turn_table(pred):
    """
    P de cada animal en el próximo sorteo de cada lotería y P de salir en
    al menos una de las que sortean a la misma hora (La Granjita + Lotto Activo).
    """
    turn = min(pred[L]["datetime"] for L in SAME_TURN)
    same = [L for L in SAME_TURN if pred[L]["datetime"] == turn]
    q = union_probability([pred[L]["probs"] for L in same])
    df_out = pd.DataFrame({
        "animal": [NUM_ANIMAL[i] for i in range(N_ANIMALS)],
        "numero": ["00" if i == 37 else str(i) for i in range(N_ANIMALS)],
    })
    for L in LOTTERIES:
        df_out[f"p_{L}"] = pred[L]["probs"]
        df_out[f"hora_{L}"] = pred[L]["datetime"]
    df_out["p_consolidado"] = q
    df_out["datetime"] = turn
    df_out = df_out.sort_values("p_consolidado", ascending=False).reset_index(drop=True)
    df_out["rank"] = np.arange(1, len(df_out) + 1)
    return df_out, turn, same


# ----------------------------------------------------------------
# Monte Carlo: P(aparecer al menos una vez en los próximos N turnos)
# ----------------------------------------------------------------

def simulate_appearance(state, models_for, turns, n_sims, rng):
    """
    Simula n_sims futuros posibles, sorteo a sorteo, con el modelo:
    cada resultado simulado actualiza el estado (antigüedad, pares...),
    exactamente como ocurriría en la realidad.
    `models_for(L)` devuelve el modelo a usar para la lotería L.
    """
    st = state.copy_tiled(n_sims)
    seen = np.zeros((n_sims, N_ANIMALS), bool)
    idx = np.arange(n_sims)
    for ts, lots in turns:
        st.start_turn(ts.normalize())
        draws = {}
        for L in lots:
            P = models_for(L).proba(st.inputs(L))
            u = rng.random(n_sims)[:, None]
            draws[L] = np.minimum((P.cumsum(axis=1) < u).sum(axis=1), N_ANIMALS - 1)
        for L, w in draws.items():
            st.update(L, w)
            seen[idx, w] = True
    return seen.mean(axis=0)


def upcoming_hours(after, schedule, n_hours):
    """Sorteos de las próximas `n_hours` franjas horarias (hh:00 y hh:30 de cada hora)."""
    turns = upcoming_turns(after, schedule, n_hours * 3)
    hours = []
    out = []
    for ts, lots in turns:
        h = ts.floor("h")
        if h not in hours:
            if len(hours) == n_hours:
                break
            hours.append(h)
        out.append((ts, lots))
    return out


def eleven_turn_prediction(state, models, last_time, schedule, rng):
    turns = upcoming_hours(last_time, schedule, ELEVEN_TURNS)
    probs = simulate_appearance(state, lambda L: models[L], turns, N_SIMULATIONS, rng)
    out = pd.DataFrame({"animal": [NUM_ANIMAL[i] for i in range(N_ANIMALS)], "probability": probs})
    out = out.sort_values("probability", ascending=False).reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1)
    n_draws = sum(len(l) for _, l in turns)
    chance_one = 1.0 - (1.0 - CHANCE) ** n_draws
    return out, turns, chance_one


def eleven_turn_history(table, fold_models, rng, n_starts=ELEVEN_HISTORY_STARTS):
    """
    Evalúa el cálculo de 11 turnos (11 franjas horarias, todas las
    loterías) en los últimos `n_starts` turnos que ya tienen sus 11 horas
    siguientes completas. Usa el modelo del bloque walk-forward
    correspondiente (sin información futura).
    """
    turns = table.index
    slots = turns.floor("h")
    uniq = slots.unique()
    if len(uniq) < n_starts + ELEVEN_TURNS + 1:
        return pd.DataFrame()
    start_slots = uniq[-(n_starts + ELEVEN_TURNS):-ELEVEN_TURNS]
    starts = [turns[slots == s][0] for s in start_slots]
    _, _, snaps = replay_history(table, snapshot_turns=starts)
    by_lottery = {L: sorted(c for (lot, c) in fold_models if lot == L) for L in LOTTERIES}

    def model_at(L, ts):
        valid = [c for c in by_lottery[L] if c <= ts]
        return fold_models[(L, valid[-1])] if valid else None

    out = []
    for s, ts in zip(start_slots, starts):
        if any(model_at(L, ts) is None for L in LOTTERIES):
            continue
        i = uniq.get_loc(s)
        window = turns[slots.isin(uniq[i:i + ELEVEN_TURNS])]
        future = table.loc[window]
        sched = [(t, [L for L in LOTTERIES if not np.isnan(future.loc[t, L])]) for t in window]
        probs = simulate_appearance(snaps[ts], lambda L: model_at(L, ts), sched,
                                    N_SIMULATIONS_HISTORY, rng)
        top = [NUM_ANIMAL[a] for a in np.argsort(-probs)[:ELEVEN_TOP]]
        real = {NUM_ANIMAL[int(v)] for v in future.to_numpy().ravel() if not np.isnan(v)}
        hits = [a for a in top if a in real]
        out.append({
            "datetime": ts, "date": ts.strftime("%Y-%m-%d"), "time": ts.strftime("%H:%M"),
            "top5": ",".join(top), "hit_animals": ",".join(sorted(hits)), "n_hits": len(hits),
            "hit": bool(hits), "distinct_real": len(real),
            "chance_hit": chance_hit_any(ELEVEN_TOP, len(real)),
            "expected_hits_model": float(np.sort(probs)[-ELEVEN_TOP:].sum()),
            "expected_hits_chance": ELEVEN_TOP * len(real) / N_ANIMALS,
        })
    return pd.DataFrame(out)


# ================================================================
# ESTRATEGIA DE APUESTA (valor esperado)
# ================================================================
#
# Con pago P (el premio es P veces lo apostado), apostar 1 unidad a un
# animal con probabilidad p tiene valor esperado  EV = p·P − 1.
# Al azar (p = 1/38) y P = 30:  EV = 30/38 − 1 = −21 %  (margen de la casa).
# Si el modelo sube p lo suficiente (p > 1/30 = 3,33 %), el EV es positivo.
#
# Reglas evaluadas (fijas, no se ajustan con el periodo de evaluación):
#   - "Top K": 1 unidad a cada uno de los K animales más probables.
#   - "Valor": 1 unidad a cada animal con p·P ≥ 1 + margen (máx. 3).
# La regla principal (MAIN_STRATEGY) se eligió con el AÑO ANTERIOR
# (mejor cota inferior del IC 95 % del ROI) y se informa su resultado en
# el ÚLTIMO AÑO, que no se usó para elegirla.
#
# ================================================================

STRATEGY_RULES = {
    "top1": ("Top 1 por sorteo", {"k": 1}),
    "top2": ("Top 2 por sorteo", {"k": 2}),
    "top3": ("Top 3 por sorteo", {"k": 3}),
    "top5": ("Top 5 por sorteo", {"k": 5}),
    "top10": ("Top 10 por sorteo", {"k": 10}),
    "valor30": ("Valor esperado ≥ +30 % (máx. 3 por sorteo)", {"margin": 0.30, "kmax": 3}),
}
# Elegida con el año 2024-10 -> 2025-10 (mejor cota inferior del IC 95 % del
# ROI: valor +32 %, top 2 +29 %, top 3 +28 %); el último año no se usó para elegir.
MAIN_STRATEGY = "valor30"
ALT_STRATEGY = "top2"
STRATEGY_DAYS = 730          # año anterior (selección) + último año (evaluación)
BOOTSTRAP_RESAMPLES = 2000


def strategy_bets(P, rule, payouts):
    """Matriz booleana (n, 38): a qué animales se apuesta 1 unidad en cada sorteo."""
    n = len(P)
    order = np.argsort(-P, axis=1)
    rank = np.empty_like(order)
    rank[np.arange(n)[:, None], order] = np.arange(N_ANIMALS)[None, :]
    if "k" in rule:
        return rank < rule["k"]
    value = P * payouts[:, None] >= 1.0 + rule["margin"]
    return value & (rank < rule.get("kmax", N_ANIMALS))


def strategy_ledger(bt, key):
    """Una fila por sorteo: unidades apostadas, cobradas y ganancia."""
    rule = STRATEGY_RULES[key][1]
    P = np.vstack(bt["probs"].to_numpy())
    payouts = bt["lottery"].map(PAYOUT).to_numpy(dtype=float)
    B = strategy_bets(P, rule, payouts)
    y = bt["winner"].to_numpy()
    won = B[np.arange(len(y)), y]
    stake = B.sum(axis=1).astype(float)
    ret = np.where(won, payouts, 0.0)
    order = np.argsort(-P, axis=1)[:, :TOP_N]      # todas las reglas apuestan dentro del top 10
    bets_txt = [",".join(NUM_ANIMAL[a] for a in o if b[a]) for o, b in zip(order, B)]
    return pd.DataFrame({
        "datetime": bt["datetime"].to_numpy(), "lottery": bt["lottery"].to_numpy(),
        "bets": bets_txt, "stake": stake, "return": ret, "profit": ret - stake,
        "won": won, "winner": [NUM_ANIMAL[a] for a in y],
    })


def summarize_ledger(led, rng):
    """ROI con IC 95 % (bootstrap por días), caídas, rachas y frecuencia de ganancia."""
    bet = led[led["stake"] > 0]
    S, R = float(led["stake"].sum()), float(led["return"].sum())
    out = {"draws": len(led), "draws_bet": len(bet), "stake": S, "return": R, "profit": R - S,
           "roi": R / S - 1 if S else float("nan"),
           "hit_rate": float(bet["won"].mean()) if len(bet) else float("nan")}
    day = led.groupby(led["datetime"].dt.normalize()).agg(stake=("stake", "sum"), ret=("return", "sum"))
    if len(day) and S:
        idx = rng.integers(0, len(day), size=(BOOTSTRAP_RESAMPLES, len(day)))
        ds, dr = day["stake"].to_numpy()[idx].sum(1), day["ret"].to_numpy()[idx].sum(1)
        boots = dr / np.maximum(ds, 1e-9) - 1
        out["roi_lo"], out["roi_hi"] = (float(v) for v in np.percentile(boots, [2.5, 97.5]))
        out["p_roi_negative"] = float(np.mean(boots <= 0))
    else:
        out["roi_lo"] = out["roi_hi"] = out["p_roi_negative"] = float("nan")
    cum = led["profit"].cumsum().to_numpy()
    peak = np.maximum.accumulate(np.concatenate([[0.0], cum]))[1:]
    out["max_drawdown"] = float((peak - cum).max()) if len(cum) else 0.0
    streak = worst = 0
    for s, w in zip(led["stake"].to_numpy(), led["won"].to_numpy()):
        if s > 0:
            streak = 0 if w else streak + 1
            worst = max(worst, streak)
    out["losing_streak"] = worst
    pnl = led.set_index("datetime")["profit"]
    for name, freq in (("days", "D"), ("weeks", "W"), ("months", "MS")):
        g = pnl.resample(freq).sum()
        g = g[led.set_index("datetime")["stake"].resample(freq).sum() > 0]
        out[f"pct_{name}_up"] = float((g > 0).mean()) if len(g) else float("nan")
        out[f"worst_{name[:-1]}"] = float(g.min()) if len(g) else float("nan")
    return out


def bankroll_simulation(led, rng, horizons=(7, 30, 90, 365), n_sims=5000):
    """
    Monte Carlo: se re-muestrean días reales del periodo de evaluación
    para estimar la distribución de la ganancia tras N días.
    """
    daily = led.groupby(led["datetime"].dt.normalize())["profit"].sum().to_numpy()
    stake_day = float(led.groupby(led["datetime"].dt.normalize())["stake"].sum().mean())
    out = []
    if len(daily) == 0:
        return pd.DataFrame(), stake_day
    for h in horizons:
        sims = daily[rng.integers(0, len(daily), size=(n_sims, h))]
        total = sims.sum(axis=1)
        path = np.cumsum(sims, axis=1)
        dd = (np.maximum.accumulate(np.maximum(path, 0), axis=1) - path).max(axis=1)
        out.append({"dias": h, "p_perder": float(np.mean(total < 0)),
                    "ganancia_mediana": float(np.median(total)),
                    "ganancia_p5": float(np.percentile(total, 5)),
                    "ganancia_p95": float(np.percentile(total, 95)),
                    "caida_max_p95": float(np.percentile(dd, 95)),
                    "apostado": stake_day * h})
    return pd.DataFrame(out), stake_day


def evaluate_strategies(bt_long, eval_start, rng):
    """Todas las reglas en el año anterior (selección) y en el último año (evaluación)."""
    sel = bt_long[bt_long["datetime"] < eval_start]
    ev = bt_long[bt_long["datetime"] >= eval_start]
    rows_out = []
    ledgers = {}
    for key, (label, _) in STRATEGY_RULES.items():
        led_ev = strategy_ledger(ev, key)
        ledgers[key] = led_ev
        for period, data, led in (("seleccion", sel, None), ("evaluacion", ev, led_ev)):
            if data.empty:
                continue
            led = led if led is not None else strategy_ledger(data, key)
            s = summarize_ledger(led, rng)
            s.update(strategy=key, label=label, period=period)
            rows_out.append(s)
    per_lot = []
    for key in (MAIN_STRATEGY, ALT_STRATEGY):
        led = ledgers[key]
        for L, g in led.groupby("lottery"):
            S, R = g["stake"].sum(), g["return"].sum()
            per_lot.append({"strategy": key, "lottery": L, "stake": S, "ret": R,
                            "roi": R / S - 1 if S else float("nan")})
    return pd.DataFrame(rows_out), ledgers, pd.DataFrame(per_lot)


def kelly_fraction(p, payout):
    """Fracción de Kelly para una apuesta simple: f* = (p·P − 1) / (P − 1)."""
    return max(0.0, (p * payout - 1.0) / (payout - 1.0))


# ================================================================
# DASHBOARD HTML (SVG en línea, sin dependencias externas)
# ================================================================


# ----------------------------------------------------------------
# Formato
# ----------------------------------------------------------------

def esc(x):
    return _html.escape(str(x))


def pct(x, d=1):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    return f"{x * 100:.{d}f}".replace(".", ",") + " %"


def num(x, d=2):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    return f"{x:.{d}f}".replace(".", ",")


def intf(x):
    """Entero con separador de miles: 12345 -> '12.345'."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    return f"{x:,.0f}".replace(",", ".")


def p_text(p):
    return format_p(p).replace(".", ",") if p is not None else "—"


def label_list(names):
    return ", ".join(animal_label(a) for a in names if a)


def split_names(text):
    return [a for a in str(text).split(",") if a and a != "nan"]


# ----------------------------------------------------------------
# Piezas HTML
# ----------------------------------------------------------------

def status_pill(hit):
    if hit:
        return '<span class="pill pill-hit"><span class="pill-icon">✓</span>HIT</span>'
    return '<span class="pill pill-miss"><span class="pill-icon">✗</span>MISS</span>'


def verdict_pill(text):
    yes = text.startswith("Se cumple") or text.startswith("No uniforme") or text.startswith("Se replica")
    cls = "pill-yes" if yes else "pill-no"
    icon = "●" if yes else "○"
    return f'<span class="pill {cls}"><span class="pill-icon">{icon}</span>{esc(text)}</span>'


def stat_tile(label, value, sub="", hero=False):
    cls = "tile hero" if hero else "tile"
    return (f'<div class="{cls}"><div class="tile-label">{esc(label)}</div>'
            f'<div class="tile-value">{esc(value)}</div>'
            f'<div class="tile-sub">{sub}</div></div>')


def prob_bar(p, scale_max, chance=None):
    """Barra horizontal de probabilidad con marca del azar."""
    w = 0 if scale_max <= 0 else min(100.0, p / scale_max * 100)
    marker = ""
    if chance is not None and scale_max > 0:
        c = min(100.0, chance / scale_max * 100)
        marker = f'<span class="bar-chance" style="left:{c:.2f}%"></span>'
    return (f'<div class="pbar"><span class="pbar-fill" style="width:{w:.2f}%"></span>{marker}</div>')


def table_html(headers, rows, cls="data", num_cols=()):
    th = "".join(f'<th class="{"num" if i in num_cols else ""}">{h}</th>' for i, h in enumerate(headers))
    body = []
    for r in rows:
        tds = "".join(f'<td class="{"num" if i in num_cols else ""}">{c}</td>' for i, c in enumerate(r))
        body.append(f"<tr>{tds}</tr>")
    return (f'<div class="table-wrap"><table class="{cls}"><thead><tr>{th}</tr></thead>'
            f'<tbody>{"".join(body)}</tbody></table></div>')


def data_details(headers, rows, num_cols=()):
    return (f'<details class="tableview"><summary>Ver los datos en tabla</summary>'
            f'{table_html(headers, rows, num_cols=num_cols)}</details>')


# ----------------------------------------------------------------
# Gráficos SVG
# ----------------------------------------------------------------

_chart_counter = [0]


def _nice_ticks(lo, hi, n=5):
    span = hi - lo
    if span <= 0:
        return [lo]
    raw = span / n
    mag = 10 ** math.floor(math.log10(raw))
    step = min((m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw), default=raw)
    start = math.ceil(lo / step) * step
    ticks = []
    v = start
    while v <= hi + 1e-12:
        ticks.append(round(v, 12))
        v += step
    return ticks


def _spread_labels(items, gap=14.0):
    """
    Separa verticalmente etiquetas que chocan. items: [y, texto, fija].
    Las fijas (p. ej. la del azar, pegada a su línea) no se mueven.
    """
    items.sort(key=lambda e: e[0])
    for _ in range(50):
        moved = False
        for k in range(1, len(items)):
            a, b = items[k - 1], items[k]
            overlap = gap - (b[0] - a[0])
            if overlap <= 0:
                continue
            moved = True
            if a[2] and not b[2]:
                b[0] += overlap
            elif b[2] and not a[2]:
                a[0] -= overlap
            else:
                a[0] -= overlap / 2
                b[0] += overlap / 2
        if not moved:
            break
    return items


def svg_line_chart(x_labels, series, y_min, y_max, ref=None, y_fmt=lambda v: pct(v, 0),
                   tip_fmt=lambda v: pct(v, 2), aria="", width=760, height=300,
                   x_title="", show_every=1):
    """
    series: [{"name", "values": [float|None], "cls": "s1"|"s2"|"s3", "short" (opcional)}]
            "short" es la etiqueta al final de la línea (el tooltip usa "name").
    ref:    (valor, etiqueta) -> línea de referencia horizontal (azar).
    """
    _chart_counter[0] += 1
    cid = f"c{_chart_counter[0]}"
    n = len(x_labels)
    ml, mr, mt = 52, 150, 14
    band = (width - ml - mr) / n
    # Si las etiquetas del eje X no caben en una fila, se alternan en dos.
    label_w = max(len(str(l)) for l in x_labels) * 6.6
    stagger = label_w > band * show_every - 4
    mb = 60 if stagger else 46
    pw, ph = width - ml - mr, height - mt - mb

    def X(i):
        return ml + (pw * (i + 0.5) / n)

    def Y(v):
        return mt + ph * (1 - (v - y_min) / (y_max - y_min))

    parts = [f'<svg class="chart" viewBox="0 0 {width} {height}" role="img" aria-label="{esc(aria)}">']
    for t in _nice_ticks(y_min, y_max):
        y = Y(t)
        parts.append(f'<line class="grid" x1="{ml}" x2="{ml + pw}" y1="{y:.1f}" y2="{y:.1f}"/>')
        parts.append(f'<text class="tick" x="{ml - 8}" y="{y + 4:.1f}" text-anchor="end">{esc(y_fmt(t))}</text>')
    parts.append(f'<line class="axis" x1="{ml}" x2="{ml + pw}" y1="{mt + ph}" y2="{mt + ph}"/>')
    shown = [i for i in range(n) if i % show_every == 0]
    if (n - 1) not in shown and (n - 1 - shown[-1]) >= show_every * 0.75:
        shown.append(n - 1)
    for k, i in enumerate(shown):
        dy = 18 + (14 if stagger and k % 2 else 0)
        parts.append(f'<text class="tick" x="{X(i):.1f}" y="{mt + ph + dy}" text-anchor="middle">{esc(x_labels[i])}</text>')
    if x_title:
        parts.append(f'<text class="axis-title" x="{ml + pw / 2:.1f}" y="{height - 6}" text-anchor="middle">{esc(x_title)}</text>')
    right_labels = []
    if ref is not None:
        y = Y(ref[0])
        parts.append(f'<line class="ref" x1="{ml}" x2="{ml + pw}" y1="{y:.1f}" y2="{y:.1f}"/>')
        right_labels.append([y, ref[1], True, "ref-label"])
    parts.append(f'<line class="crosshair" id="{cid}-x" x1="0" x2="0" y1="{mt}" y2="{mt + ph}" style="display:none"/>')
    for s in series:
        pts = [(X(i), Y(v)) for i, v in enumerate(s["values"]) if v is not None and not math.isnan(v)]
        if not pts:
            continue
        d = "M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in pts)
        parts.append(f'<path class="line {s["cls"]}" d="{d}"/>')
        for x, y in pts:
            parts.append(f'<circle class="dot {s["cls"]}" cx="{x:.1f}" cy="{y:.1f}" r="4"/>')
        right_labels.append([pts[-1][1], s.get("short", s["name"]), False, "end-label"])
    # Etiquetas directas a la derecha (series y azar), separadas si chocan
    for y, text, _, cls in _spread_labels(right_labels):
        parts.append(f'<text class="{cls}" x="{ml + pw + 8}" y="{y + 4:.1f}">{esc(text)}</text>')
    # Zonas de hover por posición X (crosshair + tooltip con todas las series)
    for i, lab in enumerate(x_labels):
        lines = [str(lab)]
        for s in series:
            v = s["values"][i]
            if v is not None and not math.isnan(v):
                lines.append(f"{tip_fmt(v)}  {s['name']}")
        if ref is not None:
            lines.append(f"{tip_fmt(ref[0])}  Azar")
        tip = esc("\n".join(lines))
        parts.append(f'<rect class="hit" x="{ml + i * band:.1f}" y="{mt}" width="{band:.1f}" height="{ph}" '
                     f'data-tip="{tip}" data-cross="{cid}-x" data-x="{X(i):.1f}" tabindex="0"/>')
    parts.append("</svg>")
    return "".join(parts)


def svg_dumbbell(rows, x_max, width=760, row_h=30, aria=""):
    """rows: [{"label", "chance", "model", "tip"}] -> azar (gris) a modelo (azul)."""
    ml, mr, mt, mb = 190, 70, 10, 34
    height = mt + mb + row_h * len(rows)
    pw = width - ml - mr

    def X(v):
        return ml + pw * v / x_max

    parts = [f'<svg class="chart" viewBox="0 0 {width} {height}" role="img" aria-label="{esc(aria)}">']
    for t in _nice_ticks(0, x_max, 5):
        x = X(t)
        parts.append(f'<line class="grid" x1="{x:.1f}" x2="{x:.1f}" y1="{mt}" y2="{height - mb}"/>')
        parts.append(f'<text class="tick" x="{x:.1f}" y="{height - mb + 18}" text-anchor="middle">{esc(pct(t, 0))}</text>')
    for i, r in enumerate(rows):
        y = mt + row_h * (i + 0.5)
        xc, xm = X(r["chance"]), X(r["model"])
        parts.append(f'<text class="row-label" x="{ml - 10}" y="{y + 4:.1f}" text-anchor="end">{esc(r["label"])}</text>')
        parts.append(f'<line class="connector" x1="{xc:.1f}" x2="{xm:.1f}" y1="{y:.1f}" y2="{y:.1f}"/>')
        parts.append(f'<circle class="dot ref-dot" cx="{xc:.1f}" cy="{y:.1f}" r="5"/>')
        parts.append(f'<circle class="dot s1" cx="{xm:.1f}" cy="{y:.1f}" r="5"/>')
        parts.append(f'<text class="value-label" x="{xm + 10:.1f}" y="{y + 4:.1f}">{esc(pct(r["model"], 1))}</text>')
        parts.append(f'<rect class="hit" x="{ml}" y="{y - row_h / 2:.1f}" width="{pw}" height="{row_h}" '
                     f'data-tip="{esc(r["tip"])}" tabindex="0"/>')
    parts.append("</svg>")
    return "".join(parts)


def svg_calibration(points, width=420, height=320, aria=""):
    """points: [{"pred", "obs", "n", "label"}]; diagonal = calibración perfecta."""
    ml, mr, mt, mb = 52, 16, 14, 46
    pw, ph = width - ml - mr, height - mt - mb
    hi = max(max(p["pred"] for p in points), max(p["obs"] for p in points)) * 1.1

    def X(v):
        return ml + pw * v / hi

    def Y(v):
        return mt + ph * (1 - v / hi)

    parts = [f'<svg class="chart" viewBox="0 0 {width} {height}" role="img" aria-label="{esc(aria)}">']
    for t in _nice_ticks(0, hi, 5):
        parts.append(f'<line class="grid" x1="{ml}" x2="{ml + pw}" y1="{Y(t):.1f}" y2="{Y(t):.1f}"/>')
        parts.append(f'<text class="tick" x="{ml - 8}" y="{Y(t) + 4:.1f}" text-anchor="end">{esc(pct(t, 0))}</text>')
        parts.append(f'<text class="tick" x="{X(t):.1f}" y="{mt + ph + 18}" text-anchor="middle">{esc(pct(t, 0))}</text>')
    parts.append(f'<line class="ref" x1="{X(0):.1f}" y1="{Y(0):.1f}" x2="{X(hi):.1f}" y2="{Y(hi):.1f}"/>')
    parts.append(f'<text class="axis-title" x="{ml + pw / 2:.1f}" y="{height - 6}" text-anchor="middle">Probabilidad predicha</text>')
    pts = sorted(points, key=lambda p: p["pred"])
    d = "M" + " L".join(f"{X(p['pred']):.1f},{Y(p['obs']):.1f}" for p in pts)
    parts.append(f'<path class="line s1" d="{d}"/>')
    for p in pts:
        parts.append(f'<circle class="dot s1" cx="{X(p["pred"]):.1f}" cy="{Y(p["obs"]):.1f}" r="4"/>')
        tip = esc(f"{pct(p['obs'], 2)} observado\n{pct(p['pred'], 2)} predicho\n{p['label']} · {p['n']:,} casos")
        parts.append(f'<circle class="hit" cx="{X(p["pred"]):.1f}" cy="{Y(p["obs"]):.1f}" r="12" data-tip="{tip}" tabindex="0"/>')
    parts.append("</svg>")
    return "".join(parts)


# ----------------------------------------------------------------
# CSS y JS
# ----------------------------------------------------------------

CSS = """
:root {
  color-scheme: light;
  --page: #f9f9f7; --surface: #fcfcfb; --ink: #0b0b0b; --ink-2: #52514e; --muted: #898781;
  --grid: #e1e0d9; --axis: #c3c2b7; --border: rgba(11,11,11,0.10);
  --s1: #2a78d6; --s2: #eb6834; --s3: #1baf7a; --s1-wash: rgba(42,120,214,0.12);
  --good: #0ca30c; --critical: #d03b3b; --good-bg: #e6f4e6; --critical-bg: #fbe9e9;
  --yes-bg: #e3eefb; --no-bg: #efeee9; --hero-bg: #eef4fc;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    color-scheme: dark;
    --page: #0d0d0d; --surface: #1a1a19; --ink: #ffffff; --ink-2: #c3c2b7; --muted: #898781;
    --grid: #2c2c2a; --axis: #383835; --border: rgba(255,255,255,0.10);
    --s1: #3987e5; --s2: #d95926; --s3: #199e70; --s1-wash: rgba(57,135,229,0.16);
    --good: #0ca30c; --critical: #d03b3b; --good-bg: rgba(12,163,12,0.16); --critical-bg: rgba(208,59,59,0.18);
    --yes-bg: rgba(57,135,229,0.18); --no-bg: #262624; --hero-bg: #16202c;
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --page: #0d0d0d; --surface: #1a1a19; --ink: #ffffff; --ink-2: #c3c2b7; --muted: #898781;
  --grid: #2c2c2a; --axis: #383835; --border: rgba(255,255,255,0.10);
  --s1: #3987e5; --s2: #d95926; --s3: #199e70; --s1-wash: rgba(57,135,229,0.16);
  --good: #0ca30c; --critical: #d03b3b; --good-bg: rgba(12,163,12,0.16); --critical-bg: rgba(208,59,59,0.18);
  --yes-bg: rgba(57,135,229,0.18); --no-bg: #262624; --hero-bg: #16202c;
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--page); color: var(--ink);
  font-family: system-ui, -apple-system, "Segoe UI", sans-serif; font-size: 15px; line-height: 1.5; }
.container { max-width: 1200px; margin: 0 auto; padding: 24px 16px 48px; }
header.top { padding: 8px 0 20px; }
header.top h1 { margin: 0 0 6px; font-size: 26px; }
header.top p { margin: 2px 0; color: var(--ink-2); }
nav.toc { display: flex; flex-wrap: wrap; gap: 8px; margin: 12px 0 0; }
nav.toc a { color: var(--ink-2); text-decoration: none; font-size: 13px; padding: 4px 10px;
  border: 1px solid var(--border); border-radius: 999px; background: var(--surface); }
nav.toc a:hover { color: var(--ink); }
.section { background: var(--surface); border: 1px solid var(--border); border-radius: 14px;
  padding: 22px; margin-bottom: 20px; }
.section h2 { margin: 0 0 6px; font-size: 20px; }
.section h3 { margin: 18px 0 8px; font-size: 16px; }
.section > p.lead, .note { color: var(--ink-2); margin: 0 0 14px; }
.note { font-size: 13px; }
.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 14px; margin-bottom: 20px; }
.tile { background: var(--surface); border: 1px solid var(--border); border-radius: 14px; padding: 16px 18px; }
.tile.hero { background: var(--hero-bg); grid-column: span 2; }
.tile-label { color: var(--ink-2); font-size: 13px; }
.tile-value { font-size: 30px; font-weight: 600; margin-top: 4px; }
.tile.hero .tile-value { font-size: 52px; line-height: 1.1; }
.tile-sub { color: var(--ink-2); font-size: 13px; margin-top: 4px; }
@media (max-width: 640px) { .tile.hero { grid-column: span 1; } .tile.hero .tile-value { font-size: 44px; } }
.grid-2 { display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 18px; }
.grid-3 { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 18px; }
.table-wrap { overflow-x: auto; }
table { width: 100%; border-collapse: collapse; font-size: 14px; }
th, td { padding: 8px 10px; border-bottom: 1px solid var(--grid); text-align: left; vertical-align: top; }
th { color: var(--ink-2); font-weight: 600; font-size: 13px; white-space: nowrap; }
td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; }
td:first-child { min-width: 150px; }
tr.final td { font-weight: 600; }
.animal { font-weight: 600; white-space: nowrap; }
.won { font-weight: 700; text-decoration: underline; text-decoration-color: var(--s1); text-decoration-thickness: 2px; }
.pbar { position: relative; height: 8px; background: var(--grid); border-radius: 4px; margin-top: 4px; min-width: 120px; }
.pbar-fill { position: absolute; left: 0; top: 0; bottom: 0; background: var(--s1); border-radius: 0 4px 4px 0; }
.bar-chance { position: absolute; top: -3px; bottom: -3px; width: 2px; background: var(--ink-2); }
.pill { display: inline-flex; align-items: center; gap: 6px; padding: 2px 10px; border-radius: 999px;
  font-size: 13px; font-weight: 600; white-space: nowrap; color: var(--ink); }
.pill-hit { background: var(--good-bg); } .pill-hit .pill-icon { color: var(--good); }
.pill-miss { background: var(--critical-bg); } .pill-miss .pill-icon { color: var(--critical); }
.pill-yes { background: var(--yes-bg); } .pill-yes .pill-icon { color: var(--s1); }
.pill-no { background: var(--no-bg); } .pill-no .pill-icon { color: var(--muted); }
.legend { display: flex; flex-wrap: wrap; gap: 16px; font-size: 13px; color: var(--ink-2); margin: 4px 0 8px; }
.key { display: inline-flex; align-items: center; gap: 6px; }
.key-line { width: 18px; height: 2px; display: inline-block; }
.key-dot { width: 10px; height: 10px; border-radius: 50%; display: inline-block; }
.key-s1 { background: var(--s1); } .key-s2 { background: var(--s2); } .key-s3 { background: var(--s3); }
.key-ref { background: var(--ink-2); }
svg.chart { width: 100%; height: auto; display: block; }
svg .grid { stroke: var(--grid); stroke-width: 1; }
svg .axis { stroke: var(--axis); stroke-width: 1; }
svg .tick, svg .axis-title { fill: var(--muted); font-size: 12px; }
svg .row-label, svg .value-label, svg .end-label, svg .ref-label { fill: var(--ink-2); font-size: 12px; }
svg .value-label { fill: var(--ink); font-weight: 600; }
svg .ref { stroke: var(--ink-2); stroke-width: 1.5; }
svg .line { fill: none; stroke-width: 2; stroke-linejoin: round; stroke-linecap: round; }
svg .line.s1 { stroke: var(--s1); } svg .line.s2 { stroke: var(--s2); } svg .line.s3 { stroke: var(--s3); }
svg .dot { stroke: var(--surface); stroke-width: 2; }
svg .dot.s1 { fill: var(--s1); } svg .dot.s2 { fill: var(--s2); } svg .dot.s3 { fill: var(--s3); }
svg .dot.ref-dot { fill: var(--muted); }
svg .connector { stroke: var(--axis); stroke-width: 2; }
svg .crosshair { stroke: var(--ink-2); stroke-width: 1; }
svg .hit { fill: transparent; cursor: default; outline: none; }
svg .hit:focus { stroke: var(--s1); stroke-width: 1; }
details.tableview { margin-top: 8px; }
details.tableview summary { cursor: pointer; color: var(--ink-2); font-size: 13px; }
.tooltip { position: fixed; z-index: 1000; display: none; pointer-events: none; max-width: 320px;
  background: var(--ink); color: var(--page); border-radius: 10px; padding: 8px 12px; font-size: 12px;
  box-shadow: 0 8px 24px rgba(0,0,0,0.25); white-space: pre-line; }
.tooltip .tt-title { font-weight: 700; margin-bottom: 2px; }
ul.method li { margin-bottom: 6px; }
code { font-size: 13px; }
"""

JS = """
(function () {
  var tip = document.getElementById('tooltip');
  function place(x, y) {
    var w = tip.offsetWidth, h = tip.offsetHeight;
    var left = x + 14, top = y + 14;
    if (left + w > window.innerWidth - 8) left = x - w - 14;
    if (top + h > window.innerHeight - 8) top = y - h - 14;
    tip.style.left = Math.max(8, left) + 'px';
    tip.style.top = Math.max(8, top) + 'px';
  }
  function show(el, x, y) {
    tip.textContent = '';
    el.getAttribute('data-tip').split('\\n').forEach(function (line, i) {
      var d = document.createElement('div');
      d.textContent = line;
      if (i === 0) d.className = 'tt-title';
      tip.appendChild(d);
    });
    tip.style.display = 'block';
    place(x, y);
    var id = el.getAttribute('data-cross');
    if (id) {
      var ln = document.getElementById(id);
      ln.setAttribute('x1', el.getAttribute('data-x'));
      ln.setAttribute('x2', el.getAttribute('data-x'));
      ln.style.display = 'block';
    }
  }
  function hide(el) {
    tip.style.display = 'none';
    var id = el.getAttribute('data-cross');
    if (id) document.getElementById(id).style.display = 'none';
  }
  document.querySelectorAll('[data-tip]').forEach(function (el) {
    el.addEventListener('pointerenter', function (e) { show(el, e.clientX, e.clientY); });
    el.addEventListener('pointermove', function (e) { place(e.clientX, e.clientY); });
    el.addEventListener('pointerleave', function () { hide(el); });
    el.addEventListener('focus', function () {
      var r = el.getBoundingClientRect(); show(el, r.left + r.width / 2, r.top);
    });
    el.addEventListener('blur', function () { hide(el); });
  });
})();
"""


# ----------------------------------------------------------------
# Secciones
# ----------------------------------------------------------------

def section(sid, title, lead, body):
    lead_html = f'<p class="lead">{lead}</p>' if lead else ""
    return f'<section class="section" id="{sid}"><h2>{esc(title)}</h2>{lead_html}{body}</section>'


def next_turn_section(nt, turn, same, pred):
    cols = []
    for L in LOTTERIES:
        sub = nt.sort_values(f"p_{L}", ascending=False).head(TOP_N)
        mx = max(sub[f"p_{L}"].max(), CHANCE) * 1.05
        rows = []
        for i, r in enumerate(sub.itertuples(), 1):
            p = getattr(r, f"p_{L}")
            rows.append([str(i), f'<span class="animal">{esc(animal_label(r.animal))}</span>',
                         f'{pct(p, 2)}{prob_bar(p, mx, CHANCE)}'])
        when = pred[L]["datetime"].strftime("%Y-%m-%d %H:%M")
        cols.append(f'<div><h3>{esc(LOTTERY_LABEL[L])} · {esc(when)}</h3>'
                    f'{table_html(["#", "Animal", "Probabilidad"], rows, num_cols=(0,))}</div>')
    # Qué apostaría cada estrategia en el próximo sorteo de cada lotería
    rows = []
    for L in LOTTERIES:
        p = pred[L]["probs"]
        cells = [esc(LOTTERY_LABEL[L]), esc(pred[L]["datetime"].strftime("%Y-%m-%d %H:%M"))]
        for key in (MAIN_STRATEGY, ALT_STRATEGY):
            B = strategy_bets(p[None, :], STRATEGY_RULES[key][1], np.array([PAYOUT[L]]))[0]
            picks = [a for a in np.argsort(-p) if B[a]]
            cells.append(", ".join(f"{esc(animal_label(NUM_ANIMAL[a]))} ({pct(p[a], 1)}, EV "
                                   f"{num((p[a] * PAYOUT[L] - 1) * 100, 0)} %)" for a in picks)
                         or "No apostar (ningún animal llega al umbral)")
        rows.append(cells)
    bets = ('<h3>Qué apostaría cada estrategia en el próximo sorteo</h3>'
            + table_html(["Lotería", "Sorteo", STRATEGY_RULES[MAIN_STRATEGY][0], STRATEGY_RULES[ALT_STRATEGY][0]],
                         rows)
            + '<p class="note">EV = valor esperado por unidad apostada según el modelo (p × 30 − 1). '
              'Es una simulación académica, no una recomendación de apostar; ver la sección de estrategia.</p>')
    sub = nt.head(TOP_N)
    mx = max(sub["p_consolidado"].max(), 2 * CHANCE) * 1.05
    rows = []
    for r in sub.itertuples():
        rows.append([str(r.rank), f'<span class="animal">{esc(animal_label(r.animal))}</span>',
                     f'{pct(r.p_consolidado, 2)}{prob_bar(r.p_consolidado, mx, 1 - (1 - CHANCE) ** len(same))}',
                     pct(r.p_lagranjita, 2), pct(r.p_lottoactivo, 2)])
    consolidated = (f'<h3>Consolidado La Granjita + Lotto Activo · {esc(turn.strftime("%Y-%m-%d %H:%M"))} '
                    f'(probabilidad de salir en al menos una de las dos)</h3>'
                    + table_html(["#", "Animal", "Al menos una", "La Granjita", "Lotto Activo"], rows,
                                 num_cols=(0, 3, 4)))
    lead = ("Probabilidades calibradas del modelo para el próximo sorteo de cada lotería. La marca vertical de "
            "cada barra es el azar (2,63 %). Aun el mejor animal tiene una probabilidad baja: el modelo mejora "
            "las probabilidades, no las vuelve seguras.")
    return section("proximo", "🎯 Próximo sorteo", lead,
                   f'<div class="grid-3">{"".join(cols)}</div>{bets}{consolidated}')


def eleven_section(e11, turns, chance_one, h11):
    sub = e11.head(TOP_N)
    mx = max(sub["probability"].max(), chance_one) * 1.05
    rows = [[str(r.rank), f'<span class="animal">{esc(animal_label(r.animal))}</span>',
             f'{pct(r.probability, 1)}{prob_bar(r.probability, mx, chance_one)}'] for r in sub.itertuples()]
    first, last = turns[0][0], turns[-1][0]
    body = table_html(["#", "Animal", "P(sale al menos una vez)"], rows, num_cols=(0,))
    n_sims = f"{N_SIMULATIONS:,}".replace(",", ".")
    body += (f'<p class="note">Turnos simulados: {esc(first.strftime("%Y-%m-%d %H:%M"))} → '
             f'{esc(last.strftime("%Y-%m-%d %H:%M"))} ({n_sims} simulaciones Monte Carlo). '
             f'Un animal cualquiera tiene {pct(chance_one, 1)} de salir al menos una vez en esta ventana '
             f'(marca vertical).</p>')
    if h11 is not None and not h11.empty:
        rate = h11["hit"].mean()
        chance = h11["chance_hit"].mean()
        mh = h11["n_hits"].mean()
        ch = h11["expected_hits_chance"].mean()
        rows = []
        for r in h11.sort_values("datetime", ascending=False).itertuples():
            rows.append([esc(r.date), esc(r.time), esc(label_list(split_names(r.top5))),
                         f"{r.n_hits}/{ELEVEN_TOP}", esc(label_list(split_names(r.hit_animals))),
                         status_pill(r.hit)])
        body += (f'<h3>Histórico: top-{ELEVEN_TOP} de 11 turnos en los últimos {len(h11)} turnos</h3>'
                 f'<p class="note">Acierto {pct(rate, 1)} frente a {pct(chance, 1)} eligiendo {ELEVEN_TOP} '
                 f'animales al azar; aciertos medios {num(mh)} frente a {num(ch)} al azar. Como en 11 turnos '
                 f'salen ~{num(h11["distinct_real"].mean(), 0)} animales distintos, este indicador casi '
                 f'siempre da HIT incluso al azar: no sirve para juzgar el modelo.</p>'
                 + table_html(["Fecha", "Hora", f"Top {ELEVEN_TOP} (11 turnos)", "Aciertos",
                               "Animales acertados", "Resultado"], rows))
    lead = ("Probabilidad de que cada animal salga al menos una vez en los próximos 11 turnos (11 horas, en "
            "cualquiera de las tres loterías). Se calcula simulando miles de futuros posibles sorteo a sorteo "
            "con el modelo, de modo que la anti-repetición y los pares se aplican también dentro de la "
            "simulación.")
    return section("once", "🔮 Próximos 11 turnos", lead, body)


def latest_closed_miss_streaks(turns, hit_col, lengths):
    """Última racha exacta de cada longitud que terminó con un acierto."""
    latest = {}
    streak = None  # Hasta el primer acierto no sabemos dónde empezó la racha.
    last_miss = None
    for i, (ts, hit) in enumerate(turns[["datetime", hit_col]].itertuples(index=False, name=None), 1):
        if hit:
            if streak in lengths:
                latest[streak] = (last_miss, ts, len(turns) - i)
            streak = 0
        elif streak is not None:
            streak += 1
            last_miss = ts
    return latest


def miss_streaks_html(cons):
    lengths = range(3, 11)
    turns = cons.sort_values("datetime")
    top10 = latest_closed_miss_streaks(turns, "hit", lengths)
    top7 = latest_closed_miss_streaks(turns, "hit_top7", lengths)

    def cell(latest, length):
        if length not in latest:
            return "No se registró"
        miss, hit, ago = latest[length]
        return (f"Último fallo: {miss:%d/%m/%Y %H:%M}<br>"
                f"Acierto: {hit:%d/%m/%Y %H:%M} · hace {ago} {'turno' if ago == 1 else 'turnos'}")

    rows = [[str(n), cell(top10, n), cell(top7, n)] for n in lengths]
    return (
        '<h3 id="rachas-fallos">Últimas rachas de fallos cerradas con un acierto</h3>'
        f'<p class="note">Última racha de cada longitud exacta en el backtest de {EVAL_DAYS} días. '
        'Por ejemplo, 5 fallos seguidos y un acierto cuentan como una racha de 5, no de 4. '
        'No se cuentan rachas incompletas. La distancia se mide en turnos consolidados posteriores al acierto.</p>'
        + table_html(["Fallos seguidos", "Top 10 consolidado", f"Top {TOP_PER_LOTTERY}×2"], rows, num_cols=(0,))
    )


def today_section(bt, cons, day):
    tb = bt[bt["datetime"].dt.normalize() == day].sort_values(["datetime", "lottery"], ascending=[False, True])
    tc = cons[cons["datetime"].dt.normalize() == day].sort_values("datetime", ascending=False)
    if tb.empty:
        return section("hoy", "📊 Último día con resultados", "No hay sorteos registrados.", "")
    rows = []
    for r in tb.itertuples():
        top = split_names(r.top10)
        cells = []
        for i, a in enumerate(top):
            t = esc(animal_label(a))
            cells.append(f'<span class="won">{t}</span>' if a == r.real_animal else t)
        rows.append([esc(r.datetime.strftime("%H:%M")), esc(LOTTERY_LABEL[r.lottery]), ", ".join(cells),
                     f'<span class="animal">{esc(animal_label(r.real_animal))}</span>', f"{r.rank}",
                     status_pill(r.hit_top5 >= 0.5), status_pill(r.hit_top10 >= 0.5)])
    per_lot = table_html(["Hora", "Lotería", "Top 10 predicho (en orden)", "Resultado", "Posición",
                          "Top 5", "Top 10"], rows, num_cols=(4,))
    rows = []
    for r in tc.itertuples():
        rows.append([esc(r.time), esc(label_list(split_names(r.top10))), esc(label_list(split_names(r.real_animals))),
                     esc(label_list(split_names(r.hit_animals))) or "—", status_pill(r.hit),
                     esc(label_list(split_names(r.top_list))), status_pill(r.hit_top7)])
    cons_tbl = table_html(["Hora", "Top 10 consolidado", "Resultados", "Coincidencia", "Top 10",
                           f"Top {TOP_PER_LOTTERY} de cada lotería (unidos)", f"Top {TOP_PER_LOTTERY}×2"], rows)
    n = len(tb)
    k10 = tb["hit_top10"].sum()
    k5 = tb["hit_top5"].sum()
    lead = (f"Predicciones hechas ANTES de cada sorteo (sin usar su resultado). {esc(day.strftime('%Y-%m-%d'))}: "
            f"top 10 acertó {num(k10, 0)} de {n} sorteos (azar esperado {num(n * 10 / 38, 1)}), "
            f"top 5 acertó {num(k5, 0)} (azar {num(n * 5 / 38, 1)}). Un solo día es muy poca muestra: "
            "lo relevante es el backtest de 365 días.")
    body = (f"<h3>Por lotería</h3>{per_lot}<h3>Consolidado por turno (La Granjita + Lotto Activo)</h3>{cons_tbl}"
            f"{miss_streaks_html(cons)}")
    return section("hoy", f"📊 Último día con resultados · {day.strftime('%Y-%m-%d')}", lead, body)


def performance_section(ctx):
    s = ctx["topk"]
    bt = ctx["bt"]
    cons = ctx["cons"]
    body = []
    # Tabla top-K
    rows = []
    for r in s.itertuples():
        name = "Todas" if r.lottery == "todas" else LOTTERY_LABEL[r.lottery]
        rows.append([esc(name), f"Top {r.K}", f"{r.n:,}".replace(",", "."), pct(r.rate, 2),
                     f"{pct(r.ci_low, 1)} – {pct(r.ci_high, 1)}", pct(r.chance, 2), f"×{num(r.lift)}",
                     num(r.z, 1), p_text(r.p_value)])
    body.append(table_html(["Lotería", "Ranking", "Sorteos", "Acierto", "IC 95 %", "Azar", "Mejora",
                            "z", "p-valor"], rows, num_cols=(2, 3, 4, 5, 6, 7, 8)))
    # Dumbbell
    drows = []
    for r in s[s["lottery"] != "todas"].itertuples():
        lab = f"Top {r.K} · {LOTTERY_LABEL[r.lottery]}"
        drows.append({"label": lab, "chance": r.chance, "model": r.rate,
                      "tip": f"{pct(r.rate, 2)} modelo\n{pct(r.chance, 2)} azar\n{lab}"})
    body.append('<h3>Modelo frente al azar</h3><div class="legend"><span class="key"><span class="key-dot key-s1"></span>'
                'Modelo (fuera de muestra)</span><span class="key"><span class="key-dot key-ref"></span>Azar (K/38)</span></div>')
    body.append(svg_dumbbell(drows, x_max=max(0.45, s["rate"].max() * 1.1),
                             aria="Tasa de acierto del modelo frente al azar por ranking y lotería"))
    # Evolución semanal
    wk = bt.copy()
    wk["week"] = wk["datetime"].dt.to_period("W-SUN").dt.start_time
    g = wk.groupby("week").agg(hits=("hit_top10", "sum"), n=("hit_top10", "size")).reset_index()
    g["rate"] = g["hits"] / g["n"]
    g["roll"] = g["hits"].rolling(4, min_periods=1).sum() / g["n"].rolling(4, min_periods=1).sum()
    labels = [d.strftime("%d/%m/%y") for d in g["week"]]
    chart = svg_line_chart(
        labels, [{"name": "Modelo (media 4 sem.)", "values": list(g["roll"]), "cls": "s1"}],
        y_min=0.15, y_max=0.55, ref=(10 / 38, "Azar 26,3 %"),
        aria="Tasa de acierto top 10 por semana, media móvil de 4 semanas",
        show_every=max(1, len(labels) // 8), x_title="Semana")
    body.append(f"<h3>Acierto top 10 semana a semana (media móvil de 4 semanas, todas las loterías)</h3>{chart}")
    body.append(data_details(["Semana", "Sorteos", "Aciertos top 10", "Tasa semanal", "Media 4 semanas"],
                             [[esc(l), str(int(r.n)), num(r.hits, 0), pct(r.rate, 1), pct(r.roll, 1)]
                              for l, r in zip(labels, g.itertuples())], num_cols=(1, 2, 3, 4)))
    # Consolidado
    c10, ch10 = cons["hit"].mean(), cons["chance_top10"].mean()
    c7, ch7 = cons["hit_top7"].mean(), cons["chance_top7"].mean()
    rows = [["Top 10 consolidado (P de salir en al menos una lotería)", f"{len(cons):,}".replace(",", "."),
             pct(c10, 1), pct(ch10, 1), pct(cons["q_top10"].mean(), 1)],
            [f"Top {TOP_PER_LOTTERY} de cada lotería unidos (hasta {2 * TOP_PER_LOTTERY} animales)",
             f"{len(cons):,}".replace(",", "."), pct(c7, 1), pct(ch7, 1), "—"]]
    body.append("<h3>Por turno en La Granjita + Lotto Activo (acierta si sale en cualquiera de las dos)</h3>")
    body.append(table_html(["Estrategia", "Turnos", "Acierto", "Azar con el mismo nº de animales",
                            "Esperado por el modelo"], rows, num_cols=(1, 2, 3, 4)))
    # Ablación
    rows = []
    for r in ctx["ablation"]:
        cls_final = r["key"] == FINAL_MODEL
        rows.append([("<strong>" if cls_final else "") + esc(MODEL_LABELS[r["key"]]) + (" ✓ (usado)" if cls_final else "") + ("</strong>" if cls_final else ""),
                     pct(r["top1"], 2), pct(r["top5"], 1), pct(r["top10"], 1),
                     num(r["gain"] * 1000, 0) if r["gain"] is not None else "0"])
    body.append("<h3>¿Qué aporta cada patrón? (modelos anidados, mismos 365 días)</h3>")
    body.append(table_html(["Modelo", "Top 1", "Top 5", "Top 10", "Información (milibits/sorteo)"], rows,
                           num_cols=(1, 2, 3, 4)))
    body.append('<p class="note">Cada fila añade un patrón al modelo anterior y se re-evalúa fuera de muestra. '
                'La información (log-verosimilitud frente al azar) es la medida más sensible: si un patrón no la '
                'sube, no se usa.</p>')
    # Calibración
    P = np.vstack(bt["probs"].to_numpy())
    Y = np.zeros_like(P, dtype=bool)
    Y[np.arange(len(bt)), bt["winner"].to_numpy()] = True
    edges = [0, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.05, 1.0]
    pts = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (P >= lo) & (P < hi)
        if m.sum() >= 500:
            pts.append({"pred": float(P[m].mean()), "obs": float(Y[m].mean()), "n": int(m.sum()),
                        "label": f"{pct(lo, 1)}–{pct(hi, 1)}"})
    cal = svg_calibration(pts, aria="Calibración: probabilidad predicha frente a frecuencia observada")
    cal_tbl = data_details(["Rango predicho", "Casos", "Predicho medio", "Observado"],
                           [[esc(p["label"]), f"{p['n']:,}".replace(",", "."), pct(p["pred"], 2), pct(p["obs"], 2)]
                            for p in pts], num_cols=(1, 2, 3))
    body.append(f'<div class="grid-2"><div><h3>Calibración</h3>{cal}{cal_tbl}</div>'
                f'<div><h3>¿Qué significa?</h3><p class="note">Si el modelo dice "4 %", ese animal debería salir '
                f'~4 de cada 100 veces. Los puntos sobre la diagonal indican probabilidades honestas. '
                f'Ganancia de información media: <strong>{num(ctx["gain_bits"] * 1000, 0)} milibits por sorteo</strong> '
                f'(0 sería azar).</p><p class="note">Los porcentajes de esta sección son fuera de muestra: cada '
                f'sorteo se predijo con un modelo entrenado solo con datos anteriores (re-entrenado cada '
                f'{REFIT_DAYS} días).</p></div></div>')
    lead = (f"Backtest walk-forward de los últimos {EVAL_DAYS} días: el modelo nunca ve el resultado que "
            "predice. Se compara con elegir animales al azar (K/38) e incluye intervalos de confianza "
            "del 95 % (Wilson) y una prueba z frente al azar.")
    return section("rendimiento", "📈 Rendimiento real del modelo", lead, "".join(body))


def patterns_section(ctx):
    pt = ctx["patterns"]
    rows = []
    for r in pt.itertuples():
        lot = "Todas" if r.loteria == "todas" else LOTTERY_LABEL.get(r.loteria, r.loteria)
        stat = num(r.estadistico, 2) if r.estadistico is not None and not (isinstance(r.estadistico, float) and math.isnan(r.estadistico)) else "—"
        gl = "" if r.gl is None or (isinstance(r.gl, float) and math.isnan(r.gl)) else f" (gl {int(r.gl)})"
        p = r.p_valor if r.p_valor is not None and not (isinstance(r.p_valor, float) and math.isnan(r.p_valor)) else None
        rows.append([esc(r.categoria), esc(r.prueba), esc(lot), stat + gl, p_text(p),
                     esc(r.efecto), verdict_pill(r.veredicto)])
    body = [table_html(["Tipo", "Hipótesis", "Lotería", "Estadístico", "p-valor", "Efecto observado", "Veredicto"],
                       rows, num_cols=(3, 4))]
    body.append('<p class="note">"Se cumple" = el patrón es estadísticamente significativo (p &lt; 0,05). '
                'Con miles de sorteos, incluso efectos pequeños son significativos; por eso se mide además '
                'cuánto mejoran la tasa de aciertos (sección de rendimiento).</p>')
    # Curva de riesgo
    hz = ctx["hazard"]
    lots = [L for L in LOTTERIES if (hz["lottery"] == L).any()]
    classes = dict(zip(LOTTERIES, ("s1", "s2", "s3")))
    buckets = list(hz[hz["lottery"] == lots[0]]["bucket"])
    series = [{"name": LOTTERY_LABEL[L], "values": list(hz[hz["lottery"] == L]["rate"]), "cls": classes[L]}
              for L in lots]
    chart = svg_line_chart(buckets, series, 0.0, 0.04, ref=(CHANCE, "Azar 2,63 %"),
                           aria="Probabilidad de salir según los sorteos que lleva sin salir",
                           x_title="Sorteos que lleva sin salir (misma lotería)")
    legend = ('<div class="legend">' + "".join(
        f'<span class="key"><span class="key-line key-{classes[L]}"></span>{esc(LOTTERY_LABEL[L])}</span>'
        for L in lots) + '<span class="key"><span class="key-line key-ref"></span>Azar</span></div>')
    tbl_rows = []
    for b in buckets:
        cells = [esc(b)]
        for L in lots:
            r = hz[(hz["lottery"] == L) & (hz["bucket"] == b)].iloc[0]
            cells += [pct(r["rate"], 2), num(r["z"], 1)]
        tbl_rows.append(cells)
    headers = ["Sorteos sin salir"] + [h for L in lots for h in (LOTTERY_LABEL[L], "z")]
    body.append(f"<h3>Anti-repetición: probabilidad de salir según cuánto lleva sin salir</h3>{legend}{chart}")
    body.append(data_details(headers, tbl_rows, num_cols=tuple(range(1, len(headers)))))
    body.append('<p class="note">Un animal que acaba de salir casi nunca repite (≈0,7–0,9 % frente a 2,63 %). '
                'Los que llevan 13–50 sorteos sin salir son los más probables (~3,0–3,4 %). Ojo: los muy '
                '"atrasados" (151+) salen MENOS que el azar, así que la idea de que "le toca" es falsa a '
                'largo plazo.</p>')
    # Entre loterías: dos gráficos (Activo <-> Granjita y Activo <-> Activo Int)
    lags = ctx["lags"]
    for (a, b), title, note in [
        (("lottoactivo", "lagranjita"), "Lotto Activo ↔ La Granjita",
         "El efecto es asimétrico: La Granjita tiende a repetir, 2–6 horas después, el animal que salió en "
         "Lotto Activo; al revés casi no ocurre."),
        (("lottoactivoint", "lottoactivo"), "Lotto Activo Int ↔ Lotto Activo",
         "Lotto Activo Int sortea a las hh:30, entre dos sorteos de Lotto Activo: k = 1 es el sorteo de la "
         "otra lotería de 30 minutos antes."),
    ]:
        s1 = lags[(lags["source"] == a) & (lags["target"] == b)].sort_values("lag")
        s2 = lags[(lags["source"] == b) & (lags["target"] == a)].sort_values("lag")
        if s1.empty or s2.empty:
            continue
        xs = [str(k) for k in sorted(set(s1["lag"]) | set(s2["lag"]))]
        v1 = dict(zip(s1["lag"].astype(str), s1["rate"]))
        v2 = dict(zip(s2["lag"].astype(str), s2["rate"]))
        z1 = dict(zip(s1["lag"].astype(str), s1["z"]))
        z2 = dict(zip(s2["lag"].astype(str), s2["z"]))
        n1 = f"{LOTTERY_LABEL[b]} repite {LOTTERY_LABEL[a]}"
        n2 = f"{LOTTERY_LABEL[a]} repite {LOTTERY_LABEL[b]}"
        lo = min(min(v1.values()), min(v2.values()))
        hi = max(max(v1.values()), max(v2.values()))
        short = {"lagranjita": "Granjita", "lottoactivo": "Activo", "lottoactivoint": "Int"}
        chart = svg_line_chart(xs, [
            {"name": n1, "short": f"{short[b]} repite {short[a]}", "values": [v1.get(x) for x in xs], "cls": "s1"},
            {"name": n2, "short": f"{short[a]} repite {short[b]}", "values": [v2.get(x) for x in xs], "cls": "s2"}],
            min(0.02, math.floor(lo * 200) / 200), max(0.04, math.ceil(hi * 200) / 200),
            ref=(float(s1["chance"].iloc[0]), "Azar 2,6 %"), y_fmt=lambda v: pct(v, 1),
            aria=f"Probabilidad de que una lotería repita el animal que salió en la otra k sorteos antes: {title}",
            x_title="k = sorteos de diferencia (0 = misma hora)")
        legend = (f'<div class="legend"><span class="key"><span class="key-line key-s1"></span>{esc(n1)} '
                  f'(k sorteos antes)</span><span class="key"><span class="key-line key-s2"></span>{esc(n2)}</span>'
                  '<span class="key"><span class="key-line key-ref"></span>Azar</span></div>')
        body.append(f"<h3>Efecto entre loterías: {esc(title)}</h3>{legend}{chart}")
        body.append(data_details(["k", n1, "z", n2, "z"],
                                 [[x, pct(v1[x], 2) if x in v1 else "—", num(z1[x], 1) if x in z1 else "—",
                                   pct(v2[x], 2) if x in v2 else "—", num(z2[x], 1) if x in z2 else "—"]
                                  for x in xs], num_cols=(1, 2, 3, 4)))
        body.append(f'<p class="note">{esc(note)}</p>')
    # Pares
    pr = ctx["pairs"]
    sig = pr[(pr["p_bonferroni"] < 0.05) & (pr["z"] < 0)].head(20)
    rows = [[f'<span class="animal">{esc(r.par)}</span>', num(r.observado, 0), num(r.esperado, 0),
             pct(r.ratio, 0), num(r.z, 1), p_text(r.p_bonferroni), num(r.z_1a_mitad, 1), num(r.z_2a_mitad, 1),
             "Sí" if r.replica else "No"] for r in sig.itertuples()]
    body.append("<h3>Pares que se evitan el mismo día (los 20 más fuertes)</h3>")
    body.append(table_html(["Par", "Días juntos", "Esperado", "Ratio", "z", "p (Bonferroni)", "z 1ª mitad",
                            "z 2ª mitad", "Se replica"], rows, num_cols=(1, 2, 3, 4, 5, 6, 7)))
    attracted = pr[(pr["p_bonferroni"] < 0.05) & (pr["z"] > 0)]
    top_pos = pr.iloc[-1]
    body.append(f'<p class="note">Ningún par sale junto más de lo esperado ({len(attracted)} significativos tras '
                f'Bonferroni; el mayor exceso es {esc(top_pos["par"])} con {pct(top_pos["ratio"], 0)} de lo '
                f'esperado, compatible con el azar). En cambio, muchos pares "semánticos" se evitan: toro~vaca, '
                f'ratón~elefante, zorro~gallina, oso~pescado, ratón~gato, león~tigre... y el patrón se repite en '
                f'las dos mitades del histórico. El archivo <code>pares.csv</code> tiene los 703 pares.</p>')
    lead = ("Cada fila es una hipótesis contrastada con todo el histórico frente a la hipótesis nula de sorteos "
            "independientes y uniformes. Los patrones que se cumplen se incorporan al modelo; los que no, se descartan.")
    return section("patrones", "🔎 Patrones: ¿se cumplen o no?", lead, "".join(body))


def strategy_section(ctx):
    st = ctx["strategies"]
    ev = st[st["period"] == "evaluacion"].set_index("strategy")
    sel = st[st["period"] == "seleccion"].set_index("strategy")
    m = ev.loc[MAIN_STRATEGY]
    payout = PAYOUT[LOTTERIES[0]]
    tk = ctx["topk"]
    top1 = float(tk[(tk["lottery"] == "todas") & (tk["K"] == 1)]["rate"].iloc[0])
    body = []
    body.append(
        '<div class="grid-2"><div><h3>La cuenta del valor esperado</h3>'
        f'<p class="note">Cada lotería paga <strong>{num(payout, 0)} veces</strong> lo apostado. Si apuestas 1 '
        f'unidad a un animal con probabilidad <em>p</em> de salir, el valor esperado es '
        f'<strong>EV = p × {num(payout, 0)} − 1</strong>.</p>'
        f'<ul class="method"><li>Al azar, p = 1/38 = 2,63 % → EV = {num((payout / 38 - 1) * 100, 1).replace("-", "−")} % '
        f'(la ventaja de la casa).</li>'
        f'<li>Para no perder hace falta p &gt; 1/{num(payout, 0)} = {pct(1 / payout, 2)}.</li>'
        f'<li>El top 1 del modelo acierta ≈ {pct(top1, 2)} → EV ≈ {num((top1 * payout - 1) * 100, 0)} % '
        'por unidad.</li></ul></div>'
        '<div><h3>Cómo se eligió la estrategia (sin hacer trampa)</h3>'
        '<p class="note">Todas las reglas se evaluaron con las predicciones fuera de muestra del backtest. La '
        'regla principal se eligió mirando SOLO el año anterior (la de mejor cota inferior del IC 95 % del '
        'ROI) y sus números del último año no se usaron para elegirla. Las apuestas son de 1 unidad fija por '
        'animal: «1 u» es lo que decidas apostar por animal.</p></div></div>')
    rows = []
    for key, (label, _) in STRATEGY_RULES.items():
        a, b = sel.loc[key] if key in sel.index else None, ev.loc[key]
        strong = key == MAIN_STRATEGY
        name = (f"<strong>{esc(label)} ✓ (principal)</strong>" if strong else esc(label))
        rows.append([
            name,
            (f"{pct(a['roi'], 1)} ({pct(a['roi_lo'], 0)} a {pct(a['roi_hi'], 0)})" if a is not None else "—"),
            f"<strong>{pct(b['roi'], 1)}</strong> ({pct(b['roi_lo'], 0)} a {pct(b['roi_hi'], 0)})",
            num(b["stake"] / max(b["draws_bet"], 1), 1),
            pct(b["draws_bet"] / max(b["draws"], 1), 0),
            pct(b["hit_rate"], 1),
            intf(b["profit"]),
            intf(b["max_drawdown"]),
            str(int(b["losing_streak"])),
            f"{pct(b['pct_weeks_up'], 0)} / {pct(b['pct_months_up'], 0)}",
        ])
    body.append('<h3>Resultados de cada regla (pago 30×, 1 unidad por animal)</h3>')
    body.append(table_html(
        ["Regla", "ROI año anterior (IC 95 %)", "ROI último año (IC 95 %)", "Apuestas por sorteo",
         "Sorteos con apuesta", "Sorteos ganados", "Ganancia último año (u)", "Caída máx. (u)",
         "Racha sin ganar (sorteos)", "Semanas / meses en ganancia"],
        rows, num_cols=(1, 2, 3, 4, 5, 6, 7, 8, 9)))
    # Curva de ganancia acumulada (último año)
    series = []
    labels = None
    for key, cls in ((MAIN_STRATEGY, "s1"), (ALT_STRATEGY, "s2")):
        led = ctx["ledgers"][key]
        weekly = led.set_index("datetime")["profit"].resample("W-SUN").sum().cumsum()
        if labels is None:
            idx = weekly.index
            labels = [d.strftime("%d/%m/%y") for d in idx]
        weekly = weekly.reindex(idx).ffill().fillna(0.0)
        short = {"valor30": "Valor ≥ +30 %"}.get(key, STRATEGY_RULES[key][0].replace(" por sorteo", ""))
        series.append({"name": STRATEGY_RULES[key][0].split(" (")[0], "short": short,
                       "values": list(weekly.values), "cls": cls})
    lo = min(min(s["values"]) for s in series)
    hi = max(max(s["values"]) for s in series)
    span = hi - min(lo, 0)
    step = 10 ** math.floor(math.log10(max(span, 1)))
    units = lambda v: f"{v:,.0f}".replace(",", ".") + " u"
    chart = svg_line_chart(labels, series, min(0, math.floor(lo / step) * step), math.ceil(hi / step) * step,
                           ref=(0.0, "Sin ganar ni perder"), y_fmt=units, tip_fmt=units,
                           aria="Ganancia acumulada de las estrategias en el último año",
                           show_every=max(1, len(labels) // 8), x_title="Semana (fin de semana)")
    legend = ('<div class="legend">' + "".join(
        f'<span class="key"><span class="key-line key-{s["cls"]}"></span>{esc(s["name"])}</span>' for s in series)
        + '<span class="key"><span class="key-line key-ref"></span>0 u</span></div>')
    body.append(f'<h3>Ganancia acumulada en el último año (unidades)</h3>{legend}{chart}')
    # Riesgo: Monte Carlo
    sim = ctx["bankroll"]
    if sim is not None and not sim.empty:
        rows = [[f"{int(r.dias)} días", intf(r.apostado), pct(r.p_perder, 1), intf(r.ganancia_mediana),
                 f"{intf(r.ganancia_p5)} a {intf(r.ganancia_p95)}", intf(r.caida_max_p95)]
                for r in sim.itertuples()]
        body.append(f'<h3>¿Qué puede pasar? Simulación Monte Carlo de «{esc(STRATEGY_RULES[MAIN_STRATEGY][0])}»</h3>')
        body.append(table_html(["Horizonte", "Apostado (u)", "P(terminar perdiendo)", "Ganancia mediana (u)",
                                "Rango 90 % (u)", "Caída máx. (P95, u)"], rows, num_cols=(1, 2, 3, 4, 5)))
        body.append(f'<p class="note">{num(5000, 0)} trayectorias re-muestreando días reales del último año. '
                    'Supone que el futuro se parece al último año, cosa que no está garantizada.</p>')
    # ROI por lotería
    pl = ctx["strategy_per_lottery"]
    if pl is not None and not pl.empty:
        rows = [[esc(STRATEGY_RULES[r.strategy][0]), esc(LOTTERY_LABEL[r.lottery]), intf(r.stake),
                 intf(r.ret), pct(r.roi, 1)] for r in pl.itertuples()]
        body.append("<h3>Por lotería (último año)</h3>")
        body.append(table_html(["Regla", "Lotería", "Apostado (u)", "Cobrado (u)", "ROI"], rows,
                               num_cols=(2, 3, 4)))
    p_bet = ctx.get("kelly_p", 0.045)
    body.append(
        '<h3>Tamaño de la apuesta y advertencias</h3><ul class="method">'
        f'<li><strong>Criterio de Kelly:</strong> fracción del saldo por apuesta f* = (p·{num(payout, 0)} − 1)/'
        f'{num(payout - 1, 0)}. Con p ≈ {pct(p_bet, 1)} sale f* ≈ {pct(kelly_fraction(p_bet, payout), 2)} del '
        'saldo; con la incertidumbre del modelo conviene usar como mucho ¼ de Kelly.</li>'
        f'<li><strong>Varianza:</strong> la peor racha sin ganar del último año fue de {int(m["losing_streak"])} '
        f'sorteos y la caída máxima de {intf(m["max_drawdown"])} u. Hace falta un saldo varias veces mayor '
        'que esa caída para aguantarla.</li>'
        '<li><strong>Riesgo de modelo:</strong> los patrones cambian con los años (por eso el modelo pesa más '
        'lo reciente) y el operador puede cambiar su mecanismo en cualquier momento; si lo hace, el ROI '
        'pasaría a ser el del azar: −21 %.</li>'
        '<li><strong>Condiciones reales:</strong> el cálculo supone pago íntegro de 30×, sin comisiones, sin '
        'topes por animal y apostando antes de cada sorteo con el resultado anterior ya publicado.</li>'
        '<li><strong>Esto es un ejercicio académico de probabilidad</strong>, no asesoría financiera ni una '
        'recomendación de apostar. Nunca apuestes dinero que no puedas perder.</li></ul>')
    lead = (f"¿Se puede ganar con el modelo? Con pago 30×, la regla principal ({esc(STRATEGY_RULES[MAIN_STRATEGY][0])}) "
            f"tuvo un ROI de <strong>{pct(m['roi'], 1)}</strong> en el último año (IC 95 % {pct(m['roi_lo'], 1)} a "
            f"{pct(m['roi_hi'], 1)}), fuera de muestra. Debajo están el resto de reglas, el riesgo y las condiciones.")
    return section("estrategia", "💰 Estrategia de apuesta: valor esperado y riesgo", lead, "".join(body))


def hitmiss_payload(bt, cons, days=90):
    """Datos compactos (JSON) para el gráfico HIT/MISS interactivo de los últimos `days` días."""
    start = bt["datetime"].max().normalize() - timedelta(days=days - 1)
    b = bt[bt["datetime"] >= start]
    P = np.vstack(b["probs"].to_numpy()) if len(b) else np.zeros((0, N_ANIMALS))
    order = np.argsort(-P, axis=1)[:, :TOP_N]
    day_list = sorted({d.strftime("%Y-%m-%d") for d in b["datetime"]})
    day_ix = {d: i for i, d in enumerate(day_list)}

    def when(ts):
        # [índice del día, minuto del día] -> más compacto que el texto de la fecha
        return day_ix[ts.strftime("%Y-%m-%d")], ts.hour * 60 + ts.minute

    draws = []
    for i, r in enumerate(b.itertuples()):
        top = [int(a) for a in order[i]]
        draws.append([*when(r.datetime), LOTTERIES.index(r.lottery), int(r.winner), top,
                      [int(round(float(P[i, a]) * 1e5)) for a in top[:3]]])
    c = cons[cons["datetime"] >= start]
    turns = [[*when(r.datetime), [ANIMAL_NUM[a] for a in split_names(r.top10)],
              [ANIMAL_NUM[a] for a in split_names(r.real_animals)]] for r in c.itertuples()]
    return {
        "animals": [animal_label(NUM_ANIMAL[i]) for i in range(N_ANIMALS)],
        "lotteries": [LOTTERY_LABEL[L] for L in LOTTERIES],
        "payout": [PAYOUT[L] for L in LOTTERIES],
        "valueMargin": STRATEGY_RULES["valor30"][1]["margin"],
        "valueMax": STRATEGY_RULES["valor30"][1]["kmax"],
        "days": day_list, "draws": draws, "turns": turns,
    }


def hitmiss_section(ctx):
    payload = json.dumps(hitmiss_payload(ctx["bt"], ctx["cons"]), separators=(",", ":"))
    payload = payload.replace("<", "\\u003c")
    options = [("cons", "Top 10 consolidado por turno (La Granjita + Lotto Activo)")]
    options += [(f"top10_{i}", f"Top 10 · {LOTTERY_LABEL[L]}") for i, L in enumerate(LOTTERIES)]
    options += [("strat_top2", f"Estrategia: {STRATEGY_RULES['top2'][0]} (todas las loterías)"),
                ("strat_value", f"Estrategia: {STRATEGY_RULES['valor30'][0]}")]
    opts = "".join(f'<option value="{k}">{esc(v)}</option>' for k, v in options)
    body = (
        '<div class="hm-controls"><label for="hmSelect">Mostrar:</label>'
        f'<select id="hmSelect">{opts}</select></div>'
        '<div class="chart-head"><div class="legend">'
        '<span class="key"><span class="key-dot key-hit"></span>HIT</span>'
        '<span class="key"><span class="key-dot key-miss"></span>MISS</span>'
        '<span class="key"><span class="key-line key-s1"></span>Tendencia (media móvil de 25)</span>'
        '</div><div class="chart-summary" id="hmSummary"></div></div>'
        '<div class="chart-scroll" id="hmScroll" tabindex="0"><div class="chart-canvas" id="hmCanvas"></div></div>'
        '<div class="chart-detail" id="hmDetail"><em>Haz clic en una barra para ver el detalle completo.</em></div>'
        f'<script type="application/json" id="hmData">{payload}</script>')
    lead = ("Cada barra es un turno (o un sorteo) de los últimos 90 días del backtest: verde = HIT, rojo = MISS. "
            "Desplázate horizontalmente, pasa el cursor sobre una barra para ver sus datos y haz clic para fijarlos "
            "abajo. El resumen compara la tasa de acierto con la del azar y, en las estrategias, muestra lo "
            "apostado y lo ganado. Los datos completos están en backtest.csv.")
    return section("evolucion", "📈 Evolución de HIT / MISS — últimos 3 meses", lead, body)


HITMISS_CSS = """
.hm-controls { display: flex; align-items: center; gap: 10px; margin: 0 0 12px; flex-wrap: wrap; }
.hm-controls select { font: inherit; padding: 6px 10px; border-radius: 8px; border: 1px solid var(--axis);
  background: var(--surface); color: var(--ink); max-width: 100%; }
.chart-head { display: flex; flex-wrap: wrap; align-items: center; gap: 14px; margin-bottom: 12px; }
.chart-summary { margin-left: auto; font-size: 13px; color: var(--ink-2); background: var(--no-bg);
  padding: 6px 12px; border-radius: 999px; }
.key-hit { background: var(--good); border-radius: 3px; } .key-miss { background: var(--critical); border-radius: 3px; }
.chart-scroll { overflow-x: auto; overflow-y: hidden; border: 1px solid var(--border); border-radius: 12px;
  padding: 16px 12px 8px; background: var(--surface); }
.chart-canvas { position: relative; min-width: 100%; }
.bar-col { position: absolute; display: flex; align-items: flex-end; justify-content: center; cursor: pointer; z-index: 1; }
.bar-rect { border-radius: 4px 4px 0 0; transition: filter .12s ease; }
.bar-hit { background: linear-gradient(180deg, #34d399 0%, var(--good) 100%); }
.bar-miss { background: linear-gradient(180deg, #f3a3a3 0%, var(--critical) 100%); }
.bar-col:hover .bar-rect { filter: brightness(1.15); }
.bar-col.selected .bar-rect { outline: 2px solid var(--ink); outline-offset: 1px; }
.day-line { position: absolute; width: 1px; background: var(--grid); z-index: 0; }
.day-label { position: absolute; bottom: 2px; font-size: 10px; color: var(--muted); white-space: nowrap; }
.trend-svg { position: absolute; left: 0; z-index: 2; pointer-events: none; }
.trend-line { fill: none; stroke: var(--s1); stroke-width: 2; stroke-linejoin: round; stroke-linecap: round; }
.chance-line { stroke: var(--ink-2); stroke-width: 1; }
.chart-detail { margin-top: 12px; min-height: 36px; }
.chart-detail-card { border-radius: 12px; padding: 14px 16px; max-width: 680px; border: 1px solid var(--border); }
.detail-hit { background: var(--good-bg); } .detail-miss { background: var(--critical-bg); }
.chart-detail-head { display: flex; align-items: center; gap: 12px; margin-bottom: 8px; }
.tt-table { border-collapse: collapse; width: 100%; font-size: 12px; }
.tt-table td { padding: 3px 8px 3px 0; border: 0; vertical-align: top; }
.tt-key { opacity: .7; white-space: nowrap; }
"""


HITMISS_JS = """
(function () {
  var dataEl = document.getElementById('hmData');
  if (!dataEl) return;
  var D = JSON.parse(dataEl.textContent);
  var A = D.animals, LOT = D.lotteries;
  var scroll = document.getElementById('hmScroll'), canvas = document.getElementById('hmCanvas');
  var detail = document.getElementById('hmDetail'), summary = document.getElementById('hmSummary');
  var select = document.getElementById('hmSelect'), tip = document.getElementById('tooltip');
  var SLOT = 16, BAR_W = 12, AREA = 190, LABEL_H = 34, HIT_H = 160, MISS_H = 58, WIN = 25;

  function names(ix) { return ix.map(function (i) { return A[i]; }).join(', '); }
  function pad(n) { return (n < 10 ? '0' : '') + n; }
  function ts(r) { return D.days[r[0]] + ' ' + pad(Math.floor(r[1] / 60)) + ':' + pad(r[1] % 60); }
  function comb(n, k) { if (k < 0 || k > n) return 0; var r = 1; for (var i = 1; i <= k; i++) r = r * (n - k + i) / i; return r; }
  function fmt(x, d) { return x.toFixed(d).replace('.', ','); }
  function units(x) { return (x >= 0 ? '+' : '') + Math.round(x).toLocaleString('es') + ' u'; }

  function build(key) {
    var items = [], chance = 0, stake = 0, ret = 0;
    if (key === 'cons') {
      D.turns.forEach(function (t) {
        var top = t[2], real = t[3], when = ts(t);
        var hits = top.filter(function (a) { return real.indexOf(a) >= 0; });
        chance += 1 - comb(38 - real.length, 10) / comb(38, 10);
        items.push({ t: when, hit: hits.length > 0, rows: [
          ['Turno', when], ['Top 10 consolidado', names(top)], ['Resultados', names(real)],
          ['Coincidencias', hits.length ? names(hits) : '—']] });
      });
      return { items: items, chance: items.length ? chance / items.length : 0, unit: 'turnos' };
    }
    if (key.indexOf('top10_') === 0) {
      var L = +key.split('_')[1];
      D.draws.forEach(function (d) {
        if (d[2] !== L) return;
        var pos = d[4].indexOf(d[3]), when = ts(d);
        items.push({ t: when, hit: pos >= 0, rows: [
          ['Sorteo', when + ' · ' + LOT[d[2]]], ['Top 10 predicho', names(d[4])], ['Ganador', A[d[3]]],
          ['Posición del ganador', pos >= 0 ? String(pos + 1) : 'fuera del top 10'],
          ['Prob. top 1', fmt(d[5][0] / 1000, 2) + ' %']] });
      });
      return { items: items, chance: 10 / 38, unit: 'sorteos' };
    }
    D.draws.forEach(function (d) {
      var bets, L = d[2], when = ts(d);
      if (key === 'strat_top2') { bets = d[4].slice(0, 2); }
      else {
        bets = [];
        for (var j = 0; j < D.valueMax; j++) {
          if (d[5][j] / 1e5 * D.payout[L] >= 1 + D.valueMargin) bets.push(d[4][j]);
        }
      }
      if (!bets.length) return;
      var won = bets.indexOf(d[3]) >= 0, pay = won ? D.payout[L] : 0;
      stake += bets.length; ret += pay;
      chance += bets.length / 38;
      items.push({ t: when, hit: won, rows: [
        ['Sorteo', when + ' · ' + LOT[L]], ['Apuestas (1 u c/u)', names(bets)], ['Ganador', A[d[3]]],
        ['Resultado', units(pay - bets.length)]] });
    });
    return { items: items, chance: items.length ? chance / items.length : 0, unit: 'sorteos con apuesta',
             stake: stake, ret: ret };
  }

  function table(rows) {
    var t = document.createElement('table'); t.className = 'tt-table';
    rows.forEach(function (r) {
      var tr = document.createElement('tr'), a = document.createElement('td'), b = document.createElement('td');
      a.className = 'tt-key'; a.textContent = r[0]; b.textContent = r[1];
      tr.appendChild(a); tr.appendChild(b); t.appendChild(tr);
    });
    return t;
  }

  function render(key) {
    var ds = build(key), data = ds.items, n = data.length;
    canvas.textContent = ''; detail.textContent = '';
    var em = document.createElement('em'); em.textContent = 'Haz clic en una barra para ver el detalle completo.';
    detail.appendChild(em);
    if (!n) { canvas.textContent = 'Sin datos.'; summary.textContent = ''; return; }
    var hits = data.filter(function (d) { return d.hit; }).length;
    var txt = n.toLocaleString('es') + ' ' + ds.unit + ' · ' + hits + ' HIT · ' + (n - hits) + ' MISS · tasa '
      + fmt(hits / n * 100, 1) + ' % · azar ' + fmt(ds.chance * 100, 1) + ' %';
    if (ds.stake) {
      txt += ' · apostado ' + Math.round(ds.stake).toLocaleString('es') + ' u · ganancia ' + units(ds.ret - ds.stake)
        + ' (ROI ' + fmt((ds.ret / ds.stake - 1) * 100, 1) + ' %)';
    }
    summary.textContent = txt;
    canvas.style.width = (n * SLOT + 10) + 'px';
    canvas.style.height = (AREA + LABEL_H) + 'px';
    var prevDate = null;
    data.forEach(function (d, i) {
      var col = document.createElement('div');
      col.className = 'bar-col'; col.dataset.index = i;
      col.style.left = (i * SLOT) + 'px'; col.style.width = SLOT + 'px';
      col.style.bottom = LABEL_H + 'px'; col.style.height = AREA + 'px';
      var bar = document.createElement('div');
      bar.className = 'bar-rect ' + (d.hit ? 'bar-hit' : 'bar-miss');
      bar.style.width = BAR_W + 'px'; bar.style.height = (d.hit ? HIT_H : MISS_H) + 'px';
      col.appendChild(bar); canvas.appendChild(col);
      var date = d.t.slice(0, 10);
      if (date !== prevDate) {
        prevDate = date;
        var line = document.createElement('div'); line.className = 'day-line';
        line.style.left = (i * SLOT - 2) + 'px'; line.style.bottom = LABEL_H + 'px'; line.style.height = AREA + 'px';
        canvas.appendChild(line);
        var lbl = document.createElement('div'); lbl.className = 'day-label';
        lbl.textContent = date.slice(5); lbl.style.left = (i * SLOT) + 'px';
        canvas.appendChild(lbl);
      }
    });
    var NS = 'http://www.w3.org/2000/svg', svg = document.createElementNS(NS, 'svg');
    svg.setAttribute('class', 'trend-svg'); svg.setAttribute('width', n * SLOT); svg.setAttribute('height', AREA);
    svg.style.bottom = LABEL_H + 'px';
    var y = function (r) { return AREA - r * (AREA - 10) - 5; };
    var ch = document.createElementNS(NS, 'line');
    ch.setAttribute('class', 'chance-line'); ch.setAttribute('x1', 0); ch.setAttribute('x2', n * SLOT);
    ch.setAttribute('y1', y(ds.chance)); ch.setAttribute('y2', y(ds.chance));
    svg.appendChild(ch);
    var pts = [], win = [];
    data.forEach(function (d, i) {
      win.push(d.hit ? 1 : 0); if (win.length > WIN) win.shift();
      var r = win.reduce(function (a, b) { return a + b; }, 0) / win.length;
      pts.push((i * SLOT + SLOT / 2) + ',' + y(r));
    });
    var poly = document.createElementNS(NS, 'polyline');
    poly.setAttribute('points', pts.join(' ')); poly.setAttribute('class', 'trend-line');
    svg.appendChild(poly); canvas.appendChild(svg);
    scroll.scrollLeft = scroll.scrollWidth;
    canvas.onclick = function (ev) {
      var col = ev.target.closest ? ev.target.closest('.bar-col') : null;
      if (!col) return;
      canvas.querySelectorAll('.bar-col.selected').forEach(function (p) { p.classList.remove('selected'); });
      col.classList.add('selected');
      var d = data[+col.dataset.index];
      var card = document.createElement('div');
      card.className = 'chart-detail-card ' + (d.hit ? 'detail-hit' : 'detail-miss');
      var head = document.createElement('div'); head.className = 'chart-detail-head';
      var title = document.createElement('strong'); title.textContent = d.t;
      var badge = document.createElement('span');
      badge.className = 'pill ' + (d.hit ? 'pill-hit' : 'pill-miss');
      badge.textContent = d.hit ? '✓ HIT' : '✗ MISS';
      head.appendChild(title); head.appendChild(badge); card.appendChild(head); card.appendChild(table(d.rows));
      detail.textContent = ''; detail.appendChild(card);
    };
    canvas.onmousemove = function (ev) {
      var col = ev.target.closest ? ev.target.closest('.bar-col') : null;
      if (!col) { tip.style.display = 'none'; return; }
      var d = data[+col.dataset.index];
      tip.textContent = '';
      var t = document.createElement('div'); t.className = 'tt-title';
      t.textContent = (d.hit ? '✓ HIT · ' : '✗ MISS · ') + d.t;
      tip.appendChild(t); tip.appendChild(table(d.rows.slice(1)));
      tip.style.display = 'block';
      var w = tip.offsetWidth, h = tip.offsetHeight, left = ev.clientX + 14, top = ev.clientY + 14;
      if (left + w > window.innerWidth - 8) left = ev.clientX - w - 14;
      if (top + h > window.innerHeight - 8) top = ev.clientY - h - 14;
      tip.style.left = Math.max(8, left) + 'px'; tip.style.top = Math.max(8, top) + 'px';
    };
    canvas.onmouseleave = function () { tip.style.display = 'none'; };
  }

  select.addEventListener('change', function () { render(select.value); });
  render(select.value);
})();
"""


def _effect_text(model, name, beta):
    """Efecto multiplicativo legible; para las afinidades, en el par más evitado."""
    kind = CONTINUOUS_FEATURES.get(name, "")
    if kind == "":
        return f"×{num(math.exp(beta))}"
    if kind is None:
        return f"×{num(math.exp(beta))} por cada 4 apariciones extra"
    M = model.mats[kind]
    i, j = np.unravel_index(np.argmin(M + np.eye(N_ANIMALS) * 99), M.shape)
    return f"{NUM_ANIMAL[i]} → {NUM_ANIMAL[j]}: ×{num(math.exp(beta * M[i, j]))}"


def model_section(ctx):
    blocks = []
    for L in LOTTERIES:
        m = ctx["models"][L]
        names = feature_names(m.groups)
        rows = []
        for nm, b, se in zip(names, m.beta, m.se):
            z = b / se if se > 0 else float("nan")
            rows.append([esc(feature_description(nm, L)), num(b, 3), num(se, 3), num(z, 1),
                         esc(_effect_text(m, nm, b))])
        blocks.append(f'<div><h3>{esc(LOTTERY_LABEL[L])}</h3>' + table_html(
            ["Variable", "β", "Error est. (aprox.)", "z", "Efecto en la probabilidad"], rows,
            num_cols=(1, 2, 3, 4)) + "</div>")
    lead = ("Coeficientes del logit condicional entrenado con todo el histórico (pesos con vida media de "
            f"{HALF_LIFE_DAYS} días). «×0,25» significa que esa condición multiplica la probabilidad del animal "
            "por 0,25 frente a la referencia. Para las afinidades de pares y transiciones se muestra el efecto en "
            "el par más evitado: «toro → vaca: ×0,5» quiere decir que, si acaba de salir toro, esa variable "
            "multiplica la probabilidad de vaca por 0,5.")
    return section("modelo", "🧠 El modelo por dentro", lead, f'<div class="grid-2">{"".join(blocks)}</div>')


def method_section(ctx):
    items = [
        "<strong>Datos:</strong> cada sorteo de La Granjita y Lotto Activo desde 2021 (" +
        f"{ctx['n_rows']:,}".replace(",", ".") + " resultados). Se eliminan duplicados y se normalizan los nombres.",
        "<strong>Modelo:</strong> logit condicional (modelo de elección discreta de McFadden). Para cada sorteo, "
        "cada uno de los 38 animales recibe una puntuación x·β y la probabilidad es exp(x·β)/Σ exp(x·β). Los β se "
        "estiman por máxima verosimilitud (Newton-Raphson) con regularización L2.",
        "<strong>Variables:</strong> sorteos desde la última aparición en la misma lotería y en las otras dos, si "
        "ya salió hoy, afinidad de pares del mismo día y de transiciones (aprendidas de los datos con contracción "
        "hacia 0). Las matrices de pares se estiman con validación cruzada por años para no sobreajustar.",
        "<strong>Loterías:</strong> La Granjita y Lotto Activo sortean a las hh:00 y Lotto Activo Int a las hh:30 "
        "(datos desde 2022). Cada lotería tiene su propio modelo, que usa también lo que salió antes en las otras.",
        "<strong>Estrategia:</strong> valor esperado con pago 30×; reglas fijas evaluadas fuera de muestra, con la "
        "regla principal elegida usando solo el año anterior al periodo evaluado.",
        f"<strong>Peso del pasado:</strong> los patrones cambian con el tiempo, así que cada sorteo pesa "
        f"0,5^(antigüedad/{HALF_LIFE_DAYS} días).",
        f"<strong>Sin fuga de datos:</strong> walk-forward; el modelo se re-entrena cada {REFIT_DAYS} días usando "
        "solo el pasado y predice el bloque siguiente.",
        "<strong>Patrones descartados:</strong> hora del día, día de la semana, frecuencia histórica, tendencia "
        "reciente y conteo reciente no mejoran fuera de muestra.",
        "<strong>Referencia:</strong> el modelo V7 anterior (pesos fijos) obtuvo en sus últimos 90 días de "
        "backtest (hasta 2026-09-23) un top 10 de 27,2 % en La Granjita y 33,0 % en Lotto Activo; el V8 obtiene "
        "~38 %.",
        "<strong>Advertencia:</strong> que un backtest sea rentable no garantiza ganancias futuras: el operador "
        "puede cambiar su mecanismo en cualquier momento (los efectos ya varían por año) y las condiciones reales "
        "de pago pueden diferir. Este análisis es académico, no asesoría financiera.",
    ]
    return section("metodo", "📚 Metodología y advertencias", "",
                   '<ul class="method">' + "".join(f"<li>{i}</li>" for i in items) + "</ul>")


def generate_html(ctx):
    s = ctx["topk"]
    both10 = s[(s["lottery"] == "todas") & (s["K"] == 10)].iloc[0]
    both1 = s[(s["lottery"] == "todas") & (s["K"] == 1)].iloc[0]
    both3 = s[(s["lottery"] == "todas") & (s["K"] == 3)].iloc[0]
    both5 = s[(s["lottery"] == "todas") & (s["K"] == 5)].iloc[0]
    cons = ctx["cons"]
    st = ctx["strategies"]
    main_ev = st[(st["period"] == "evaluacion") & (st["strategy"] == MAIN_STRATEGY)].iloc[0]
    tiles = [
        stat_tile(f"Acierto top 10 por lotería · últimos {EVAL_DAYS} días", pct(both10.rate, 1),
                  f"Azar {pct(both10.chance, 1)} · ×{num(both10.lift)} · IC 95 % {pct(both10.ci_low, 1)}–{pct(both10.ci_high, 1)}",
                  hero=True),
        stat_tile("ROI de la estrategia principal · último año", pct(main_ev["roi"], 1),
                  f"{esc(STRATEGY_RULES[MAIN_STRATEGY][0])} · IC 95 % {pct(main_ev['roi_lo'], 0)} a "
                  f"{pct(main_ev['roi_hi'], 0)} · al azar sería −21 %"),
        stat_tile("Top 1", pct(both1.rate, 2), f"Azar {pct(both1.chance, 2)} · ×{num(both1.lift)}"),
        stat_tile("Top 3 por lotería", pct(both3.rate, 2), f"Azar {pct(both3.chance, 2)} · ×{num(both3.lift)}"),
        stat_tile("Top 5", pct(both5.rate, 1), f"Azar {pct(both5.chance, 1)} · ×{num(both5.lift)}"),
        stat_tile("Top 10 consolidado por turno", pct(cons["hit"].mean(), 1),
                  f"Azar {pct(cons['chance_top10'].mean(), 1)} (La Granjita + Lotto Activo)"),
    ]
    toc = "".join(f'<a href="#{a}">{t}</a>' for a, t in [
        ("proximo", "Próximo sorteo"), ("estrategia", "Estrategia"), ("evolucion", "HIT / MISS"),
        ("once", "11 turnos"), ("hoy", "Último día"), ("rendimiento", "Rendimiento"), ("patrones", "Patrones"),
        ("modelo", "Modelo"), ("metodo", "Metodología")])
    last_txt = " · ".join(f"{LOTTERY_LABEL[L]} {t:%H:%M}: {animal_label(a)}" for L, (t, a) in ctx["last_by_lottery"].items())
    parts = [
        '<!DOCTYPE html><html lang="es"><head><meta charset="UTF-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1.0">',
        '<title>Predictor de Animales V8</title>',
        f"<style>{CSS}{HITMISS_CSS}</style></head><body><div class=\"container\">",
        '<header class="top"><h1>🐾 Predictor de Animales · V8</h1>',
        '<p>Logit condicional + pruebas de hipótesis sobre La Granjita, Lotto Activo y Lotto Activo Int.</p>',
        f'<p>Actualizado: <strong>{esc(ctx["generated"])}</strong> · Datos: {esc(ctx["data_range"])}</p>',
        f'<p>Últimos resultados ({esc(ctx["today"].strftime("%Y-%m-%d"))}): <strong>{esc(last_txt)}</strong></p>',
        f'<nav class="toc">{toc}</nav></header>',
        f'<div class="tiles">{"".join(tiles)}</div>',
        next_turn_section(ctx["next_table"], ctx["next_turn"], ctx["next_same"], ctx["next_pred"]),
        strategy_section(ctx),
        hitmiss_section(ctx),
        eleven_section(ctx["e11"], ctx["e11_turns"], ctx["e11_chance"], ctx["h11"]),
        today_section(ctx["bt"], ctx["cons"], ctx["today"]),
        performance_section(ctx),
        patterns_section(ctx),
        model_section(ctx),
        method_section(ctx),
        '</div><div class="tooltip" id="tooltip" role="tooltip"></div>',
        f"<script>{JS}{HITMISS_JS}</script></body></html>",
    ]
    return "\n".join(parts)


# ================================================================
# MAIN
# ================================================================

def save_outputs(ctx):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    def out(name):
        return os.path.join(OUTPUT_DIR, name)

    def write(frame, name):
        frame.to_csv(out(name), index=False, encoding="utf-8-sig", float_format="%.6g")

    bt_out = ctx["bt"].drop(columns=["probs", "real_animal"]).copy()
    bt_out.insert(1, "date", bt_out["datetime"].dt.strftime("%Y-%m-%d"))
    bt_out.insert(2, "time", bt_out["datetime"].dt.strftime("%H:%M"))
    bt_out["winner"] = bt_out["winner"].map(NUM_ANIMAL)
    bt_out = bt_out.rename(columns={"cutoff": "model_trained_until", "winner": "winner_animal"})
    write(bt_out, "backtest.csv")

    cons = ctx["cons"]
    day = ctx["today"]
    today_c = cons[cons["datetime"].dt.normalize() == day].copy()
    today_c["status"] = np.where(today_c["hit"], "HIT", "MISS")
    write(today_c[["datetime", "date", "time", "status", "top10", "real_animals", "hit_animals"]],
          "turnos_hoy.csv")
    today_c["status"] = np.where(today_c["hit_top7"], "HIT", "MISS")
    write(today_c[["datetime", "date", "time", "status", "top_list", "real_animals", "hit_animals_top7"]]
          .rename(columns={"hit_animals_top7": "hit_animals"}), "turnos_hoy_top7_loteria.csv")

    write(ctx["next_table"], "prob-animal-next-turn.csv")
    write(ctx["e11"], "prob-11-turnos.csv")
    if ctx["h11"] is not None and not ctx["h11"].empty:
        write(ctx["h11"], "historico-11-turnos.csv")
    write(ctx["patterns"], "patrones.csv")
    write(ctx["pairs"], "pares.csv")
    write(ctx["hazard"], "curva_riesgo.csv")

    coef = []
    for L in LOTTERIES:
        m = ctx["models"][L]
        for nm, b, se in zip(feature_names(m.groups), m.beta, m.se):
            coef.append({"lottery": L, "variable": feature_label(nm, L), "descripcion": feature_description(nm, L),
                         "beta": b, "se": se, "z": b / se if se > 0 else float("nan"),
                         "efecto": _effect_text(m, nm, b)})
    write(pd.DataFrame(coef), "coeficientes.csv")

    met = ctx["topk"].copy()
    met.insert(0, "modelo", FINAL_MODEL)
    abl = pd.DataFrame([{"modelo": r["key"], "lottery": "todas", "K": K, "rate": r[f"top{K}"],
                         "chance": K / N_ANIMALS, "info_bits": r["gain"]}
                        for r in ctx["ablation"] for K in TOP_KS])
    write(pd.concat([met, abl], ignore_index=True), "metricas.csv")

    st = ctx["strategies"].copy()
    st.insert(0, "regla", st.pop("label"))
    write(st, "estrategias.csv")
    led = ctx["ledgers"][MAIN_STRATEGY]
    led = led[led["datetime"] >= led["datetime"].max().normalize() - timedelta(days=89)].copy()
    led.insert(0, "regla", STRATEGY_RULES[MAIN_STRATEGY][0])
    write(led.rename(columns={"bets": "apuestas", "stake": "apostado", "return": "cobrado",
                              "profit": "ganancia", "won": "gano", "winner": "ganador"}),
          "estrategia_apuestas.csv")
    if ctx["bankroll"] is not None and not ctx["bankroll"].empty:
        write(ctx["bankroll"], "estrategia_simulacion.csv")

    html_file = out("dashboard.html")
    with open(html_file, "w", encoding="utf-8") as f:
        f.write(generate_html(ctx))
    return html_file


def run_ablation(rows, pairs, last_time):
    out = [{"key": "azar", "top1": 1 / 38, "top3": 3 / 38, "top5": 5 / 38, "top10": 10 / 38, "gain": 0.0}]
    for key, groups in MODEL_SPECS.items():
        b = add_rank_columns(walk_forward(rows, pairs, last_time, groups, EVAL_DAYS, ABLATION_REFIT_DAYS))
        rec = {"key": key, "gain": info_gain_bits(b)}
        for K in TOP_KS:
            rec[f"top{K}"] = float(b[f"hit_top{K}"].mean())
        out.append(rec)
        print(f"   {MODEL_LABELS[key]:<42} top10={rec['top10']*100:5.2f}%  info={rec['gain']*1000:5.1f} mbit")
    return out


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    t_start = time.time()
    print("=" * 70)
    print("V8 - LOGIT CONDICIONAL + PRUEBAS DE HIPÓTESIS")
    print("=" * 70)

    df = load_data()
    table = build_turn_table(df)
    last_time = df["datetime"].max()
    print(f"Resultados: {len(df):,} · desde {df['datetime'].min()} hasta {last_time}")

    rows, state, _ = replay_history(table)
    pairs = PairData(df)

    print(f"\nBacktest walk-forward ({STRATEGY_DAYS} días, re-entrenando cada {REFIT_DAYS} días)...")
    bt_long, fold_models = walk_forward(rows, pairs, last_time, FINAL_GROUPS, STRATEGY_DAYS, REFIT_DAYS,
                                        keep_models=True)
    bt_long = add_rank_columns(bt_long)
    eval_start = last_time.normalize() + timedelta(days=1) - timedelta(days=EVAL_DAYS)
    bt = bt_long[bt_long["datetime"] >= eval_start].reset_index(drop=True)
    topk = topk_summary(bt)
    cons = consolidated_backtest(bt)
    for r in topk.itertuples():
        print(f"   {r.lottery:<14} top{r.K:<2}: {r.rate*100:6.2f}%  (azar {r.chance*100:5.2f}%, z={r.z:5.1f})")
    print(f"   consolidado top10 por turno: {cons['hit'].mean()*100:.2f}% (azar {cons['chance_top10'].mean()*100:.2f}%)")

    print("\nEstrategias de apuesta (pago 30x)...")
    rng = np.random.default_rng(RANDOM_SEED)
    strategies, ledgers, strat_lot = evaluate_strategies(bt_long, eval_start, rng)
    for r in strategies.itertuples():
        print(f"   {r.label:<44} {r.period:<10} ROI {r.roi*100:+6.1f}% "
              f"[{r.roi_lo*100:+5.1f}, {r.roi_hi*100:+5.1f}]  ganancia {r.profit:+8.0f} u")
    bankroll, _ = bankroll_simulation(ledgers[MAIN_STRATEGY], rng)
    P_ev = np.vstack(bt["probs"].to_numpy())
    B_ev = strategy_bets(P_ev, STRATEGY_RULES[MAIN_STRATEGY][1], bt["lottery"].map(PAYOUT).to_numpy(dtype=float))
    kelly_p = float(P_ev[B_ev].mean()) if B_ev.any() else CHANCE

    print("\nComparación de modelos (qué aporta cada patrón)...")
    ablation = run_ablation(rows, pairs, last_time)

    print("\nPruebas de hipótesis...")
    patterns, pair_res, lags = run_pattern_tests(df, table, pairs)
    hazard = hazard_curve(rows)

    print("\nModelo final y predicciones...")
    schedule = infer_schedule(df)
    models = fit_final_models(rows, pairs, last_time)
    pred = predict_next_turn(df, table, state, models, schedule)
    next_table, next_turn, next_same = next_turn_table(pred)
    e11, e11_turns, e11_chance = eleven_turn_prediction(state, models, last_time, schedule, rng)
    h11 = eleven_turn_history(table, fold_models, rng)

    last_by_lottery = {}
    for L in LOTTERIES:
        sub = df[df["lottery"] == L]
        if len(sub):
            r = sub.loc[sub["datetime"].idxmax()]
            last_by_lottery[L] = (r["datetime"], r["animal"])
    ctx = {
        "bt": bt, "topk": topk, "cons": cons, "ablation": ablation, "gain_bits": info_gain_bits(bt),
        "patterns": patterns, "pairs": pair_res, "lags": lags, "hazard": hazard,
        "models": models, "next_pred": pred, "next_table": next_table, "next_turn": next_turn,
        "next_same": next_same, "e11": e11, "e11_turns": e11_turns, "e11_chance": e11_chance, "h11": h11,
        "strategies": strategies, "ledgers": ledgers, "strategy_per_lottery": strat_lot,
        "bankroll": bankroll, "kelly_p": kelly_p,
        "today": last_time.normalize(), "n_rows": len(df), "last_by_lottery": last_by_lottery,
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "data_range": f"{df['datetime'].min():%Y-%m-%d} → {last_time:%Y-%m-%d %H:%M} · {len(df):,} sorteos".replace(",", "."),
    }

    print(f"\nPróximo turno: {next_turn}")
    for L in LOTTERIES:
        top = next_table.sort_values(f"p_{L}", ascending=False).head(5)
        print(f"   {L:<12}: " + ", ".join(f"{a} {p*100:.1f}%" for a, p in zip(top['animal'], top[f'p_{L}'])))

    html_file = save_outputs(ctx)
    print(f"\nDashboard: {html_file}")
    print(f"Tiempo total: {time.time() - t_start:.0f} s")


if __name__ == "__main__":
    main()
