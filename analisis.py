# ================================================================
# V7 - PREDICTOR DE ANIMALES
# ================================================================
#
# FUNCIONALIDADES
#
# 1. Lee resultados.csv
# 2. Limpia y normaliza los datos
# 3. Detecta automáticamente los turnos reales
# 4. Backtest de los últimos 30 días
# 5. No utiliza información futura para generar predicciones
# 6. Analiza:
#       - frecuencia histórica
#       - frecuencia reciente
#       - frecuencia por hora
#       - recencia del animal
#       - frecuencia por lotería
#       - influencia entre loterías
#       - tendencia reciente
# 7. Calcula probabilidades para el siguiente turno REAL
# 8. Evalúa todos los turnos del día de ejecución
# 9. Determina HIT / MISS sin importar la lotería
# 10. Genera CSVs
# 11. Genera dashboard HTML + CSS + JS
#
# CSV esperado:
#
# date,lottery,number,animal,time
#
# Ejemplo:
#
# 2026-08-26,LottoActivo,12,gato,16:00
#
# ================================================================

import os
import math
import json
import html
import warnings
from collections import Counter, defaultdict
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ================================================================
# CONFIGURACIÓN
# ================================================================

# Rutas relativas al script (para que funcione desde cualquier directorio).
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

INPUT_FILE = os.path.join(BASE_DIR, "resultados.csv")

OUTPUT_DIR = os.path.join(BASE_DIR, "v7_output")

BACKTEST_DAYS = 30

# Cantidad de animales mostrados en ranking
TOP_N = 10

# Top utilizado para determinar HIT
TOP_HIT = 10

# Cantidad de animales tomados de cada lotería
# en la sección "Top 7 por lotería"
TOP_PER_LOTTERY = 7


# ================================================================
# PESOS DEL MODELO
# ================================================================
#
# La suma no necesita ser exactamente 1.
# Posteriormente se normaliza.
#
# Puedes modificar estos valores para experimentar.
#
# ================================================================

WEIGHTS = {

    # Frecuencia histórica (casi uniforme; aporta poco)
    "historical": 0.04,

    # Frecuencia reciente (contradice la anti-repetición; se reduce)
    "recent": 0.06,

    # Frecuencia específica de la hora (sin señal real)
    "hour": 0.04,

    # ANTI-REPETICIÓN: el animal casi nunca repite a corto plazo.
    # Es el patrón dominante. Puntúa ALTO a los animales que llevan
    # más TURNOS sin salir (por lotería) y BAJO a los recientes.
    "anti_repetition": 0.42,

    # Tendencia
    "trend": 0.05,

    # Influencia entre loterías (sin señal real)
    "cross_lottery": 0.04,

    # PASO NUMÉRICO: el número rara vez cae en su vecino (±1).
    "step": 0.14,

    # PARES QUE SE EVITAN: toro~vaca, raton~elefante, oso~pescado,
    # zorro~gallina, leon~tigre (evitación a corto plazo).
    "pair": 0.21,
}


# ================================================================
# TABLA OFICIAL (animal -> número) Y PARES DESCUBIERTOS
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

# Pares que se EVITAN a corto plazo (descubierto en el análisis multidimensional;
# robustos a Bonferroni global, estables por año y en ambas loterías).
AVOIDED_PAIRS = {
    ("toro", "vaca"),
    ("raton", "elefante"),
    ("oso", "pescado"),
    ("zorro", "gallina"),
    ("leon", "tigre"),
}


# ================================================================
# UTILIDADES
# ================================================================

def safe_float(value):
    try:
        return float(value)
    except Exception:
        return 0.0


def normalize_dict(values):
    """
    Convierte los valores de un diccionario a una distribución
    entre 0 y 1.
    """

    if not values:
        return {}

    maximum = max(values.values())

    if maximum <= 0:
        return {k: 0.0 for k in values}

    return {
        k: float(v) / float(maximum)
        for k, v in values.items()
    }


def softmax(values):
    """
    Convierte scores en probabilidades.
    """

    if not values:
        return {}

    keys = list(values.keys())

    x = np.array(
        [safe_float(values[k]) for k in keys],
        dtype=float
    )

    x = x - np.max(x)

    exp_x = np.exp(x)

    total = exp_x.sum()

    if total <= 0:
        return {
            k: 1.0 / len(keys)
            for k in keys
        }

    probabilities = exp_x / total

    return {
        k: float(p)
        for k, p in zip(keys, probabilities)
    }


# ================================================================
# CARGA DE DATOS
# ================================================================

def load_data():

    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(
            f"No se encontró {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    df.columns = [
        str(c).strip().lower()
        for c in df.columns
    ]

    required = [
        "date",
        "lottery",
        "animal",
        "time"
    ]

    missing = [
        c for c in required
        if c not in df.columns
    ]

    if missing:
        raise ValueError(
            "Faltan columnas en resultados.csv: "
            + ", ".join(missing)
        )

    # ------------------------------------------------------------
    # Fecha
    # ------------------------------------------------------------

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce"
    )

    # ------------------------------------------------------------
    # Hora
    # ------------------------------------------------------------

    df["time"] = (
        df["time"]
        .astype(str)
        .str.strip()
    )

    # Normalizar la hora a formato de 24 horas para ordenar los turnos.
    parsed_time = pd.to_datetime(
        df["time"],
        format="%I:%M %p",
        errors="coerce"
    )

    missing_time = parsed_time.isna()

    if missing_time.any():
        parsed_time.loc[missing_time] = pd.to_datetime(
            df.loc[missing_time, "time"],
            format="%H:%M",
            errors="coerce"
        )

    df["hour"] = parsed_time.dt.strftime("%H:%M")

    # ------------------------------------------------------------
    # Normalización
    # ------------------------------------------------------------

    df["lottery"] = (
        df["lottery"]
        .astype(str)
        .str.strip()
    )

    df["animal"] = (
        df["animal"]
        .astype(str)
        .str.strip()
        .str.lower()
        .replace({
            # 'cebra' y 'zebra' son el mismo animal
            "zebra": "cebra",
        })
    )

    # ------------------------------------------------------------
    # Número oficial (derivado del animal, tabla fija)
    # ------------------------------------------------------------

    df["number"] = df["animal"].map(ANIMAL_NUM)

    # ------------------------------------------------------------
    # Timestamp
    # ------------------------------------------------------------

    df["datetime"] = pd.to_datetime(
        df["date"].dt.strftime("%Y-%m-%d")
        + " "
        + df["hour"],
        errors="coerce"
    )

    df = df.dropna(
        subset=[
            "date",
            "datetime",
            "animal",
            "hour"
        ]
    )

    df = df.sort_values(
        "datetime"
    ).reset_index(drop=True)

    return df


# ================================================================
# DETECTAR TURNOS REALES
# ================================================================

def get_real_turns(df):

    """
    Un turno está determinado por fecha + hora.

    IMPORTANTE:

    La lotería NO forma parte del identificador del turno.

    Esto permite que:

        LottoActivo 16:00
        LaGranjita   16:00

    pertenezcan al mismo turno.

    """

    turns = (
        df[
            [
                "datetime",
                "date",
                "hour"
            ]
        ]
        .drop_duplicates()
        .sort_values("datetime")
        .reset_index(drop=True)
    )

    return turns


# ================================================================
# CONSTRUIR HISTORIAL
# ================================================================

def build_indexes(history):

    indexes = {

        "animals": set(),

        "lotteries": set(),

        "animal_count": Counter(),

        "animal_by_hour": defaultdict(Counter),

        "animal_by_lottery": defaultdict(Counter),

        "lottery_animal_count": defaultdict(Counter),

        "hour_count": Counter(),

        "recent_rows": [],

        "last_animal_time": {},

        "cross_lottery": defaultdict(Counter),

    }

    for _, row in history.iterrows():

        animal = row["animal"]
        lottery = row["lottery"]
        hour = row["hour"]

        indexes["animals"].add(animal)
        indexes["lotteries"].add(lottery)

        indexes["animal_count"][animal] += 1

        indexes["animal_by_hour"][hour][animal] += 1

        indexes["animal_by_lottery"][
            lottery
        ][animal] += 1

        indexes["lottery_animal_count"][
            lottery
        ][animal] += 1

        indexes["hour_count"][hour] += 1

        indexes["last_animal_time"][
            animal
        ] = row["datetime"]

    return indexes


# ================================================================
# FRECUENCIA HISTÓRICA
# ================================================================

def historical_scores(history, animals):

    counts = Counter(
        history["animal"]
    )

    total = max(
        len(history),
        1
    )

    raw = {}

    for animal in animals:

        raw[animal] = (
            counts.get(animal, 0)
            / total
        )

    return normalize_dict(raw)


# ================================================================
# FRECUENCIA RECIENTE
# ================================================================

def recent_scores(history, animals, window=100):

    recent = history.tail(window)

    counts = Counter(
        recent["animal"]
    )

    total = max(
        len(recent),
        1
    )

    raw = {}

    for animal in animals:

        raw[animal] = (
            counts.get(animal, 0)
            / total
        )

    return normalize_dict(raw)


# ================================================================
# FRECUENCIA POR HORA
# ================================================================

def hour_scores(history, animals, target_hour):

    hour_data = history[
        history["hour"] == target_hour
    ]

    counts = Counter(
        hour_data["animal"]
    )

    total = max(
        len(hour_data),
        1
    )

    raw = {}

    for animal in animals:

        raw[animal] = (
            counts.get(animal, 0)
            / total
        )

    return normalize_dict(raw)


# ================================================================
# ANTI-REPETICIÓN (patrón dominante descubierto)
# ================================================================

def anti_repetition_scores(history, animals, target_datetime):

    """
    El mismo animal casi nunca repite a corto plazo (lag 1 ≈ 0.64%
    vs 2.63% esperado). Este componente puntúa ALTO a los animales
    que llevan más TURNOS sin salir en cada lotería, y BAJO a los
    recientes.

    Corrige el error del modelo anterior, que premiaba a los animales
    recién salidos (dirección opuesta a la anti-repetición).
    """

    h = history.sort_values("datetime")

    turn_times = (
        h["datetime"]
        .drop_duplicates()
        .sort_values()
        .tolist()
    )

    turn_index = {
        ts: i
        for i, ts in enumerate(turn_times)
    }

    h = h.copy()

    h["turn"] = h["datetime"].map(turn_index)

    n_turns = len(turn_times)

    # Último turno de aparición de cada (animal, lotería) — vectorizado.
    last_map = (
        h.groupby(["animal", "lottery"])["turn"]
        .max()
        .to_dict()
    )

    lotteries = h["lottery"].dropna().unique()

    raw = {}

    for animal in animals:

        total = 0.0

        for lottery in lotteries:

            k = float(
                n_turns
                - last_map.get((animal, lottery), -24)
            )

            # Liberación: crece con k (turnos desde la última aparición).
            # k=1 → ~0.15 ; k=6 → ~0.63 ; k=12 → ~0.86 ; k→∞ → 1.
            total += 1.0 - math.exp(-max(k, 0.0) / 6.0)

        raw[animal] = total

    return normalize_dict(raw)


# ================================================================
# PASO NUMÉRICO (vecindad)
# ================================================================

def step_scores(history, animals):

    """
    El número rara vez cae en su vecino inmediato (paso ±1) y nunca
    repite (paso 0, ya cubierto por la anti-repetición). Penaliza los
    animales cuyo número es vecino del último número de cada lotería.
    """

    factor = [1.0] * 38

    factor[0] = 0.15   # repetir número (refuerza la anti-repetición)
    factor[1] = 0.60   # vecino +1 (suprimido)
    factor[37] = 0.70  # vecino -1 (suprimido)

    raw = {
        animal: 1.0
        for animal in animals
    }

    h = history.sort_values("datetime")

    for lottery in h["lottery"].dropna().unique():

        sub = h[h["lottery"] == lottery]

        if sub.empty:
            continue

        last_num = int(sub.iloc[-1]["number"])

        for animal in animals:

            num = int(ANIMAL_NUM.get(animal, -1))

            if num < 0:
                continue

            step = (num - last_num) % 38

            raw[animal] *= factor[step]

    return normalize_dict(raw)


# ================================================================
# PARES QUE SE EVITAN (hallazgo nuevo)
# ================================================================

def pair_scores(history, animals):

    """
    Pares de animales que se evitan a corto plazo (toro↔vaca,
    raton↔elefante, oso↔pescado, zorro↔gallina, leon↔tigre).

    Si el último animal de una lotería pertenece a un par evitado,
    su pareja recibe una penalización.
    """

    raw = {
        animal: 1.0
        for animal in animals
    }

    h = history.sort_values("datetime")

    for lottery in h["lottery"].dropna().unique():

        sub = h[h["lottery"] == lottery]

        if sub.empty:
            continue

        last = sub.iloc[-1]["animal"]

        for (x, y) in AVOIDED_PAIRS:

            if last == x:
                raw[y] *= 0.55

            elif last == y:
                raw[x] *= 0.55

    return normalize_dict(raw)


# ================================================================
# TENDENCIA
# ================================================================

def trend_scores(history, animals):

    """
    Compara frecuencia reciente contra frecuencia histórica.

    Si un animal está aumentando su presencia recientemente,
    obtiene mayor score.
    """

    recent = history.tail(50)

    previous = history.iloc[:-50]

    recent_count = Counter(
        recent["animal"]
    )

    previous_count = Counter(
        previous["animal"]
    )

    raw = {}

    for animal in animals:

        recent_rate = (
            recent_count.get(animal, 0)
            / max(len(recent), 1)
        )

        previous_rate = (
            previous_count.get(animal, 0)
            / max(len(previous), 1)
        )

        if previous_rate <= 0:

            score = (
                1.0
                if recent_rate > 0
                else 0.0
            )

        else:

            score = (
                recent_rate
                / previous_rate
            )

        raw[animal] = score

    return normalize_dict(raw)


# ================================================================
# INFLUENCIA ENTRE LOTERÍAS
# ================================================================

def cross_lottery_scores(
    history,
    animals,
    target_datetime,
    target_lottery=None
):

    """
    Busca si el mismo animal apareció recientemente
    en otra lotería.

    Ejemplo:

        LottoActivo -> gato
        LaGranjita  -> ?

    Si existe evidencia histórica de que después de
    LottoActivo = gato, LaGranjita = gato aparece con
    frecuencia superior a la base, el score aumenta.

    """

    raw = {
        animal: 0.0
        for animal in animals
    }

    if target_lottery is None:

        return normalize_dict(raw)

    previous = history[
        history["datetime"] < target_datetime
    ]

    if previous.empty:

        return normalize_dict(raw)

    lotteries = list(
        previous["lottery"]
        .dropna()
        .unique()
    )

    # ------------------------------------------------------------
    # Buscar resultados recientes de otras loterías
    # ------------------------------------------------------------

    for lottery in lotteries:

        if lottery == target_lottery:
            continue

        source = previous[
            previous["lottery"] == lottery
        ]

        if source.empty:
            continue

        last_source = source.iloc[-1]

        source_animal = last_source["animal"]

        # Historial donde el animal de otra lotería
        # fue seguido por cada animal objetivo.

        source_rows = source[
            source["animal"] == source_animal
        ]

        if source_rows.empty:
            continue

        for _, src in source_rows.tail(200).iterrows():

            next_rows = previous[
                (
                    previous["datetime"]
                    > src["datetime"]
                )
                &
                (
                    previous["datetime"]
                    <=
                    src["datetime"]
                    + timedelta(hours=12)
                )
                &
                (
                    previous["lottery"]
                    == target_lottery
                )
            ]

            for _, nxt in next_rows.iterrows():

                animal = nxt["animal"]

                if animal in raw:

                    raw[animal] += 1.0

    return normalize_dict(raw)


# ================================================================
# MODELO PRINCIPAL
# ================================================================

def calculate_prediction(
    history,
    target_datetime,
    target_hour,
    target_lottery=None
):

    animals = sorted(
        history["animal"]
        .dropna()
        .unique()
    )

    if not animals:

        return pd.DataFrame()

    # ------------------------------------------------------------
    # Componentes
    # ------------------------------------------------------------

    historical = historical_scores(
        history,
        animals
    )

    recent = recent_scores(
        history,
        animals
    )

    hour = hour_scores(
        history,
        animals,
        target_hour
    )

    anti_repetition = anti_repetition_scores(
        history,
        animals,
        target_datetime
    )

    step = step_scores(
        history,
        animals
    )

    pair = pair_scores(
        history,
        animals
    )

    trend = trend_scores(
        history,
        animals
    )

    cross = cross_lottery_scores(
        history,
        animals,
        target_datetime,
        target_lottery
    )

    # ------------------------------------------------------------
    # Score combinado (suma ponderada)
    # ------------------------------------------------------------

    scores = {}

    for animal in animals:

        score = (

            WEIGHTS["historical"]
            * historical.get(animal, 0)

            +

            WEIGHTS["recent"]
            * recent.get(animal, 0)

            +

            WEIGHTS["hour"]
            * hour.get(animal, 0)

            +

            WEIGHTS["anti_repetition"]
            * anti_repetition.get(animal, 0)

            +

            WEIGHTS["trend"]
            * trend.get(animal, 0)

            +

            WEIGHTS["cross_lottery"]
            * cross.get(animal, 0)

            +

            WEIGHTS["step"]
            * step.get(animal, 0)

            +

            WEIGHTS["pair"]
            * pair.get(animal, 0)

        )

        scores[animal] = score

    probabilities = softmax(
        scores
    )

    # ------------------------------------------------------------
    # DataFrame
    # ------------------------------------------------------------

    rows = []

    for animal in animals:

        rows.append({

            "animal": animal,

            "probability":
                probabilities.get(
                    animal,
                    0
                ),

            "score":
                scores.get(
                    animal,
                    0
                ),

            "historical":
                historical.get(
                    animal,
                    0
                ),

            "recent":
                recent.get(
                    animal,
                    0
                ),

            "hour":
                hour.get(
                    animal,
                    0
                ),

            "anti_repetition":
                anti_repetition.get(
                    animal,
                    0
                ),

            "step":
                step.get(
                    animal,
                    0
                ),

            "pair":
                pair.get(
                    animal,
                    0
                ),

            "trend":
                trend.get(
                    animal,
                    0
                ),

            "cross_lottery":
                cross.get(
                    animal,
                    0
                ),
        })

    result = pd.DataFrame(rows)

    result = result.sort_values(
        "probability",
        ascending=False
    ).reset_index(drop=True)

    result["rank"] = (
        np.arange(len(result)) + 1
    )

    return result


# ================================================================
# MAPA DE RESULTADOS REALES
# ================================================================

def build_actual_results(df):

    """
    Para HIT ignoramos lottery.

    La llave es:

        datetime + animal

    """

    result = defaultdict(set)

    for _, row in df.iterrows():

        key = row["datetime"]

        result[key].add(
            row["animal"]
        )

    return result


# ================================================================
# BACKTEST
# ================================================================

def build_actual_by_lottery(df):

    """
    Ganador real de cada (turno, lotería). La anti-repetición opera
    DENTRO de cada lotería, por eso la precisión se mide por lotería.
    """

    actual = {}

    for row in df.itertuples(index=False):
        actual[(row.datetime, row.lottery)] = row.animal

    return actual


def run_backtest(df):

    print("")
    print("=" * 70)
    print("BACKTEST - ÚLTIMOS 30 DÍAS")
    print("=" * 70)

    max_date = df["datetime"].max()

    start_date = (
        max_date
        - timedelta(
            days=BACKTEST_DAYS
        )
    )

    target_turns = (
        df[
            df["datetime"] >= start_date
        ][
            [
                "datetime",
                "hour"
            ]
        ]
        .drop_duplicates()
        .sort_values("datetime")
        .reset_index(drop=True)
    )

    actual_results = build_actual_results(df)

    actual_lot = build_actual_by_lottery(df)

    # Contadores de precisión POR LOTERÍA (donde está la señal).
    lot_hits = {
        lot: {K: 0 for K in (1, 3, 5, 10)}
        for lot in ["lagranjita", "lottoactivo"]
    }

    lot_total = {
        lot: 0
        for lot in ["lagranjita", "lottoactivo"]
    }

    backtest_rows = []

    total = len(target_turns)

    for index, (_, turn) in enumerate(
        target_turns.iterrows(),
        start=1
    ):

        target_datetime = turn["datetime"]
        target_hour = turn["hour"]

        # --------------------------------------------------------
        # MUY IMPORTANTE:
        #
        # Solo datos ANTERIORES al turno.
        # --------------------------------------------------------

        history = df[
            df["datetime"]
            < target_datetime
        ]

        if history.empty:
            continue

        # --------------------------------------------------------
        # Precisión POR LOTERÍA (modelo multiplicativo).
        # --------------------------------------------------------

        for lottery in ["lagranjita", "lottoactivo"]:

            prediction = calculate_prediction(
                history=history,
                target_datetime=target_datetime,
                target_hour=target_hour,
                target_lottery=lottery
            )

            if prediction.empty:
                continue

            winner = actual_lot.get(
                (target_datetime, lottery)
            )

            if winner is None:
                continue

            lot_total[lottery] += 1

            ranked = prediction["animal"].tolist()

            for K in (1, 3, 5, 10):
                if winner in ranked[:K]:
                    lot_hits[lottery][K] += 1

        # --------------------------------------------------------
        # Predicción consolidada (para el dashboard y el CSV).
        # --------------------------------------------------------

        consolidated = calculate_prediction(
            history=history,
            target_datetime=target_datetime,
            target_hour=target_hour,
            target_lottery=None
        )

        if consolidated.empty:
            continue

        top10 = consolidated.head(
            TOP_HIT
        )

        predicted_animals = set(
            top10["animal"].tolist()
        )

        real_animals = actual_results.get(
            target_datetime,
            set()
        )

        hits = (
            predicted_animals
            .intersection(
                real_animals
            )
        )

        hit = len(hits) > 0

        hit_animals = ",".join(
            sorted(hits)
        )

        real_animals_text = ",".join(
            sorted(real_animals)
        )

        top10_text = ",".join(
            top10["animal"].tolist()
        )

        backtest_rows.append({

            "datetime":
                target_datetime,

            "date":
                target_datetime.strftime(
                    "%Y-%m-%d"
                ),

            "time":
                target_hour,

            "top10":
                top10_text,

            "real_animals":
                real_animals_text,

            "hit":
                hit,

            "hit_animals":
                hit_animals,

            "top1":
                consolidated.iloc[0]["animal"],

            "top1_probability":
                consolidated.iloc[0]["probability"],

            "top10_max_probability":
                top10["probability"].max(),

            "top10_min_probability":
                top10["probability"].min(),

        })

        if (
            index % 25 == 0
            or index == total
        ):

            print(
                f"Backtest "
                f"{index}/{total}"
            )

    result = pd.DataFrame(
        backtest_rows
    )

    print_precision(lot_hits, lot_total, result)

    return result


def print_precision(lot_hits, lot_total, backtest):

    """
    Imprime la precisión real: HIT top-K POR LOTERÍA frente al azar
    (K/38), que es donde la anti-repetición produce señal.
    """

    print("")
    print("-" * 70)
    print("PRECISIÓN POR LOTERÍA  (HIT top-K · azar = K/38)")
    print("-" * 70)

    print(
        f"  {'lotería':<12} "
        f"{'top1':>14} "
        f"{'top3':>14} "
        f"{'top5':>14} "
        f"{'top10':>16}"
    )

    for lot in ["lagranjita", "lottoactivo"]:

        n = lot_total[lot]

        if n == 0:
            continue

        cells = []

        for K in (1, 3, 5, 10):
            rate = lot_hits[lot][K] / n
            base = K / 38.0
            cells.append(f"{rate*100:.2f}%/{base*100:.2f}%")

        print(
            f"  {lot:<12} "
            f"{cells[0]:>14} "
            f"{cells[1]:>14} "
            f"{cells[2]:>14} "
            f"{cells[3]:>16}"
        )

    if not backtest.empty:

        n = len(backtest)

        hits = int(backtest["hit"].sum())

        print("")
        print(
            f"  Consolidado (top-{TOP_HIT}, mezclando las 2 loterías): "
            f"{hits}/{n} = {hits/n*100:.2f}%"
        )
        print(
            "  (la señal está DENTRO de cada lotería; al mezclar 2 sorteos"
        )
        print(
            "   independientes el top-10 consolidado queda cerca del azar)."
        )


# ================================================================
# EVALUAR TURNOS DEL DÍA ACTUAL
# ================================================================

def evaluate_today_turns(
    df,
    backtest
):

    """
    IMPORTANTE:

    NO genera nuevas predicciones.

    Simplemente busca las predicciones históricas
    que ya fueron calculadas por el backtest.

    De esta forma se muestran todos los turnos ejecutados
    durante el día en que se ejecuta el análisis.

    """

    if backtest.empty:

        return pd.DataFrame()

    analysis_date = datetime.now().date()

    turns = (
        df[
            df["datetime"].dt.date == analysis_date
        ][
            [
                "datetime",
                "hour"
            ]
        ]
        .drop_duplicates()
        .sort_values(
            "datetime"
        )
    )

    rows = []

    for _, turn in turns.iterrows():

        dt = turn["datetime"]

        match = backtest[
            backtest["datetime"] == dt
        ]

        if match.empty:

            rows.append({

                "datetime": dt,

                "status":
                    "SIN PREDICCION",

                "top10": "",

                "real_animals": "",

                "hit_animals": "",

            })

            continue

        row = match.iloc[0]

        rows.append({

            "datetime":
                dt,

            "date":
                dt.strftime(
                    "%Y-%m-%d"
                ),

            "time":
                dt.strftime(
                    "%H:%M"
                ),

            "status":
                "HIT"
                if bool(row["hit"])
                else "MISS",

            "top10":
                row["top10"],

            "real_animals":
                row["real_animals"],

            "hit_animals":
                row["hit_animals"],

            "top1":
                row["top1"],

            "top1_probability":
                row[
                    "top1_probability"
                ],

        })

    return pd.DataFrame(rows)


# ================================================================
# EVALUAR TURNOS DEL DÍA ACTUAL - TOP 7 POR LOTERÍA
# ================================================================

def evaluate_today_turns_by_lottery(df):

    """
    Para cada turno del día de ejecución:

        1. Genera la predicción de CADA lotería por separado.
        2. Toma los TOP_PER_LOTTERY mejores animales de cada una.
        3. Si un animal ya fue tomado en otra lotería,
           se omite y se toma el siguiente.
        4. Une todo en una sola lista y compara contra
           los resultados reales del turno.

    NO genera predicciones para el futuro:
    únicamente usa resultados anteriores al turno.
    """

    analysis_date = datetime.now().date()

    turns = (
        df[
            df["datetime"].dt.date == analysis_date
        ][
            [
                "datetime",
                "hour"
            ]
        ]
        .drop_duplicates()
        .sort_values(
            "datetime"
        )
    )

    actual_results = build_actual_results(df)

    rows = []

    for _, turn in turns.iterrows():

        dt = turn["datetime"]
        hour = turn["hour"]

        history = df[
            df["datetime"] < dt
        ]

        if history.empty:
            continue

        lotteries = sorted(
            history["lottery"]
            .dropna()
            .unique()
        )

        merged = []
        seen = set()

        for lottery in lotteries:

            prediction = calculate_prediction(
                history=history,
                target_datetime=dt,
                target_hour=hour,
                target_lottery=lottery
            )

            if prediction.empty:
                continue

            taken = 0

            for _, row in prediction.iterrows():

                animal = row["animal"]

                if animal in seen:
                    continue

                merged.append(animal)
                seen.add(animal)
                taken += 1

                if taken >= TOP_PER_LOTTERY:
                    break

        real_animals = actual_results.get(
            dt,
            set()
        )

        hits = [
            a
            for a in merged
            if a in real_animals
        ]

        hit = len(hits) > 0

        rows.append({

            "datetime":
                dt,

            "date":
                dt.strftime(
                    "%Y-%m-%d"
                ),

            "time":
                dt.strftime(
                    "%H:%M"
                ),

            "status":
                "HIT"
                if hit
                else "MISS",

            "top_list":
                ",".join(merged),

            "real_animals":
                ",".join(
                    sorted(real_animals)
                ),

            "hit_animals":
                ",".join(
                    sorted(hits)
                ),

        })

    return pd.DataFrame(rows)


# ================================================================
# TURNOS DEL DÍA ACTUAL — POR LOTERÍA (top-N de cada lotería)
# ================================================================

def evaluate_today_turns_per_lottery(df, top_n):

    """
    Para cada turno del día de ejecución y CADA lotería por separado:

        1. Predice el siguiente sorteo de ESA lotería.
        2. Toma los `top_n` mejores animales de esa lotería.
        3. Compara contra el ganador real de ESA lotería en ese turno.

    Devuelve una fila por (turno, lotería) con HIT/MISS.
    """

    analysis_date = datetime.now().date()

    turns = (
        df[
            df["datetime"].dt.date == analysis_date
        ][
            ["datetime", "hour"]
        ]
        .drop_duplicates()
        .sort_values("datetime")
    )

    actual_lot = build_actual_by_lottery(df)

    rows = []

    for _, turn in turns.iterrows():

        dt = turn["datetime"]
        hour = turn["hour"]

        history = df[
            df["datetime"] < dt
        ]

        if history.empty:
            continue

        for lottery in ["lagranjita", "lottoactivo"]:

            winner = actual_lot.get((dt, lottery))

            prediction = calculate_prediction(
                history=history,
                target_datetime=dt,
                target_hour=hour,
                target_lottery=lottery,
            )

            if prediction.empty:

                top_list = ""

                hit = False

            else:

                top = prediction.head(top_n)

                top_list = ",".join(top["animal"].tolist())

                hit = (
                    winner is not None
                    and winner in top["animal"].tolist()
                )

            rows.append({

                "datetime":
                    dt,

                "date":
                    dt.strftime("%Y-%m-%d"),

                "time":
                    dt.strftime("%H:%M"),

                "lottery":
                    lottery,

                "top_list":
                    top_list,

                "real_animal":
                    winner if winner is not None else "",

                "hit":
                    hit,

            })

    return pd.DataFrame(rows)


# ================================================================
# PROBABILIDAD DE APARECER EN LOS PRÓXIMOS 11 TURNOS
# ================================================================

def eleven_turn_probabilities(history):

    """
    Para cada animal: probabilidad de aparecer AL MENOS UNA VEZ en los
    próximos 11 turnos (consolidado: en cualquiera de las dos loterías).

    Se modela con la curva empírica de anti-repetición r(k): si el animal
    salió hace k turnos en una lotería, su probabilidad por turno en esa
    lotería es base · r(k). Recorriendo 11 turnos:

        P(aparece ≥1 en L) = 1 − Π_{t=1..11} (1 − base_L · r(k_L + t))

    y luego se combina: P = 1 − (1 − P_gr)(1 − P_la).
    """

    num_to_name = {
        v: k
        for k, v in ANIMAL_NUM.items()
    }

    survive = np.ones(38)

    for lot in ["lagranjita", "lottoactivo"]:

        g = history[history["lottery"] == lot].sort_values("datetime")
        seq = g["number"].to_numpy()

        if len(seq) == 0:
            continue

        # ---- curva anti-repetición r(k) ----
        curve = np.ones(25)
        for k in range(1, 25):
            if len(seq) > k:
                rep = (seq[:-k] == seq[k:]).mean()
                curve[k] = max(rep / (1.0 / 38.0), 0.05)

        # ---- base por lotería (Laplace) ----
        cnt = np.bincount(seq, minlength=38).astype(float)
        base = (cnt + 1.0) / (cnt.sum() + 38.0)

        # ---- turnos desde la última aparición en esta lotería ----
        last_pos = np.full(38, -1, dtype=int)
        for pos, num in enumerate(seq):
            last_pos[num] = pos

        t = len(seq)
        k0 = np.where(last_pos >= 0, t - last_pos, 24)
        k0 = np.clip(k0, 1, 24).astype(int)

        # ---- probabilidad de NO aparecer en 11 turnos ----
        no_appear = np.ones(38)
        for step in range(11):
            k = np.clip(k0 + step, 1, 24)
            p = base * curve[k]
            no_appear *= (1.0 - p)

        survive *= no_appear

    appear = 1.0 - survive

    return {
        num_to_name[i]: float(appear[i])
        for i in range(38)
    }


def eleven_turn_prediction(history):

    """DataFrame ordenado de P(aparecer en los próximos 11 turnos)."""

    probs = eleven_turn_probabilities(history)

    rows = [
        {"animal": a, "probability": p}
        for a, p in probs.items()
    ]

    result = (
        pd.DataFrame(rows)
        .sort_values("probability", ascending=False)
        .reset_index(drop=True)
    )

    result["rank"] = np.arange(len(result)) + 1

    return result


def evaluate_eleven_turns_history(df, turns=30):

    """
    Histórico de HIT/MISS del cálculo de 11 turnos: para cada uno de los
    últimos `turns` turnos, se toman los 5 animales con mayor probabilidad
    de aparecer en los siguientes 11 turnos y se comprueba si alguno de
    ellos apareció realmente en esa ventana.
    """

    all_turns = (
        df[["datetime", "hour"]]
        .drop_duplicates()
        .sort_values("datetime")
        .reset_index(drop=True)
    )

    start_turns = all_turns.tail(turns)

    rows = []

    for _, turn in start_turns.iterrows():

        dt = turn["datetime"]

        history = df[df["datetime"] < dt]

        if history.empty:
            continue

        probs = eleven_turn_probabilities(history)

        top5 = sorted(
            probs,
            key=probs.get,
            reverse=True
        )[:5]

        future_turns = (
            all_turns[
                all_turns["datetime"] > dt
            ]["datetime"]
            .head(11)
        )

        if future_turns.empty:
            continue

        future = df[df["datetime"].isin(future_turns)]

        future_animals = set(
            future["animal"].unique()
        )

        hit_animals = [
            a
            for a in top5
            if a in future_animals
        ]

        hit = len(hit_animals) > 0

        rows.append({

            "datetime":
                dt,

            "date":
                dt.strftime("%Y-%m-%d"),

            "time":
                dt.strftime("%H:%M"),

            "top5":
                ",".join(top5),

            "hit_animals":
                ",".join(sorted(hit_animals)),

            "n_hits":
                len(hit_animals),

            "hit":
                hit,

        })

    return pd.DataFrame(rows)


# ================================================================
# SIGUIENTE TURNO REAL
# ================================================================

def get_next_real_turn(df):

    """
    Determina el siguiente turno basándose en los horarios
    que realmente existen en resultados.csv.

    NO inventa horarios.
    """

    last_datetime = df["datetime"].max()

    # Horarios históricos reales
    known_hours = sorted(
        df["hour"]
        .dropna()
        .unique()
    )

    if not known_hours:
        return None

    # ------------------------------------------------------------
    # Buscar siguiente horario del mismo día
    # ------------------------------------------------------------

    last_time = last_datetime.strftime(
        "%H:%M"
    )

    future_hours = [
        h
        for h in known_hours
        if h > last_time
    ]

    if future_hours:

        next_hour = future_hours[0]

        return pd.Timestamp(
            f"{last_datetime.strftime('%Y-%m-%d')} "
            f"{next_hour}"
        )

    # ------------------------------------------------------------
    # Si no quedan horarios:
    # primer turno del siguiente día
    # ------------------------------------------------------------

    first_hour = known_hours[0]

    next_day = (
        last_datetime
        + timedelta(days=1)
    ).strftime(
        "%Y-%m-%d"
    )

    return pd.Timestamp(
        f"{next_day} {first_hour}"
    )


# ================================================================
# PREDICCIÓN SIGUIENTE TURNO
# ================================================================

def predict_next_turn(df):

    next_turn = get_next_real_turn(
        df
    )

    if next_turn is None:

        return (
            None,
            pd.DataFrame()
        )

    history = df[
        df["datetime"] < next_turn
    ]

    prediction = calculate_prediction(

        history=history,

        target_datetime=next_turn,

        target_hour=
            next_turn.strftime(
                "%H:%M"
            ),

        target_lottery=None
    )

    if prediction.empty:

        return (
            next_turn,
            prediction
        )

    prediction = prediction.copy()

    prediction["datetime"] = next_turn

    prediction["date"] = (
        next_turn.strftime(
            "%Y-%m-%d"
        )
    )

    prediction["time"] = (
        next_turn.strftime(
            "%H:%M"
        )
    )

    return (
        next_turn,
        prediction
    )


# ================================================================
# PATRONES
# ================================================================

def generate_patterns(
    df,
    backtest
):

    rows = []

    # ------------------------------------------------------------
    # Tasa global
    # ------------------------------------------------------------

    if not backtest.empty:

        total = len(backtest)

        hits = int(
            backtest["hit"].sum()
        )

        rate = (
            hits / total
            if total
            else 0
        )

        rows.append({

            "pattern":
                "TOP_10_GLOBAL",

            "description":
                "Al menos uno de los Top 10 "
                "coincide con el resultado "
                "del turno",

            "samples":
                total,

            "hits":
                hits,

            "hit_rate":
                rate,

        })

    # ------------------------------------------------------------
    # Mejor hora
    # ------------------------------------------------------------

    if not backtest.empty:

        grouped = (
            backtest
            .groupby("time")
            .agg(
                samples=("hit", "size"),
                hits=("hit", "sum")
            )
            .reset_index()
        )

        grouped["hit_rate"] = (
            grouped["hits"]
            / grouped["samples"]
        )

        for _, row in grouped.iterrows():

            rows.append({

                "pattern":
                    "HORA",

                "description":
                    f"Hora {row['time']}",

                "samples":
                    int(row["samples"]),

                "hits":
                    int(row["hits"]),

                "hit_rate":
                    float(row["hit_rate"]),

            })

    return pd.DataFrame(rows)


# ================================================================
# GENERAR CSVs
# ================================================================

def save_csvs(
    df,
    backtest,
    last5,
    today_by_lottery,
    next_prediction,
    patterns
):

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    # ------------------------------------------------------------
    # Backtest
    # ------------------------------------------------------------

    backtest.to_csv(

        os.path.join(
            OUTPUT_DIR,
            "backtest.csv"
        ),

        index=False,

        encoding="utf-8-sig"
    )

    # ------------------------------------------------------------
    # Turnos del día actual
    # ------------------------------------------------------------

    last5.to_csv(

        os.path.join(
            OUTPUT_DIR,
            "turnos_hoy.csv"
        ),

        index=False,

        encoding="utf-8-sig"
    )

    # ------------------------------------------------------------
    # Turnos del día actual - Top 7 por lotería
    # ------------------------------------------------------------

    today_by_lottery.to_csv(

        os.path.join(
            OUTPUT_DIR,
            "turnos_hoy_top7_loteria.csv"
        ),

        index=False,

        encoding="utf-8-sig"
    )

    # ------------------------------------------------------------
    # Predicción siguiente turno
    # ------------------------------------------------------------

    next_prediction.to_csv(

        os.path.join(
            OUTPUT_DIR,
            "prob-animal-next-turn.csv"
        ),

        index=False,

        encoding="utf-8-sig"
    )

    # ------------------------------------------------------------
    # Patrones
    # ------------------------------------------------------------

    patterns.to_csv(

        os.path.join(
            OUTPUT_DIR,
            "patrones.csv"
        ),

        index=False,

        encoding="utf-8-sig"
    )


# ================================================================
# HTML
# ================================================================

def _next_turn_table_rows(prediction):

    """
    Filas HTML de una tabla de predicción del siguiente turno
    (# · Animal · Probabilidad con barra).
    """

    rows = ""

    if prediction is None or prediction.empty:
        return rows

    for _, row in prediction.head(TOP_N).iterrows():

        probability = row["probability"] * 100

        width = min(probability * 3, 100)

        rows += f"""
            <tr>
                <td class="rank">
                    {int(row["rank"])}
                </td>
                <td class="animal">
                    {html.escape(str(row["animal"]))}
                </td>
                <td>
                    <div class="prob">
                        {probability:.2f}%
                    </div>
                    <div class="bar">
                        <div
                            class="bar-fill"
                            style="width:{width:.2f}%"
                        ></div>
                    </div>
                </td>
            </tr>
            """

    return rows


def _today_lottery_rows(dframe):

    """
    Filas HTML de turnos de hoy por lotería (HIT/MISS),
    con columna de lotería.
    """

    rows = ""

    if dframe is None or dframe.empty:
        return rows

    for _, row in (
        dframe
        .sort_values("datetime", ascending=False)
        .iterrows()
    ):

        hit = bool(row["hit"])

        status = "HIT" if hit else "MISS"

        css = "hit" if hit else "miss"

        icon = "✓" if hit else "✗"

        rows += f"""
            <tr>
                <td>
                    {row["datetime"].strftime("%Y-%m-%d")}
                </td>
                <td>
                    {row["datetime"].strftime("%H:%M")}
                </td>
                <td>
                    {html.escape(str(row["lottery"]))}
                </td>
                <td>
                    {html.escape(str(row["top_list"]))}
                </td>
                <td>
                    {html.escape(str(row["real_animal"]))}
                </td>
                <td>
                    <span class="status {css}">
                        {icon} {status}
                    </span>
                </td>
            </tr>
            """

    return rows


def _eleven_turn_rows(pred_df):

    """
    Filas HTML de la tabla de probabilidad de aparecer en 11 turnos.
    """

    rows = ""

    if pred_df is None or pred_df.empty:
        return rows

    for _, row in pred_df.head(TOP_N).iterrows():

        probability = row["probability"] * 100

        width = min(probability * 1.6, 100)

        rows += f"""
            <tr>
                <td class="rank">
                    {int(row["rank"])}
                </td>
                <td class="animal">
                    {html.escape(str(row["animal"]))}
                </td>
                <td>
                    <div class="prob">
                        {probability:.1f}%
                    </div>
                    <div class="bar">
                        <div
                            class="bar-fill"
                            style="width:{width:.1f}%"
                        ></div>
                    </div>
                </td>
            </tr>
            """

    return rows


def _eleven_history_rows(dframe):

    """
    Filas HTML del histórico HIT/MISS del cálculo de 11 turnos.
    """

    rows = ""

    if dframe is None or dframe.empty:
        return rows

    for _, row in (
        dframe
        .sort_values("datetime", ascending=False)
        .iterrows()
    ):

        hit = bool(row["hit"])

        status = "HIT" if hit else "MISS"

        css = "hit" if hit else "miss"

        icon = "✓" if hit else "✗"

        rows += f"""
            <tr>
                <td>
                    {row["datetime"].strftime("%Y-%m-%d")}
                </td>
                <td>
                    {row["datetime"].strftime("%H:%M")}
                </td>
                <td>
                    {html.escape(str(row["top5"]))}
                </td>
                <td>
                    {int(row["n_hits"])}/5
                </td>
                <td>
                    {html.escape(str(row["hit_animals"]))}
                </td>
                <td>
                    <span class="status {css}">
                        {icon} {status}
                    </span>
                </td>
            </tr>
            """

    return rows


def generate_html(
    df,
    backtest,
    last5,
    today_by_lottery,
    next_turn,
    next_prediction,
    patterns,
    next_pred_gr=None,
    next_pred_la=None,
    today_top5=None,
    today_top7=None,
    eleven_pred=None,
    eleven_history=None,
):

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    # ------------------------------------------------------------
    # Estadísticas
    # ------------------------------------------------------------

    total_backtest = len(
        backtest
    )

    total_hits = (
        int(
            backtest["hit"].sum()
        )
        if not backtest.empty
        else 0
    )

    hit_rate = (
        total_hits
        / total_backtest
        if total_backtest
        else 0
    )

    last_result_datetime = (
        df["datetime"].max()
    )

    last_result = df[
        df["datetime"]
        == last_result_datetime
    ]

    real_last_animals = sorted(
        last_result["animal"]
        .unique()
    )

    # ------------------------------------------------------------
    # TOP siguiente turno
    # ------------------------------------------------------------

    next_rows = ""

    if not next_prediction.empty:

        for _, row in (
            next_prediction
            .head(TOP_N)
            .iterrows()
        ):

            probability = (
                row["probability"]
                * 100
            )

            width = min(
                probability * 3,
                100
            )

            next_rows += f"""
            <tr>
                <td class="rank">
                    {int(row["rank"])}
                </td>

                <td class="animal">
                    {html.escape(
                        str(row["animal"])
                    )}
                </td>

                <td>
                    <div class="prob">
                        {probability:.2f}%
                    </div>

                    <div class="bar">
                        <div
                            class="bar-fill"
                            style="width:{width:.2f}%"
                        ></div>
                    </div>
                </td>

                <td>
                    {row["hour"] if "hour" in row else ""}
                </td>

                <td>
                    {row["historical"]:.3f}
                </td>

                <td>
                    {row["recent"]:.3f}
                </td>

                <td>
                    {row["hour"]:.3f}
                </td>

                <td>
                    {row["cross_lottery"]:.3f}
                </td>
            </tr>
            """

    # ------------------------------------------------------------
    # Turnos del día actual
    # ------------------------------------------------------------

    history_rows = ""

    if not last5.empty:

        for _, row in (
            last5
            .sort_values(
                "datetime",
                ascending=False
            )
            .iterrows()
        ):

            status = row["status"]

            if status == "HIT":
                css = "hit"
                icon = "✓"

            elif status == "MISS":
                css = "miss"
                icon = "✗"

            else:
                css = "none"
                icon = "?"

            history_rows += f"""
            <tr>

                <td>
                    {row["datetime"].strftime(
                        "%Y-%m-%d"
                    )}
                </td>

                <td>
                    {row["datetime"].strftime(
                        "%H:%M"
                    )}
                </td>

                <td>
                    {html.escape(
                        str(row["top10"])
                    )}
                </td>

                <td>
                    {html.escape(
                        str(row["real_animals"])
                    )}
                </td>

                <td>
                    {html.escape(
                        str(row["hit_animals"])
                    )}
                </td>

                <td>
                    <span class="status {css}">
                        {icon} {status}
                    </span>
                </td>

            </tr>
            """

    # ------------------------------------------------------------
    # Turnos del día actual - Top 7 por lotería
    # ------------------------------------------------------------

    lottery_rows = ""

    if not today_by_lottery.empty:

        for _, row in (
            today_by_lottery
            .sort_values(
                "datetime",
                ascending=False
            )
            .iterrows()
        ):

            status = row["status"]

            if status == "HIT":
                css = "hit"
                icon = "✓"

            elif status == "MISS":
                css = "miss"
                icon = "✗"

            else:
                css = "none"
                icon = "?"

            lottery_rows += f"""
            <tr>

                <td>
                    {row["datetime"].strftime(
                        "%Y-%m-%d"
                    )}
                </td>

                <td>
                    {row["datetime"].strftime(
                        "%H:%M"
                    )}
                </td>

                <td>
                    {html.escape(
                        str(row["top_list"])
                    )}
                </td>

                <td>
                    {html.escape(
                        str(row["real_animals"])
                    )}
                </td>

                <td>
                    {html.escape(
                        str(row["hit_animals"])
                    )}
                </td>

                <td>
                    <span class="status {css}">
                        {icon} {status}
                    </span>
                </td>

            </tr>
            """

    # ------------------------------------------------------------
    # Próximo turno — por lotería
    # ------------------------------------------------------------

    next_rows_gr = _next_turn_table_rows(
        next_pred_gr
    )

    next_rows_la = _next_turn_table_rows(
        next_pred_la
    )

    # ------------------------------------------------------------
    # Turnos de hoy — top-N por lotería
    # ------------------------------------------------------------

    top5_rows = _today_lottery_rows(
        today_top5
    )

    top7_rows = _today_lottery_rows(
        today_top7
    )

    # ------------------------------------------------------------
    # Próximos 11 turnos
    # ------------------------------------------------------------

    eleven_rows = _eleven_turn_rows(
        eleven_pred
    )

    eleven_hist_rows = _eleven_history_rows(
        eleven_history
    )

    # ------------------------------------------------------------
    # Patrones
    # ------------------------------------------------------------

    pattern_rows = ""

    if not patterns.empty:

        patterns_sorted = (
            patterns
            .sort_values(
                "hit_rate",
                ascending=False
            )
            .head(15)
        )

        for _, row in (
            patterns_sorted.iterrows()
        ):

            pattern_rows += f"""
            <tr>

                <td>
                    {html.escape(
                        str(row["pattern"])
                    )}
                </td>

                <td>
                    {html.escape(
                        str(row["description"])
                    )}
                </td>

                <td>
                    {int(row["samples"])}
                </td>

                <td>
                    {int(row["hits"])}
                </td>

                <td>
                    {row["hit_rate"] * 100:.2f}%
                </td>

            </tr>
            """

    # ------------------------------------------------------------
    # JSON para JS
    # ------------------------------------------------------------

    chart_labels = []

    chart_values = []

    if not backtest.empty:

        ordered = (
            backtest
            .sort_values(
                "datetime"
            )
            .tail(30)
        )

        for _, row in ordered.iterrows():

            chart_labels.append(
                row["datetime"].strftime(
                    "%m-%d %H:%M"
                )
            )

            chart_values.append(
                1 if row["hit"] else 0
            )

    chart_labels_json = json.dumps(
        chart_labels
    )

    chart_values_json = json.dumps(
        chart_values
    )

    # ------------------------------------------------------------
    # HTML
    # ------------------------------------------------------------

    html_content = f"""
<!DOCTYPE html>

<html lang="es">

<head>

<meta charset="UTF-8">

<meta name="viewport"
      content="width=device-width, initial-scale=1.0">

<title>
V7 - Predictor de Animales
</title>

<style>

* {{
    box-sizing: border-box;
}}

body {{
    margin: 0;
    padding: 0;

    font-family:
        Arial,
        Helvetica,
        sans-serif;

    background:
        #f4f6f8;

    color:
        #1f2937;
}}

.container {{
    max-width: 1500px;

    margin:
        0 auto;

    padding:
        30px;
}}

.header {{
    background:
        linear-gradient(
            135deg,
            #111827,
            #374151
        );

    color:
        white;

    padding:
        30px;

    border-radius:
        18px;

    margin-bottom:
        25px;
}}

.header h1 {{
    margin:
        0 0 10px 0;
}}

.header p {{
    opacity:
        .8;
}}

.cards {{
    display:
        grid;

    grid-template-columns:
        repeat(
            auto-fit,
            minmax(
                200px,
                1fr
            )
        );

    gap:
        18px;

    margin-bottom:
        25px;
}}

.card {{
    background:
        white;

    border-radius:
        14px;

    padding:
        22px;

    box-shadow:
        0 4px 15px
        rgba(
            0,
            0,
            0,
            .06
        );
}}

.card .value {{
    font-size:
        30px;

    font-weight:
        bold;

    margin-top:
        8px;
}}

.section {{
    background:
        white;

    padding:
        25px;

    border-radius:
        16px;

    margin-bottom:
        25px;

    box-shadow:
        0 4px 15px
        rgba(
            0,
            0,
            0,
            .05
        );
}}

.section h2 {{
    margin-top:
        0;
}}

table {{
    width:
        100%;

    border-collapse:
        collapse;
}}

th,
td {{
    padding:
        12px;

    border-bottom:
        1px solid
        #e5e7eb;

    text-align:
        left;
}}

th {{
    background:
        #f9fafb;
}}

.rank {{
    font-weight:
        bold;

    font-size:
        18px;
}}

.animal {{
    font-weight:
        bold;

    font-size:
        16px;
}}

.prob {{
    font-weight:
        bold;
}}

.bar {{
    width:
        100%;

    max-width:
        300px;

    height:
        8px;

    background:
        #e5e7eb;

    border-radius:
        5px;

    margin-top:
        5px;

    overflow:
        hidden;
}}

.bar-fill {{
    height:
        100%;

    background:
        #2563eb;

    border-radius:
        5px;
}}

.status {{
    display:
        inline-block;

    padding:
        6px 12px;

    border-radius:
        20px;

    font-weight:
        bold;
}}

.hit {{
    background:
        #dcfce7;

    color:
        #166534;
}}

.miss {{
    background:
        #fee2e2;

    color:
        #991b1b;
}}

.none {{
    background:
        #f3f4f6;

    color:
        #6b7280;
}}

.next {{
    border:
        3px solid
        #2563eb;
}}

.highlight {{
    font-size:
        24px;

    font-weight:
        bold;

    margin:
        10px 0;
}}

canvas {{
    width:
        100%;

    height:
        380px;
    
    border: 1px solid #dbe2e8;
    border-radius: 6px;
}}

@media(max-width:900px) {{

    .container {{
        padding:
            15px;
    }}

    table {{
        font-size:
            12px;
    }}

    th,
    td {{
        padding:
            7px;
    }}

}}

</style>

</head>

<body>

<div class="container">

<div class="header">

<h1>
🐾 V7 — Predictor de Animales
</h1>

<p>
Modelo probabilístico basado en histórico,
hora, recencia, tendencia e influencia entre loterías.
</p>

<p>
Última actualización:
<strong>
{datetime.now().strftime(
    "%Y-%m-%d %H:%M:%S"
)}
</strong>
</p>

</div>


<!-- ========================================================= -->
<!-- RESUMEN -->
<!-- ========================================================= -->

<div class="cards">

<div class="card">

<div>
Último resultado
</div>

<div class="value">
{last_result_datetime.strftime(
    "%Y-%m-%d %H:%M"
)}
</div>

</div>


<div class="card">

<div>
Animales resultado
</div>

<div class="value">
{html.escape(
    ", ".join(real_last_animals)
)}
</div>

</div>


<div class="card">

<div>
Backtest
</div>

<div class="value">
{total_backtest}
</div>

</div>


<div class="card">

<div>
Hits
</div>

<div class="value">
{total_hits}
</div>

</div>


<div class="card">

<div>
Tasa de aciertos
</div>

<div class="value">
{hit_rate * 100:.2f}%
</div>

</div>

</div>


<!-- ========================================================= -->
<!-- SIGUIENTE TURNO -->
<!-- ========================================================= -->

<div class="section next">

<h2>
🎯 Próximo turno real
</h2>

<div class="highlight">

{
    next_turn.strftime(
        "%Y-%m-%d %H:%M"
    )
    if next_turn is not None
    else "No disponible"
}

</div>

<p>
Esta es la única predicción futura que genera
el V7.
</p>

<table>

<thead>

<tr>

<th>
#
</th>

<th>
Animal
</th>

<th>
Probabilidad
</th>

<th>
Hora
</th>

<th>
Histórico
</th>

<th>
Reciente
</th>

<th>
Hora
</th>

<th>
Cross Lottery
</th>

</tr>

</thead>

<tbody>

{next_rows}

</tbody>

</table>

</div>


<!-- ========================================================= -->
<!-- SIGUIENTE TURNO — POR LOTERÍA -->
<!-- ========================================================= -->

<div class="section next">

<h2>
🎯 Próximo turno real — lagranjita
</h2>

<div class="highlight">

{
    next_turn.strftime(
        "%Y-%m-%d %H:%M"
    )
    if next_turn is not None
    else "No disponible"
}

</div>

<table>

<thead>

<tr>

<th>
#
</th>

<th>
Animal
</th>

<th>
Probabilidad
</th>

</tr>

</thead>

<tbody>

{next_rows_gr}

</tbody>

</table>

</div>


<div class="section next">

<h2>
🎯 Próximo turno real — lottoactivo
</h2>

<div class="highlight">

{
    next_turn.strftime(
        "%Y-%m-%d %H:%M"
    )
    if next_turn is not None
    else "No disponible"
}

</div>

<table>

<thead>

<tr>

<th>
#
</th>

<th>
Animal
</th>

<th>
Probabilidad
</th>

</tr>

</thead>

<tbody>

{next_rows_la}

</tbody>

</table>

</div>


<!-- ========================================================= -->
<!-- PRÓXIMOS 11 TURNOS -->
<!-- ========================================================= -->

<div class="section next">

<h2>
🎯 Próximos 11 turnos — probabilidad de aparecer
</h2>

<div class="highlight">

{
    next_turn.strftime(
        "%Y-%m-%d %H:%M"
    )
    if next_turn is not None
    else "No disponible"
}

</div>

<p>
Probabilidad de que cada animal aparezca AL MENOS UNA VEZ
en los próximos 11 turnos (en cualquiera de las dos loterías).
</p>

<table>

<thead>

<tr>

<th>
#
</th>

<th>
Animal
</th>

<th>
Probabilidad (11 turnos)
</th>

</tr>

</thead>

<tbody>

{eleven_rows}

</tbody>

</table>

</div>


<!-- ========================================================= -->
<!-- TURNOS DEL DÍA ACTUAL -->
<!-- ========================================================= -->

<div class="section">

<h2>
📊 Turnos de hoy — HIT / MISS
</h2>

<p>
Se muestran únicamente predicciones que ya existían
para esos turnos. No se generan predicciones nuevas
para los turnos mostrados.
</p>

<table>

<thead>

<tr>

<th>
Fecha
</th>

<th>
Hora
</th>

<th>
Top 10 predicho
</th>

<th>
Resultado real
</th>

<th>
Coincidencia
</th>

<th>
Resultado
</th>

</tr>

</thead>

<tbody>

{history_rows}

</tbody>

</table>

</div>


<!-- ========================================================= -->
<!-- TURNOS DEL DÍA ACTUAL - TOP 5 POR LOTERÍA -->
<!-- ========================================================= -->

<div class="section">

<h2>
📊 Turnos de hoy — HIT / MISS por lotería (Top 5)
</h2>

<p>
Se toman los 5 mejores animales de CADA lotería por separado
y se comparan contra el ganador real de esa lotería.
</p>

<table>

<thead>

<tr>

<th>
Fecha
</th>

<th>
Hora
</th>

<th>
Lotería
</th>

<th>
Top 5 predicho
</th>

<th>
Resultado real
</th>

<th>
Resultado
</th>

</tr>

</thead>

<tbody>

{top5_rows}

</tbody>

</table>

</div>


<!-- ========================================================= -->
<!-- TURNOS DEL DÍA ACTUAL - TOP 7 POR LOTERÍA -->
<!-- ========================================================= -->

<div class="section">

<h2>
🏦 Turnos de hoy — Top 7 por lotería
</h2>

<p>
Se toman los 7 mejores animales de cada lotería.
Si un animal se repite entre loterías, se omite
y se toma el siguiente. Luego se unen en una sola
lista y se compara contra el resultado real.
</p>

<table>

<thead>

<tr>

<th>
Fecha
</th>

<th>
Hora
</th>

<th>
Top 7 por lotería predicho
</th>

<th>
Resultado real
</th>

<th>
Coincidencia
</th>

<th>
Resultado
</th>

</tr>

</thead>

<tbody>

{lottery_rows}

</tbody>

</table>

</div>


<!-- ========================================================= -->
<!-- TURNOS DEL DÍA ACTUAL - TOP 7 POR LOTERÍA (SEPARADOS) -->
<!-- ========================================================= -->

<div class="section">

<h2>
🏦 Turnos de hoy — Top 7 por lotería (separados)
</h2>

<p>
Cada lotería por separado: sus 7 mejores probabilidades,
comparadas contra su propio ganador real.
</p>

<table>

<thead>

<tr>

<th>
Fecha
</th>

<th>
Hora
</th>

<th>
Lotería
</th>

<th>
Top 7 predicho
</th>

<th>
Resultado real
</th>

<th>
Resultado
</th>

</tr>

</thead>

<tbody>

{top7_rows}

</tbody>

</table>

</div>


<!-- ========================================================= -->
<!-- HISTÓRICO 11 TURNOS -->
<!-- ========================================================= -->

<div class="section">

<h2>
📅 Histórico 11 turnos — HIT / MISS (últimos 30 turnos)
</h2>

<p>
Para cada turno: los 5 animales con mayor probabilidad de aparecer
en los siguientes 11 turnos, y cuántos de ellos aparecieron realmente.
</p>

<table>

<thead>

<tr>

<th>
Fecha
</th>

<th>
Hora
</th>

<th>
Top 5 (11 turnos)
</th>

<th>
Aciertos
</th>

<th>
Animales acertados
</th>

<th>
Resultado
</th>

</tr>

</thead>

<tbody>

{eleven_hist_rows}

</tbody>

</table>

</div>


<!-- ========================================================= -->
<!-- GRÁFICA -->
<!-- ========================================================= -->

<div class="section">

<h2>
📈 Evolución de HIT / MISS
</h2>

<canvas
    id="hitChart">
</canvas>

</div>


<!-- ========================================================= -->
<!-- PATRONES -->
<!-- ========================================================= -->

<div class="section">

<h2>
🔎 Patrones encontrados
</h2>

<table>

<thead>

<tr>

<th>
Patrón
</th>

<th>
Descripción
</th>

<th>
Muestras
</th>

<th>
Hits
</th>

<th>
Tasa
</th>

</tr>

</thead>

<tbody>

{pattern_rows}

</tbody>

</table>

</div>


<!-- ========================================================= -->
<!-- METODOLOGÍA -->
<!-- ========================================================= -->

<div class="section">

<h2>
🧠 Cómo funciona el modelo
</h2>

<ul>

<li>
<strong>Histórico:</strong>
frecuencia global del animal.
</li>

<li>
<strong>Reciente:</strong>
comportamiento de los últimos resultados.
</li>

<li>
<strong>Hora:</strong>
comportamiento específico del animal
en el horario objetivo.
</li>

<li>
<strong>Recencia:</strong>
cuánto tiempo ha pasado desde la última
aparición del animal.
</li>

<li>
<strong>Tendencia:</strong>
compara el comportamiento reciente
contra el histórico.
</li>

<li>
<strong>Cross Lottery:</strong>
busca relaciones entre resultados
de diferentes loterías.
</li>

<li>
<strong>Sin data leakage:</strong>
al predecir un turno solamente se utilizan
resultados anteriores a ese turno.
</li>

</ul>

</div>


</div>


<script>

const labels =
    {chart_labels_json};

const values =
    {chart_values_json};


const canvas =
    document.getElementById(
        "hitChart"
    );

const ctx =
    canvas.getContext(
        "2d"
    );


function resizeCanvas() {{

    const rect =
        canvas.getBoundingClientRect();

    canvas.width =
        rect.width * window.devicePixelRatio;

    canvas.height =
        380 * window.devicePixelRatio;

    ctx.scale(
        window.devicePixelRatio,
        window.devicePixelRatio
    );

    drawChart(
        rect.width,
        380
    );
}}


function drawChart(width, height) {{
    ctx.clearRect(0, 0, width, height);
    if (!values.length) return;

    const padding = 60;
    const chartWidth = width - padding * 2;
    const chartHeight = height - padding * 2;
    const barWidth = Math.max(chartWidth / values.length - 2, 2);
    const barMaxHeight = chartHeight - 40;

    // Ejes
    ctx.strokeStyle = '#dbe2e8';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(padding, padding);
    ctx.lineTo(padding, height - padding);
    ctx.lineTo(width - padding, height - padding);
    ctx.stroke();

    // Etiquetas eje Y
    ctx.fillStyle = '#6b7280';
    ctx.font = '11px Arial';
    ctx.textAlign = 'right';
    ctx.fillText('100%', padding - 10, padding + 5);
    ctx.fillText('50%', padding - 10, padding + (chartHeight / 2) + 5);
    ctx.fillText('0%', padding - 10, height - padding + 5);

    // Línea referencia
    ctx.strokeStyle = '#f0f0f0';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(padding, padding + (chartHeight / 2));
    ctx.lineTo(width - padding, padding + (chartHeight / 2));
    ctx.stroke();

    // Barras
    values.forEach((value, index) => {{
        const x = padding + (index * barWidth) + (barWidth / 2);
        const barHeight = value === 1 ? barMaxHeight : barMaxHeight / 2;
        const y = height - padding - barHeight;

        ctx.fillStyle = value ? '#18794e' : '#b42318';
        ctx.fillRect(x - (barWidth / 2) + 1, y, barWidth - 2, barHeight);

        if (index % Math.max(1, Math.floor(values.length / 8)) === 0) {{
            ctx.save();
            ctx.translate(x, height - padding + 8);
            ctx.rotate(-Math.PI / 4);
            ctx.textAlign = 'right';
            ctx.fillStyle = '#6b7280';
            ctx.font = '10px Arial';
            ctx.fillText(labels[index], 0, 0);
            ctx.restore();
        }}
    }});
}}


window.addEventListener(
    "resize",
    resizeCanvas
);

resizeCanvas();

</script>

</body>

</html>
"""

    html_file = os.path.join(
        OUTPUT_DIR,
        "dashboard.html"
    )

    with open(
        html_file,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            html_content
        )

    return html_file


# ================================================================
# MAIN
# ================================================================

def main():

    print("")
    print("=" * 70)
    print("V7 - ANALISIS PROBABILISTICO")
    print("=" * 70)

    # ------------------------------------------------------------
    # Cargar
    # ------------------------------------------------------------

    print("")
    print("Cargando resultados.csv...")

    df = load_data()

    print(
        f"Resultados cargados: {len(df):,}"
    )

    print(
        f"Desde: {df['datetime'].min()}"
    )

    print(
        f"Hasta: {df['datetime'].max()}"
    )

    print(
        f"Animales: "
        f"{df['animal'].nunique()}"
    )

    print(
        f"Loterías: "
        f"{df['lottery'].nunique()}"
    )

    # ------------------------------------------------------------
    # Turnos
    # ------------------------------------------------------------

    turns = get_real_turns(
        df
    )

    print(
        f"Turnos reales: "
        f"{len(turns):,}"
    )

    # ------------------------------------------------------------
    # Backtest
    # ------------------------------------------------------------

    backtest = run_backtest(
        df
    )

    # ------------------------------------------------------------
    # Turnos del día actual
    # ------------------------------------------------------------

    today_turns = evaluate_today_turns(
        df,
        backtest
    )

    today_by_lottery = evaluate_today_turns_by_lottery(
        df
    )

    today_top5 = evaluate_today_turns_per_lottery(
        df,
        5
    )

    today_top7 = evaluate_today_turns_per_lottery(
        df,
        7
    )

    # ------------------------------------------------------------
    # Histórico del cálculo de 11 turnos (últimos 30 turnos)
    # ------------------------------------------------------------

    eleven_history = evaluate_eleven_turns_history(
        df,
        30
    )

    # ------------------------------------------------------------
    # Próximo turno
    # ------------------------------------------------------------

    print("")
    print("=" * 70)
    print("PREDICCIÓN DEL SIGUIENTE TURNO")
    print("=" * 70)

    next_pred_gr = None
    next_pred_la = None

    next_turn, next_prediction = (
        predict_next_turn(df)
    )

    if next_turn is not None:

        print(
            f"\nSiguiente turno: "
            f"{next_turn}"
        )

        print("")

        history = df[
            df["datetime"] < next_turn
        ]

        # Predicción POR LOTERÍA (donde está la señal de anti-repetición).
        for lottery in ["lagranjita", "lottoactivo"]:

            pred_lot = calculate_prediction(
                history=history,
                target_datetime=next_turn,
                target_hour=next_turn.strftime("%H:%M"),
                target_lottery=lottery,
            )

            if lottery == "lagranjita":
                next_pred_gr = pred_lot
            else:
                next_pred_la = pred_lot

            print(
                f"--- {lottery} (top {TOP_N}) ---"
            )

            for _, row in pred_lot.head(TOP_N).iterrows():

                print(
                    f"  {int(row['rank']):2d}. "
                    f"{row['animal']:<15} "
                    f"{row['probability'] * 100:6.2f}%"
                )

            print("")

        print("--- Consolidado (prob. de salir en AL MENOS una lotería) ---")

        for _, row in (
            next_prediction
            .head(TOP_N)
            .iterrows()
        ):

            print(
                f"  {int(row['rank']):2d}. "
                f"{row['animal']:<15} "
                f"{row['probability'] * 100:6.2f}%"
            )

    # ------------------------------------------------------------
    # Probabilidad de aparecer en los próximos 11 turnos
    # ------------------------------------------------------------

    eleven_pred = eleven_turn_prediction(df)

    # ------------------------------------------------------------
    # Patrones
    # ------------------------------------------------------------

    patterns = generate_patterns(
        df,
        backtest
    )

    # ------------------------------------------------------------
    # CSV
    # ------------------------------------------------------------

    print("")
    print("Generando CSVs...")

    save_csvs(
        df,
        backtest,
        today_turns,
        today_by_lottery,
        next_prediction,
        patterns
    )

    # ------------------------------------------------------------
    # HTML
    # ------------------------------------------------------------

    print(
        "Generando dashboard..."
    )

    html_file = generate_html(

        df=df,

        backtest=backtest,

        last5=today_turns,

        today_by_lottery=
            today_by_lottery,

        next_turn=next_turn,

        next_prediction=
            next_prediction,

        next_pred_gr=next_pred_gr,

        next_pred_la=next_pred_la,

        today_top5=today_top5,

        today_top7=today_top7,

        eleven_pred=eleven_pred,

        eleven_history=eleven_history,

        patterns=patterns
    )

    # ------------------------------------------------------------
    # Resumen
    # ------------------------------------------------------------

    print("")
    print("=" * 70)
    print("PROCESO TERMINADO")
    print("=" * 70)

    print("")
    print(
        f"Resultados: "
        f"{len(df):,}"
    )

    print(
        f"Turnos backtest: "
        f"{len(backtest):,}"
    )

    if not backtest.empty:

        hits = int(
            backtest["hit"].sum()
        )

        rate = (
            hits
            /
            len(backtest)
            *
            100
        )

        print(
            f"Hits: {hits:,}"
        )

        print(
            f"Tasa HIT: {rate:.2f}%"
        )

    print("")
    print(
        f"Dashboard: "
        f"{html_file}"
    )

    print("")
    print(
        "Archivos generados en:"
    )

    print(
        os.path.abspath(
            OUTPUT_DIR
        )
    )

    print("")
    print("=" * 70)


# ================================================================
# EJECUCIÓN
# ================================================================

if __name__ == "__main__":

    main()