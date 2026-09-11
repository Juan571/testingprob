#!/usr/bin/env python3
"""Incremental updater: append new results to resultados.csv up to current datetime.

Usage:
  python scrape_loterias_incremental.py --file resultados.csv

If `resultados.csv` does not exist, the script exits with a message requesting to run `scrape_loterias.py` first.
"""
from __future__ import annotations

import csv
import datetime
import os
import re
import sys
import time
from typing import Dict, List

import requests
from bs4 import BeautifulSoup

try:
    from curl_cffi import requests as curl_requests
    HAS_CURL_CFFI = True
except Exception:  # pragma: no cover
    HAS_CURL_CFFI = False


DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": (
        "text/html,application/xhtml+xml,application/xml;q=0.9,"
        "image/avif,image/webp,image/apng,*/*;q=0.8"
    ),
    "Accept-Language": "es-ES,es;q=0.9,en;q=0.8",
    "Referer": "https://loteriadehoy.com/",
    "Upgrade-Insecure-Requests": "1",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "same-origin",
    "Cache-Control": "max-age=0",
}

BASE_URL = "https://loteriadehoy.com"


# Número oficial (0-37) de cada animal en la tabla estándar de animalitos
ANIMAL_NUM = {
    "delfin": "0", "carnero": "1", "toro": "2", "ciempies": "3",
    "alacran": "4", "leon": "5", "rana": "6", "perico": "7",
    "raton": "8", "aguila": "9", "tigre": "10", "gato": "11",
    "caballo": "12", "mono": "13", "paloma": "14", "zorro": "15",
    "oso": "16", "pavo": "17", "burro": "18", "chivo": "19",
    "cochino": "20", "gallo": "21", "camello": "22", "cebra": "23",
    "iguana": "24", "gallina": "25", "vaca": "26", "perro": "27",
    "zamuro": "28", "elefante": "29", "caiman": "30", "lapa": "31",
    "ardilla": "32", "pescado": "33", "venado": "34", "jirafa": "35",
    "culebra": "36", "ballena": "37",
}

# Variantes de escritura -> nombre canónico
ANIMAL_ALIASES = {
    "zebra": "cebra",
    "loro": "perico",
}


def normalize_animal(name: str) -> str:
    a = name.strip().lower()
    return ANIMAL_ALIASES.get(a, a)


def fetch_date_html(date_str: str, cookies: Dict[str, str] | None = None) -> str:
    url = f"{BASE_URL}/animalitos/resultados/{date_str}/"
    last_error: Exception | None = None

    # 1) curl_cffi con impersonación de Chrome. Esto evade el filtro por
    #    huella TLS de Cloudflare, que bloquea con 403 las IPs de datacenter
    #    de los runners de GitHub Actions.
    if HAS_CURL_CFFI:
        try:
            session = curl_requests.Session(impersonate="chrome")
            # Visita previa a la home para recoger las cookies de Cloudflare.
            session.get(BASE_URL + "/", headers=DEFAULT_HEADERS, timeout=20)
            resp = session.get(url, headers=DEFAULT_HEADERS, cookies=cookies, timeout=20)
            resp.raise_for_status()
            return resp.text
        except Exception as exc:  # pragma: no cover
            last_error = exc

    # 2) Fallback: requests normal.
    try:
        session = requests.Session()
        session.headers.update(DEFAULT_HEADERS)
        session.get(BASE_URL + "/", timeout=20)
        resp = session.get(url, cookies=cookies, timeout=20)
        resp.raise_for_status()
        return resp.text
    except Exception as exc:  # pragma: no cover
        last_error = exc

    raise RuntimeError(f"No se pudo descargar {url}: {last_error}")


def extract_for_lottery(soup: BeautifulSoup, lottery_class: str) -> List[Dict]:
    results = []
    h3 = soup.find("h3", class_=lottery_class)
    if not h3:
        return results
    container = h3.find_parent()
    # try to find the next div with js-con
    row = container.find_next(lambda tag: tag.name == "div" and tag.get("class") and any("js-con" in c for c in tag.get("class")))
    if not row:
        return results
    legends = row.find_all("div", class_="circle-legend")
    for legend in legends:
        h4 = legend.find("h4")
        h5 = legend.find("h5")
        if not h4:
            continue
        h4_text = h4.get_text(" ", strip=True)
        m = re.search(r"\b(\d{1,3}|00)\b", h4_text)
        number = m.group(0) if m else ""
        animal = normalize_animal(h4_text.replace(number, "").strip())
        # usa el número oficial del animal (corrige '00' Ballena -> 37)
        number = ANIMAL_NUM.get(animal, number)
        time_txt = h5.get_text(strip=True) if h5 else ""
        results.append({"lottery": lottery_class, "number": number, "animal": animal, "time": time_txt})
    return results


def parse_html_for_date(html: str, lotteries: List[str]) -> List[Dict]:
    soup = BeautifulSoup(html, "html.parser")
    all_results = []
    for cl in lotteries:
        all_results.extend(extract_for_lottery(soup, cl))
    return all_results


def parse_datetime_from_row(date_s: str, time_s: str) -> datetime.datetime | None:
    if not time_s:
        return None
    time_s = time_s.strip()
    # normalize common variants
    try:
        dt = datetime.datetime.strptime(f"{date_s} {time_s}", "%Y-%m-%d %I:%M %p")
        return dt
    except Exception:
        # try without AM/PM (24h)
        try:
            dt = datetime.datetime.strptime(f"{date_s} {time_s}", "%Y-%m-%d %H:%M")
            return dt
        except Exception:
            return None


def read_last_datetime(csv_file: str) -> datetime.datetime | None:
    if not os.path.exists(csv_file):
        return None
    last_dt = None
    with open(csv_file, newline='', encoding='utf-8') as f:
        r = csv.DictReader(f)
        for row in r:
            date_s = row.get("date", "").strip()
            time_s = row.get("time", "").strip()
            dt = parse_datetime_from_row(date_s, time_s)
            if dt:
                if last_dt is None or dt > last_dt:
                    last_dt = dt
    return last_dt


def append_rows(csv_file: str, rows: List[Dict]):
    """Merge `rows` with existing CSV and write sorted by datetime then lottery.

    This rewrites the CSV to keep full chronological order across lotteries.
    """
    if not rows:
        return
    fieldnames = ["date", "lottery", "number", "animal", "time"]

    # load existing rows
    existing: List[Dict] = []
    if os.path.exists(csv_file):
        with open(csv_file, newline='', encoding='utf-8') as f:
            r = csv.DictReader(f)
            for row in r:
                existing.append(row)

    combined = existing + rows

    def row_datetime(r: Dict) -> datetime.datetime:
        dt = parse_datetime_from_row(r.get("date", ""), r.get("time", ""))
        if dt:
            return dt
        # fallback: parse date only, put at midnight
        try:
            d = datetime.datetime.strptime(r.get("date", ""), "%Y-%m-%d").date()
            return datetime.datetime.combine(d, datetime.time.min)
        except Exception:
            return datetime.datetime.min

    # sort by lottery name, then by datetime (earliest first)
    combined.sort(key=lambda r: (r.get("lottery", ""), row_datetime(r)))

    # write back
    with open(csv_file, "w", newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in combined:
            w.writerow(r)


def daterange(start_date: datetime.date, end_date: datetime.date):
    for n in range(int((end_date - start_date).days) + 1):
        yield start_date + datetime.timedelta(n)


def main():
    import argparse

    p = argparse.ArgumentParser(description="Incremental updater for resultados.csv")
    p.add_argument("--file", default="resultados.csv", help="CSV file to append to")
    args = p.parse_args()

    csv_file = args.file
    last_dt = read_last_datetime(csv_file)
    if last_dt is None:
        print(f"{csv_file} not found or has no valid datetime rows. Run scrape_loterias.py first.")
        sys.exit(1)

    now = datetime.datetime.now() + datetime.timedelta(hours=1)
    print(f"Last recorded draw: {last_dt}. Fetching new results up to {now}...")

    lotteries = ["lottoactivo", "lagranjita"]
    new_rows: List[Dict] = []

    start_date = last_dt.date()
    end_date = now.date()  # include today
    for d in daterange(start_date, end_date):
        ds = d.strftime("%Y-%m-%d")
        try:
            html = fetch_date_html(ds)
        except Exception as e:
            print(f"Failed to fetch {ds}: {e}", file=sys.stderr)
            continue
        parsed = parse_html_for_date(html, lotteries)
        for item in parsed:
            item_time = item.get("time", "").strip()
            dt = parse_datetime_from_row(ds, item_time)
            if dt and last_dt < dt <= now:
                new_rows.append({"date": ds, "lottery": item["lottery"], "number": item["number"], "animal": item["animal"], "time": item_time})
        time.sleep(0.3)

    if not new_rows:
        print("No new results to append.")
        return

    append_rows(csv_file, new_rows)
    print(f"Appended {len(new_rows)} new rows to {csv_file}.")


if __name__ == "__main__":
    main()
