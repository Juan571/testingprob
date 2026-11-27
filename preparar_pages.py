"""Empaqueta V7 y V8 en un solo artefacto de GitHub Pages."""

import argparse
import re
from pathlib import Path
from shutil import copyfile


ROOT = Path(__file__).resolve().parent
BODY = re.compile(r"<body(?:\s[^>]*)?>", re.IGNORECASE)


def prepare(src_v7, src_v8, dest):
    """Escribe V7 en la raíz y V8 en /v8, con sus CSV separados."""
    builds = []
    for version, source, target, other_link, other_version in (
        ("V7", src_v7, dest, "v8/", "V8"),
        ("V8", src_v8, dest / "v8", "../", "V7"),
    ):
        page = (source / "dashboard.html").read_text(encoding="utf-8")
        csvs = sorted(source.glob("*.csv"))
        title = re.search(r"<title>([^<]*)</title>", page, re.IGNORECASE)
        if (not title or not re.search(rf"\b{version}\b", title[1]) or not BODY.search(page)
                or not (source / "backtest.csv").is_file()):
            raise ValueError(f"Dashboard {version} incompleto en {source}.")
        nav = (
            '\n<nav aria-label="Versiones del predictor" style="max-width:1200px;margin:1rem auto;'
            f'padding:0 1rem;color:inherit"><strong>Versión {version}</strong>'
            f' · <a href="{other_link}" style="color:inherit">Ver predictor {other_version}</a></nav>\n'
        )
        page = BODY.sub(lambda match: match[0] + nav, page, count=1)
        builds.append((target, page, csvs))

    for target, page, csvs in builds:
        target.mkdir(parents=True, exist_ok=True)
        (target / "index.html").write_text(page, encoding="utf-8")
        for source_csv in csvs:
            copyfile(source_csv, target / source_csv.name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v7", type=Path, default=ROOT / "v7_output")
    parser.add_argument("--v8", type=Path, default=ROOT / "v8_output")
    parser.add_argument("--output", type=Path, default=ROOT / "_site")
    args = parser.parse_args()
    prepare(args.v7, args.v8, args.output)
