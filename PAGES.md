# Publicación de V7 y V8

- `/testingprob/` sigue mostrando V7, generado por `analisis.py`.
- `/testingprob/v8/` muestra V8, generado por `analisis_v8.py`, con el Top 3 por lotería y el Top 10 consolidado en tarjetas separadas.
- El flujo de GitHub Actions actualiza `resultados.csv`, ejecuta ambas versiones y publica un único artefacto. V7 lee sólo La Granjita y Lotto Activo; V8 también usa Lotto Activo Internacional. Los CSV de V8 se publican bajo `/v8/`, sin reemplazar los de V7.

Para compilar localmente sin solicitar datos nuevos:

```sh
python -m pip install -r requirements.txt
python analisis.py
python analisis_v8.py
python preparar_pages.py
python -m http.server --directory _site
```

`analisis.py` reescribe `v7_output/`. Usa un directorio de prueba si necesitas preservar sus archivos generados.
