# Pronóstico FT y Sal

Pronóstico experimental (no va en el paper) de la posición del frente de turbidez (FT) del Río de la Plata y de la salinidad en 7 puntos de la costa uruguaya, a 0–7 días.

## Cómo funciona

1. **Modelo base** (`train_model.py`). Para cada variable (FT `Dist_km_merged` y salinidad en Punta del Tigre, Pajas Blancas, Playa del Cerro, Pocitos, Carrasco, Isla de Flores y La Paloma) se ajusta una regresión ridge con:
   - caudal del RdlP (log Q y su media de 30 días),
   - viento u, v con memoria exponencial (vida media de 1, 3 y 7 días), rapidez y rapidez³,
   - interacciones viento × caudal,
   - ciclo estacional (seno y coseno del día del año).
   Se valida por bloques de años (6 bloques; cada bloque se predice con un modelo entrenado sin esos años).
2. **Corrección con el dato en vivo**. El error del modelo en Isla de Flores el día t0 (salinidad observada menos modelada) se propaga a cada estación y plazo h con un coeficiente beta(h), estimado con los residuos de la validación cruzada.
3. **Probabilidad de extremos del FT**. P(ext.) y P(retr.) son las probabilidades de superar P95 (+100 km) o quedar por debajo de P5 (−75 km), a partir del pronóstico y su RMSE.

## Desempeño (validación cruzada por años, 2000–2025)

Modelo base (viento + caudal), todos los años:

| Variable | n | r | RMSE |
|---|---|---|---|
| FT (km) | 3343 | 0.62 | 42 km |
| Punta del Tigre | 2245 | 0.55 | 3.4 |
| Pajas Blancas | 1620 | 0.59 | 5.4 |
| Playa del Cerro | 1506 | 0.59 | 6.5 |
| Pocitos | 1535 | 0.59 | 6.8 |
| Carrasco | 1487 | 0.61 | 6.9 |
| Isla de Flores | 2481 | 0.64 | 5.5 |
| La Paloma | 7066 | 0.72 | 3.1 |

Ganancia al agregar la salinidad de Isla de Flores del día t0 (RMSE en los días con dato simultáneo, 2018–2025):

| Variable | sin dato IdF | con dato IdF, h = 0 | h = 1 d | h = 3 d |
|---|---|---|---|---|
| FT (km) | 44.8 | 44.4 | 44.4 | 44.5 |
| Punta del Tigre | 3.8 | 3.6 | 3.6 | 3.7 |
| Pajas Blancas | 5.7 | 4.8 | 5.1 | 5.4 |
| Playa del Cerro | 6.4 | 4.9 | 5.4 | 6.1 |
| Pocitos | 6.6 | 4.0 | 5.0 | 6.2 |
| Carrasco | 6.4 | 3.5 | 4.8 | 6.1 |
| Isla de Flores | 5.5 | 0 | 3.2 | 4.8 |
| La Paloma | 2.9 | 2.8 | 2.8 | 2.8 |

Detalle por plazo en `validacion_skill_por_plazo.csv` y `fig_ganancia_IdF.png`.

Probabilidad de extremos: AUC 0.78 (extensión) y 0.89 (retracción).

**Mensaje para la presentación.** El dato de Isla de Flores mejora mucho la salinidad de Montevideo en los primeros 1–2 días (explica el 40–70 % de la varianza residual en Playa del Cerro, Pocitos y Carrasco el mismo día), pero casi no mejora la posición del FT (correlación de residuos −0.14). Es otra forma de ver el desacople turbidez–salinidad del paper. Las retracciones (período seco, viento débil) son más predecibles que las extensiones. Las extensiones extremas quedan subestimadas, lo que es coherente con que la mitad no tenga una señal de salinidad.

## Archivos

- `forecast_ft_sal.py`: script operacional. Sólo usa numpy, pandas y matplotlib.
- `features.py`: predictores (compartidos entre el entrenamiento y la operación).
- `model_coefs.json`: coeficientes de todos los modelos.
- `ina_caudal.py`: caudal del RdlP en vivo desde el INA (Alerta5), con el método del manuscrito.
- `config.json`: fuente del caudal (`"ina"` por defecto, `"clim"` = mediana mensual 2000–2025, o un número en m³/s) y los `series_id` del INA.
- `train_model.py`: reentrena y regenera `model_coefs.json` y las figuras de validación (necesita scikit-learn).
- `data_tfinal_11abr.csv`: base de datos (sólo para entrenar y para `--replay`).
- `.github/workflows/pronostico.yml`: corre el pronóstico 4 veces por día en GitHub y lo publica en GitHub Pages.

## Uso

```bash
pip install numpy pandas matplotlib
python forecast_ft_sal.py                       # operacional: descarga EXO Isla de Flores + GFS
python forecast_ft_sal.py --q 35000             # con caudal fijado (m3/s)
python forecast_ft_sal.py --replay 2022-01-26   # re-pronóstico histórico (útil para la charla)
```

Salidas en `salida/`: `pronostico_ft_sal.png`, `pronostico_ft_sal.csv` e `index.html` (figura + tabla).

Fuentes en vivo:
- Salinidad: `https://exoflores.fisica.edu.uy/ultimos_4_dias.csv` (columna `Sal (psu)`, cada 10 min). Se usa la media del último día con datos.
- Caudal: INA Alerta5 (`alerta.ina.gob.ar/a5`, API pública sin token). Se usa el esquema del manuscrito (Paraná con 20 días de demora + media de 9 días del Uruguay), pero con las series que hoy publica el INA: la serie de caudal "Paraná" (26626) se discontinuó en diciembre de 2025 y el río Negro no está. Por eso los coeficientes se recalibraron contra `Caudal_Rio_de_la_Plata` de tfinal (2015–2025, 3961 días): Q_RdlP = 0.985 × Q(Santa Fe, 20 días antes) + 1.196 × media de 9 días de Q(Salto Grande Abajo) − 1273 (r = 0.974, RMSE = 2300 m³/s). Como el Paraná entra con 20 días de demora, su aporte para los próximos 7 días ya está observado; para el Uruguay se usa persistencia. Si el INA falla, se usa la climatología del mes y la figura lo avisa. Como referencia la figura muestra también la serie del INA "Descarga al Río de la Plata" (42277), que en 2015–2025 subestima la serie del paper en unos 5000 m³/s.
- Viento: GFS (NOAA) a través de la API de Open-Meteo (`api.open-meteo.com/v1/gfs`, gratuita y sin clave), promediado en 15 puntos del dominio MODIS del paper (34–36°S, 58–54°W), con la misma media diaria que el ERA5 de entrenamiento (24 h previas a las 13 h).

## Publicarlo en el Google Site

Google Sites no tiene API para actualizar imágenes, así que la figura tiene que vivir en una URL fija que se actualiza sola, y el Site la embebe.

1. Crear un repositorio en GitHub (por ejemplo `pronostico-ft-sal`) y subir esta carpeta. Si no querés publicar la base de datos, agregá `data_tfinal_11abr.csv` a `.gitignore` (el pronóstico operacional no la necesita).
2. En el repositorio: Settings → Pages → Source: **GitHub Actions**.
3. Actions → `pronostico-ft-sal` → **Run workflow** (la primera vez). Después corre solo a las 4:17, 10:17, 16:17 y 22:17 de Uruguay.
4. La página queda en `https://<usuario>.github.io/pronostico-ft-sal/` y la figura en `.../pronostico_ft_sal.png`.
5. En Google Sites: Insertar → **Incorporar** → **Por URL** → pegar `https://<usuario>.github.io/pronostico-ft-sal/` → Insertar → Publicar.

Alternativa sin GitHub: correr el script con el Programador de tareas (Windows) o cron (Mac/Linux) en una computadora del instituto y publicar `salida/` en cualquier servidor web (por ejemplo, el mismo de exoflores.fisica.edu.uy).

## Limitaciones

- Con la fórmula original (1.367 × serie INA "Paraná" + Salto Grande + 700 de río Negro) el caudal da unos 3400 m³/s más alto que la serie del paper en 2015–2025; conviene revisar qué serie de Paraná se usó en el manuscrito.
- El viento de entrenamiento es ERA5 y el operacional es GFS; puede haber sesgos sistemáticos entre ambos.
- La salinidad de las playas de Montevideo es superficial y discreta; Punta del Tigre y La Paloma tienen otra frecuencia de muestreo.
- Es un modelo estadístico lineal: suaviza los extremos y no resuelve la resuspensión.

## Detección del FT en tiempo real (`ft_realtime/`)

Todos los días (`.github/workflows/ft-realtime.yml`, 11:43 hora de Uruguay) se piden a NASA AppEEARS los últimos 10 días de MYD09GA.061 (Aqua) y MOD09GA.061 (Terra), bandas 1 y 2 a 500 m, en el mismo recorte que los .nc del paper. Después se corre la detección de `deteccion_ft_20260324.py` (`ft_detect.py`, con los mismos parámetros) y, si hay un día despejado más nuevo, se rehace la figura.

- Página para embeber en Google Sites: `https://gastonmanta.github.io/pronostico-ft-sal/ft/`
- Imágenes directas: `.../ft/ft_ultima_aqua.png`, `.../ft/ft_ultima_terra.png` (última detección válida de cada satélite) y `.../ft/ft_ultima_deteccion.png` (la más reciente de las dos).
- Para mirar más atrás: Actions → ft-realtime → Run workflow → "Días hacia atrás" (p. ej. 30).
- `ft_realtime/web/historial.csv`: todas las detecciones (claras y nubladas) que se van acumulando.
- `ft_realtime/web/ultima.json` y `ft_ultima_contorno.csv`: datos y contorno de la última detección válida.

Necesita dos Secrets del repo (Settings → Secrets and variables → Actions): `EARTHDATA_USER` y `EARTHDATA_PASS` (usuario de urs.earthdata.nasa.gov).

Diferencias con el script del paper: la máscara de tierra sale de `global-land-mask` sobre la grilla del .nc. Si copiás `land_mask_modis.mat` a `ft_realtime/` y la grilla coincide, se usa esa. El área se calcula sumando píxeles en vez de con polígonos, y la figura no usa los shapefiles de GADM.

Prueba local con .nc ya bajados: `python ft_realtime/run_ft_realtime.py --nc-dir carpeta_con_nc --out /tmp/ft`.

Terra deja de tomar datos el 1/12/2026. Desde esa fecha la detección sigue sólo con Aqua, que también está por terminar. La continuidad natural es VIIRS (VNP09GA / VJ109GA, bandas I1–I2), que todavía no está implementada.
