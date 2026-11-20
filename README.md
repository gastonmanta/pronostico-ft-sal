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
- Caudal: INA Alerta5 (`alerta.ina.gob.ar/a5`, API pública sin token). Igual que en el manuscrito: Q_RdlP = 1.367 × Q(Paraná, ciudad de Paraná, 20 días antes) + media de los 9 días previos de Q(Salto Grande) + Q(río Negro). Como el Paraná entra con 20 días de demora, el caudal del Paraná de los próximos 7 días ya está observado; para el Uruguay se usa persistencia. Si el INA falla, usa la climatología del mes y lo avisa en la figura. Como control, la figura muestra también la estación virtual del INA "Descarga al Río de la Plata" (serie 42277). La primera vez conviene correr `python ina_caudal.py`, que lista las series del INA y las que eligió, y fijar los `series_id` en `config.json`.
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

- Los `series_id` del INA para Paraná, Salto Grande y río Negro no los pude verificar desde acá (el sandbox no llega al INA): el script los busca por nombre y conviene confirmarlos una vez con `python ina_caudal.py`. Si el río Negro no está en el INA se usa un valor fijo (`Q_negro_defecto_m3s`, 700 m³/s, a revisar).
- El viento de entrenamiento es ERA5 y el operacional es GFS; puede haber sesgos sistemáticos entre ambos.
- La salinidad de las playas de Montevideo es superficial y discreta; Punta del Tigre y La Paloma tienen otra frecuencia de muestreo.
- Es un modelo estadístico lineal: suaviza los extremos y no resuelve la resuspensión.
