"""Caudal del Río de la Plata en tiempo real a partir del sistema Alerta5 del INA (Argentina).

Manuscrito (Sección 2.3): Q_RdlP = Paraná (ciudad de Paraná, 20 d antes, x1.367) + Uruguay (media 9 d de
Salto Grande + río Negro). En el INA la serie de caudal "Paraná" (26626) se discontinuó en dic-2025 y el
río Negro no está, así que se usan las series vigentes y se recalibran los coeficientes contra la serie
Caudal_Rio_de_la_Plata de tfinal_11abr.csv (2015-2025, n = 3961 días):

    Q_RdlP(t) = 0.9851 * Q_SantaFe(t - 20 d) + 1.1960 * media9d[Q_SaltoGrandeAbajo](t - 1) - 1273
    r = 0.974, RMSE = 2300 m3/s  (similar al error de la reconstrucción del paper frente a Borús, 2019)

El factor 1.196 sobre Salto Grande absorbe el aporte del río Negro y de la cuenca baja.
Con 20 días de demora, el aporte del Paraná para los próximos 7 días ya está observado; para el Uruguay
se usa persistencia del último valor.

Series INA (caudal medio diario, var_id 40): Santa Fe = 26627, Salto Grande Abajo = 26675.
API pública sin token: https://alerta.ina.gob.ar/a5/secciones
"""
import io, time, urllib.request
import numpy as np, pandas as pd

BASE = 'https://alerta.ina.gob.ar/a5'
SERIES = {'santa_fe': 26627, 'salto_grande': 26675}
SERIE_VIRTUAL_RDLP = 42277        # INA "Descarga al Río de la Plata" (sólo como referencia)
COEF = {'a_santa_fe': 0.985128, 'b_salto_grande': 1.196008, 'c': -1272.71}
LAG_PARANA = 20
VENT_URUGUAY = 9


def _get(url, timeout=90, intentos=3):
    req = urllib.request.Request(url, headers={'User-Agent': 'pronostico-ft-sal/1.0'})
    for i in range(intentos):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except Exception:
            if i == intentos - 1:
                raise
            time.sleep(10 * (i + 1))


def leer_serie(series_id, t0, t1):
    url = (f'{BASE}/getObservaciones?tipo=puntual&series_id={series_id}'
           f'&timestart={t0:%Y-%m-%d}&timeend={t1:%Y-%m-%d}&format=csvless&no_id=true')
    raw = _get(url).decode('utf-8', 'ignore').strip()
    if not raw:
        return pd.Series(dtype=float)
    df = pd.read_csv(io.StringIO(raw), header=None)
    if not np.issubdtype(df.iloc[:, -1].dtype, np.number):
        df = pd.read_csv(io.StringIO(raw))
    t = pd.to_datetime(df.iloc[:, 0], errors='coerce', utc=True).dt.tz_convert('America/Montevideo').dt.tz_localize(None)
    s = pd.Series(pd.to_numeric(df.iloc[:, -1], errors='coerce').values, index=t).dropna()
    return s.resample('D').mean()


def caudal_rdlp(hoy, ids=None, dias_hist=80, horizonte=8, verbose=True, **_):
    """Serie diaria de Q_RdlP (m3/s) desde hoy-dias_hist hasta hoy+horizonte, y metadatos."""
    ids = {**SERIES, **{k: v for k, v in (ids or {}).items() if v}}
    hoy = pd.Timestamp(hoy).normalize()
    t0 = hoy - pd.Timedelta(days=dias_hist + LAG_PARANA + VENT_URUGUAY + 5)
    sf = leer_serie(ids['santa_fe'], t0, hoy + pd.Timedelta(days=1))
    sg = leer_serie(ids['salto_grande'], t0, hoy + pd.Timedelta(days=1))
    if sf.dropna().empty or sg.dropna().empty:
        raise RuntimeError(f'faltan datos INA (Santa Fe: {len(sf.dropna())} d, Salto Grande: {len(sg.dropna())} d)')
    ult_sf, ult_sg = sf.dropna().index.max(), sg.dropna().index.max()
    idx = pd.date_range(t0, hoy + pd.Timedelta(days=horizonte), freq='D')
    sf = sf.reindex(idx).interpolate(limit=5).ffill()
    sg = sg.reindex(idx).interpolate(limit=5).ffill()          # persistencia hacia adelante
    q_par = COEF['a_santa_fe'] * sf.shift(LAG_PARANA)
    q_uru = COEF['b_salto_grande'] * sg.rolling(VENT_URUGUAY, min_periods=3).mean().shift(1)
    q = (q_par + q_uru + COEF['c']).dropna()
    q = q[q.index >= hoy - pd.Timedelta(days=dias_hist)]
    info = {'Q_hoy': float(q.get(hoy, q.iloc[-1])), 'Q_parana_hoy': float(q_par.get(hoy, np.nan)),
            'Q_uruguay_hoy': float(q_uru.get(hoy, np.nan)), 'ultimo_santa_fe': str(ult_sf.date()),
            'ultimo_salto_grande': str(ult_sg.date()), 'ids': ids}
    try:
        v = leer_serie(SERIE_VIRTUAL_RDLP, hoy - pd.Timedelta(days=10), hoy + pd.Timedelta(days=1)).dropna()
        info['Q_INA_virtual'] = float(v.iloc[-1]) if len(v) else None
    except Exception:
        info['Q_INA_virtual'] = None
    if verbose:
        print(f"[INA] Q_RdlP hoy = {info['Q_hoy']:.0f} m3/s (Paraná {info['Q_parana_hoy']:.0f} + Uruguay {info['Q_uruguay_hoy']:.0f}"
              f" {COEF['c']:+.0f}); últimos datos Santa Fe {info['ultimo_santa_fe']}, Salto Grande {info['ultimo_salto_grande']};"
              f" INA 'Descarga al RdlP': {info['Q_INA_virtual']}")
    return q, info


if __name__ == '__main__':
    q, info = caudal_rdlp(pd.Timestamp.now().normalize())
    print(q.tail(30).round(-2).to_string())
