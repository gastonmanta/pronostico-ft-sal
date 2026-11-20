"""Caudal del Río de la Plata en tiempo real a partir del sistema Alerta5 del INA (Argentina).

Método del manuscrito (Sección 2.3):
    Q_RdlP(t) = Q_Paraná(t) + Q_Uruguay(t)
    Q_Paraná(t)  = 1.367 * Q_PP(t - 20 d)              # PP = estación ciudad de Paraná (desde abr-2019)
    Q_Uruguay(t) = media de los 9 días previos de [Q_SaltoGrande + Q_Negro]

Como Q_Paraná usa el caudal en Paraná con 20 días de demora, los próximos 20 días de
Q_Paraná ya están observados: el pronóstico a 7 días no necesita pronóstico de caudal
para el Paraná. Para el Uruguay se usa persistencia del último valor.

API pública, sin token (https://alerta.ina.gob.ar/a5/secciones):
    series:  {BASE}/obs/puntual/series?format=geojson&var_id=40
    datos:   {BASE}/getObservaciones?tipo=puntual&series_id=ID&timestart=YYYY-MM-DD&timeend=YYYY-MM-DD&format=csvless&no_id=true
var_id 40 = caudal medio diario, 4 = caudal (instantáneo).

Los series_id se pueden fijar en config.json ("ina_series"). Si no, se buscan por nombre de
estación. El script imprime qué series usó para poder fijarlas después.
"""
import io, json, re, unicodedata, urllib.request
import numpy as np, pandas as pd

BASE = 'https://alerta.ina.gob.ar/a5'
COEF_PARANA = 1.367
LAG_PARANA = 20
VENT_URUGUAY = 9
SERIE_VIRTUAL_RDLP = 42277   # estación virtual INA "Descarga al Río de la Plata" (control cruzado)

# patrones de nombre de estación (sin tildes, minúsculas) para la búsqueda automática
PATRONES = {
    'parana': [r'^parana$', r'^parana\b(?!.*(guazu|palmas|de las palmas|bravo|mini))', r'puerto parana'],
    'salto_grande': [r'salto grande.*(efluente|erogad|abajo|aguas abajo)', r'^salto grande$', r'salto grande'],
    'negro': [r'(rio )?negro', r'palmar', r'paso de los toros', r'baygorria'],
}


def _get(url, timeout=60):
    req = urllib.request.Request(url, headers={'User-Agent': 'pronostico-ft-sal/1.0'})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def _norm(s):
    s = unicodedata.normalize('NFKD', str(s)).encode('ascii', 'ignore').decode().lower()
    return re.sub(r'\s+', ' ', s).strip()


def listar_series(var_ids=(40, 4)):
    """Devuelve DataFrame con series_id, var_id, estación y río."""
    rows = []
    for vid in var_ids:
        try:
            js = json.loads(_get(f'{BASE}/obs/puntual/series?format=geojson&var_id={vid}'))
        except Exception as e:  # noqa
            print(f'[INA] no pude listar var_id={vid}: {e}'); continue
        for f in js.get('features', []):
            p = f.get('properties', {})
            est = p.get('estacion') if isinstance(p.get('estacion'), dict) else {}
            nombre = p.get('estacion_nombre') or est.get('nombre') or p.get('nombre') or ''
            rio = p.get('rio') or est.get('rio') or ''
            sid = p.get('series_id') or p.get('id') or f.get('id')
            rows.append({'series_id': sid, 'var_id': vid, 'nombre': nombre, 'rio': rio,
                         'nombre_n': _norm(nombre), 'rio_n': _norm(rio)})
    return pd.DataFrame(rows)


def buscar(series, clave):
    rio = {'parana': 'parana', 'salto_grande': 'uruguay', 'negro': 'negro'}[clave]
    cand = series[series.rio_n.str.contains(rio, na=False) | (clave == 'salto_grande')]
    for pat in PATRONES[clave]:
        m = cand[cand.nombre_n.str.contains(pat, regex=True, na=False)]
        if len(m):
            m = m.sort_values('var_id', key=lambda s: s.map({40: 0, 4: 1}))  # preferir caudal medio diario
            return int(m.iloc[0].series_id), m.iloc[0].nombre
    return None, None


def leer_serie(series_id, t0, t1):
    url = (f'{BASE}/getObservaciones?tipo=puntual&series_id={series_id}'
           f'&timestart={t0:%Y-%m-%d}&timeend={t1:%Y-%m-%d}&format=csvless&no_id=true')
    raw = _get(url).decode('utf-8', 'ignore').strip()
    if not raw:
        return pd.Series(dtype=float)
    df = pd.read_csv(io.StringIO(raw), header=None)
    if not np.issubdtype(df.iloc[:, -1].dtype, np.number):  # tenía encabezado
        df = pd.read_csv(io.StringIO(raw))
    t = pd.to_datetime(df.iloc[:, 0], errors='coerce', utc=True).dt.tz_convert('America/Montevideo').dt.tz_localize(None)
    s = pd.Series(pd.to_numeric(df.iloc[:, -1], errors='coerce').values, index=t).dropna()
    return s.resample('D').mean()


def caudal_rdlp(hoy, ids=None, q_negro_defecto=700.0, dias_hist=80, horizonte=8, verbose=True):
    """Serie diaria de Q_RdlP desde hoy-dias_hist hasta hoy+horizonte (m3/s) y metadatos."""
    ids = dict(ids or {})
    hoy = pd.Timestamp(hoy).normalize()
    t0, t1 = hoy - pd.Timedelta(days=dias_hist + LAG_PARANA + VENT_URUGUAY), hoy
    info = {}
    if not all(ids.get(k) for k in ('parana', 'salto_grande')):
        series = listar_series()
        for k in ('parana', 'salto_grande', 'negro'):
            if not ids.get(k) and len(series):
                ids[k], nom = buscar(series, k)
                info[k + '_nombre'] = nom
    if verbose:
        print('[INA] series usadas:', ids, info)

    pp = leer_serie(ids['parana'], t0, t1) if ids.get('parana') else pd.Series(dtype=float)
    sg = leer_serie(ids['salto_grande'], t0, t1) if ids.get('salto_grande') else pd.Series(dtype=float)
    ng = leer_serie(ids['negro'], t0, t1) if ids.get('negro') else pd.Series(dtype=float)
    if pp.empty or sg.empty:
        raise RuntimeError(f'faltan datos INA (Paraná: {len(pp)} d, Salto Grande: {len(sg)} d)')

    ultimo_pp = pp.dropna().index.max()
    idx = pd.date_range(t0, hoy + pd.Timedelta(days=horizonte), freq='D')
    pp = pp.reindex(idx).interpolate(limit=5).ffill()
    q_par = COEF_PARANA * pp.shift(LAG_PARANA)                     # Paraná que entra al RdlP
    sg = sg.reindex(idx).interpolate(limit=5).ffill()                # persistencia hacia adelante
    ng = ng.reindex(idx).interpolate(limit=5).ffill() if not ng.empty else pd.Series(q_negro_defecto, index=idx)
    q_uru = (sg + ng).rolling(VENT_URUGUAY, min_periods=3).mean().shift(1)   # media de los 9 días previos
    q = (q_par + q_uru).dropna()
    q = q[q.index >= hoy - pd.Timedelta(days=dias_hist)]
    info.update({'ids': ids, 'Q_hoy': float(q.get(hoy, np.nan)), 'Q_parana_hoy': float(q_par.get(hoy, np.nan)),
                 'Q_uruguay_hoy': float(q_uru.get(hoy, np.nan)), 'negro_observado': not ng.empty and ids.get('negro') is not None,
                 'ultimo_dato_parana': str(ultimo_pp.date())})
    try:  # control cruzado con la estación virtual del INA
        v = leer_serie(SERIE_VIRTUAL_RDLP, hoy - pd.Timedelta(days=10), hoy + pd.Timedelta(days=10))
        info['Q_INA_virtual'] = float(v.dropna().iloc[-1]) if len(v.dropna()) else None
    except Exception:
        info['Q_INA_virtual'] = None
    if verbose:
        print(f"[INA] Q_RdlP hoy = {info['Q_hoy']:.0f} m3/s (Paraná {info['Q_parana_hoy']:.0f} + Uruguay {info['Q_uruguay_hoy']:.0f});"
              f" estación virtual INA: {info['Q_INA_virtual']}")
    return q, info


if __name__ == '__main__':
    s = listar_series()
    pd.set_option('display.width', 200)
    print(s[s.nombre_n.str.contains('parana|salto|negro|palmar|guazu|palmas', regex=True)][['series_id', 'var_id', 'nombre', 'rio']].to_string())
    q, info = caudal_rdlp(pd.Timestamp.now().normalize())
    print(q.tail(30).round(0).to_string())
