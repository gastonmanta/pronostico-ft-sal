"""Diagnóstico: lista series de caudal del INA relevantes y su último dato."""
import json, pandas as pd, ina_caudal as ic
hoy = pd.Timestamp.now().normalize()
for vid in (40, 4, 2):
    js = json.loads(ic._get(f'{ic.BASE}/obs/puntual/series?format=geojson&var_id={vid}'))
    if vid == 40: print('KEYS', list(js['features'][0]['properties'].keys()))
    for f in js['features']:
        p = f['properties']; s = json.dumps(p, ensure_ascii=False).lower()
        if any(k in s for k in ['paran', 'salto', 'negro', 'palmar', 'guaz', 'palmas', 'descarga', 'rosario', 'santa fe', 'timbu']):
            sid = p.get('series_id') or p.get('id') or f.get('id')
            try:
                x = ic.leer_serie(sid, hoy - pd.Timedelta(days=40), hoy + pd.Timedelta(days=1)).dropna()
                last = f'{x.index[-1].date()} {x.iloc[-1]:.0f} (n={len(x)})' if len(x) else 'SIN DATOS'
            except Exception as e:
                last = f'ERR {e}'
            if vid != 2 or 'paran' in s:
                print(vid, sid, '|', json.dumps({k: p.get(k) for k in p if k in ('estacion_nombre','nombre','rio','estacion_id','tabla','proc_id','unit_id')}, ensure_ascii=False)[:200], '|', last)
