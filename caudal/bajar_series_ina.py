"""Baja series diarias de caudal del INA Alerta5 desde 1970 y las imprime como CSV (para reconstruir Q_RdlP)."""
import sys, pandas as pd
sys.path.insert(0, '.')
import ina_caudal as ic
ids = {'santa_fe': 26627, 'parana': 26626, 'salto_grande_abajo': 26675, 'salto_grande_arriba': 26674,
       'guazu': 25479, 'palmas': 25480, 'ina_descarga_rdlp': 42277, 'corrientes': 26616, 'rosario': 26631}
hoy = pd.Timestamp.now().normalize(); out = {}
for k, sid in ids.items():
    parts = []
    for y0 in range(1970, hoy.year + 1, 5):
        t0, t1 = pd.Timestamp(f'{y0}-01-01'), min(pd.Timestamp(f'{y0 + 4}-12-31'), hoy + pd.Timedelta(days=1))
        try:
            s = ic.leer_serie(sid, t0, t1).dropna()
            if len(s): parts.append(s)
        except Exception as e:
            print('ERR', k, y0, e)
    out[k] = pd.concat(parts) if parts else pd.Series(dtype=float)
    print('SERIE', k, sid, len(out[k]), out[k].index.min() if len(out[k]) else None, out[k].index.max() if len(out[k]) else None)
df = pd.DataFrame(out).sort_index(); df = df[~df.index.duplicated()]
print('COLS', ','.join(df.columns))
for t, r in df.iterrows():
    print('ROW', t.date(), *[('' if pd.isna(v) else f'{v:.0f}') for v in r.values], sep=',')
