"""Diagnóstico: descarga series diarias del INA (Santa Fe, Salto Grande, descarga virtual) 2015-hoy y las imprime."""
import pandas as pd, ina_caudal as ic
hoy = pd.Timestamp.now().normalize()
ids = {'SF': 26627, 'SG': 26675, 'PAR': 26626, 'VIRT': 42277}
out = {}
for k, sid in ids.items():
    parts = []
    for y in range(2015, hoy.year + 1):
        try:
            parts.append(ic.leer_serie(sid, pd.Timestamp(f'{y}-01-01'), min(pd.Timestamp(f'{y}-12-31'), hoy + pd.Timedelta(days=1))))
        except Exception as e:
            print('ERR', k, y, e)
    out[k] = pd.concat(parts) if parts else pd.Series(dtype=float)
df = pd.DataFrame(out)
print('CSVSTART')
for t, r in df.iterrows():
    print('ROW', t.date(), *[('' if pd.isna(v) else f'{v:.0f}') for v in r.values], sep=',')
print('CSVEND', list(df.columns), df.notna().sum().to_dict())
