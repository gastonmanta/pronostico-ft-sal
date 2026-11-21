"""Baja de AppEEARS días puntuales y los guarda comprimidos (int16) para depurar el algoritmo.
Uso: python ft_realtime/dump_days.py 2026-11-08,2026-11-19 ft_realtime/debug
"""
import json, sys
from datetime import date
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import appeears, ft_detect as fd  # noqa: E402

CFG = json.loads((HERE / 'config_ft.json').read_text())
days = sorted(date.fromisoformat(s.strip()) for s in sys.argv[1].split(','))
out = Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
files = appeears.fetch(days[0], days[-1], CFG['products'], [CFG['band_red'], CFG['band_nir']],
                       CFG['bbox'], HERE / '_nc', name='ft_debug_dump')
want = {d.isoformat() for d in days}
for f in files:
    sat = 'aqua' if Path(f).name.upper().startswith('MYD') else 'terra'
    ds, lon2d, lat2d, dates, band = fd.read_appeears_nc(f)
    for i, d in enumerate(dates):
        ds_ = d.strftime('%Y-%m-%d')
        if ds_ not in want:
            continue
        enc = lambda a: np.where(np.isnan(a), -32768, np.round(a * 10000)).astype(np.int16)
        np.savez_compressed(out / f'{sat}_{ds_}.npz', lon=lon2d[0], lat=lat2d[:, 0],
                            b1=enc(band(CFG['band_red'], i)), b2=enc(band(CFG['band_nir'], i)))
        print('guardado', sat, ds_)
    ds.close()
