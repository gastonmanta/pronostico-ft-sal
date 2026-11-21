"""Detección del FT casi en tiempo real.

1. Pide a AppEEARS los últimos N días de MYD09GA (Aqua) y MOD09GA (Terra).
2. Corre la detección del FT en cada día y cada satélite.
3. Si hay una detección válida ('clear') más nueva que la publicada, rehace la figura.
4. Escribe en --out: ft_ultima_deteccion.png, ultima.json, historial.csv, index.html

Uso:
  python ft_realtime/run_ft_realtime.py                  # baja de AppEEARS (necesita EARTHDATA_USER/PASS)
  python ft_realtime/run_ft_realtime.py --nc-dir carpeta # usa .nc ya bajados (prueba local)
"""
import argparse
import json
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ft_detect as fd  # noqa: E402

CFG = json.loads((HERE / 'config_ft.json').read_text())


def sat_of(path):
    n = Path(path).name.upper()
    if n.startswith('MYD'):
        return 'Aqua'
    if n.startswith('MOD'):
        return 'Terra'
    return n.split('.')[0]


def process(nc_files):
    rows, best, cache = [], None, {}
    for f in nc_files:
        sat = sat_of(f)
        ds, lon2d, lat2d, dates, band = fd.read_appeears_nc(f)
        key = lon2d.shape
        if key not in cache:
            lm = fd.land_mask_for_grid(lon2d, lat2d, HERE / 'land_mask_modis.mat')
            cache[key] = fd.Geometry(lon2d, lat2d, lm)
        g = cache[key]
        for i, d in enumerate(dates):
            ds_ = d.strftime('%Y-%m-%d')
            try:
                res = fd.detect_day(band(CFG['band_red'], i), band(CFG['band_nir'], i), g)
            except Exception as e:
                print(f'  {sat} {ds_}: error {e}')
                continue
            print(f"  {sat} {ds_}: {res['quality']:6s} sub={res['frac_sub']:.2f} "
                  f"dist={res['dist']:.1f} km area={res['area']:.0f} km²")
            rows.append(dict(Date=ds_, Sat=sat, Status=res['quality'],
                             Frac_sub=round(res['frac_sub'], 3), Area_km2=round(res['area'], 1),
                             Dist_km=round(res['dist'], 2), Lon_cut=round(res['lon_cut'], 4)))
            if res['quality'] != 'clear':
                continue
            cand = (d, res['frac_sub'], sat, res, g)
            if best is None or (cand[0], cand[1]) > (best[0], best[1]):
                best = cand
        ds.close()
    return rows, best


def write_html(out, last, checked_utc):
    if last:
        dias = (date.today() - date.fromisoformat(last['date'])).days
        resumen = (f"<b>{last['date']}</b> (MODIS {last['sat']}) · distancia del FT "
                   f"<b>{last['dist_km']:.0f} km</b> · área estuarina "
                   f"<b>{last['area_km2']:,.0f} km²</b>".replace(',', '.'))
        hace = f"hace {dias} día{'s' if dias != 1 else ''}"
        img = f"<img src='ft_ultima_deteccion.png?v={last['date']}' alt='Última detección del FT'>"
    else:
        resumen, hace, img = 'Todavía no hay una detección válida.', '', ''
    html = f"""<!DOCTYPE html>
<html lang="es"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="refresh" content="3600">
<title>Frente de turbidez del Río de la Plata – última detección</title>
<style>
 body{{font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;margin:0;padding:12px;color:#222;background:#fff}}
 img{{width:100%;height:auto;border:1px solid #ddd}}
 p{{margin:6px 0;font-size:15px}} .s{{color:#666;font-size:12.5px}}
</style></head><body>
<p>Última detección válida: {resumen} <span class="s">({hace})</span></p>
{img}
<p class="s">Detección automática diaria con MODIS Aqua/Terra (bandas 1 y 2, 500 m) bajados de NASA AppEEARS,
algoritmo de Manta et al. (2026, JGR Oceans, en revisión). Sólo se muestran días sin nubes sobre el frente.
Última revisión: {checked_utc} UTC.</p>
</body></html>"""
    (out / 'index.html').write_text(html, encoding='utf-8')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--days', type=int, default=CFG['days_back'])
    ap.add_argument('--out', default=str(HERE / 'web'))
    ap.add_argument('--nc-dir', default=None, help='usar .nc locales en vez de AppEEARS')
    ap.add_argument('--force', action='store_true', help='rehacer la figura aunque no haya fecha nueva')
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    state_f = out / 'ultima.json'
    last = json.loads(state_f.read_text()) if state_f.exists() else None

    if a.nc_dir:
        nc_files = sorted(Path(a.nc_dir).glob('*.nc'))
    else:
        import appeears
        end = date.today() - timedelta(days=1)
        start = end - timedelta(days=a.days - 1)
        nc_files = appeears.fetch(start, end, CFG['products'],
                                  [CFG['band_red'], CFG['band_nir']],
                                  CFG['bbox'], HERE / '_nc')
    if not nc_files:
        print('No hay archivos .nc para procesar')
    rows, best = process(nc_files)

    # historial (se acumula entre corridas)
    hist_f = out / 'historial.csv'
    if rows:
        new = pd.DataFrame(rows)
        if hist_f.exists():
            new = pd.concat([pd.read_csv(hist_f), new])
        new = new.drop_duplicates(['Date', 'Sat'], keep='last').sort_values(['Date', 'Sat'])
        new.to_csv(hist_f, index=False)

    checked = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')
    if best is not None:
        d, _, sat, res, g = best
        dstr = d.strftime('%Y-%m-%d')
        if a.force or last is None or dstr >= last['date']:
            fd.make_figure(res, g, dstr, sat, out / 'ft_ultima_deteccion.png',
                           extra=f'Generado {checked} UTC · NASA AppEEARS MYD09GA/MOD09GA.061')
            last = dict(date=dstr, sat=sat, dist_km=round(res['dist'], 1),
                        area_km2=round(res['area'], 0), lon_cut=round(res['lon_cut'], 4),
                        generated_utc=checked)
            state_f.write_text(json.dumps(last, indent=2))
            np.savetxt(out / 'ft_ultima_contorno.csv', np.column_stack((res['flon'], res['flat'])),
                       delimiter=',', header='longitude,latitude', comments='', fmt='%.5f')
            print(f'Nueva detección publicada: {dstr} ({sat})')
        else:
            print(f'La detección más nueva ({dstr}) no supera la publicada ({last["date"]})')
    else:
        print('Ningún día despejado en la ventana; se mantiene la figura anterior')
    write_html(out, last, checked)


if __name__ == '__main__':
    main()
