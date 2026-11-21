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
import shutil
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
    rows, best, cache = [], {}, {}
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
            print(f"  {sat} {ds_}: {res['quality']:8s} sub={res['frac_sub']:.2f} "
                  f"dist={res['dist']:.1f} km area={res['area']:.0f} km²")
            rows.append(dict(Date=ds_, Sat=sat, Status=res['quality'],
                             Frac_sub=round(res['frac_sub'], 3), Area_km2=round(res['area'], 1),
                             Dist_km=round(res['dist'], 2), Lon_cut=round(res['lon_cut'], 4),
                             Front_cloud=round(res.get('frac_nan', np.nan), 3),
                             Front_interp=round(res.get('frac_interp', np.nan), 3)))
            if res['quality'] not in ('clear', 'clear_gf'):
                continue
            cand = (d, res['frac_sub'], sat, res, g)
            b = best.get(sat)
            if b is None or (cand[0], cand[1]) > (b[0], b[1]):
                best[sat] = cand
        ds.close()
    return rows, best


SATS = ('Aqua', 'Terra')


def _resumen(st):
    dias = (date.today() - date.fromisoformat(st['date'])).days
    return (f"<b>{st['date']}</b> · distancia del FT <b>{st['dist_km']:.0f} km</b> · "
            f"área estuarina <b>{st['area_km2']:,.0f} km²</b>".replace(',', '.')
            + f" <span class='s'>(hace {dias} día{'s' if dias != 1 else ''})</span>")


def write_html(out, states, checked_utc):
    bloques = []
    for sat in SATS:
        st = states.get(sat)
        if st:
            bloques.append(f"<h3>MODIS {sat}</h3><p>Última detección válida: {_resumen(st)}</p>"
                           f"<img src='ft_ultima_{sat.lower()}.png?v={st['date']}' "
                           f"alt='Última detección del FT con MODIS {sat}'>")
        else:
            bloques.append(f"<h3>MODIS {sat}</h3><p>Sin detección válida todavía.</p>")
    html = f"""<!DOCTYPE html>
<html lang="es"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="refresh" content="3600">
<title>Frente de turbidez del Río de la Plata – última detección</title>
<style>
 body{{font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;margin:0;padding:12px;color:#222;background:#fff}}
 img{{width:100%;height:auto;border:1px solid #ddd;margin-bottom:14px}}
 h3{{margin:10px 0 2px;font-size:17px}} p{{margin:4px 0 8px;font-size:15px}} .s{{color:#666;font-size:12.5px}}
</style></head><body>
{''.join(bloques)}
<p class="s">Detección automática diaria con MODIS Aqua y Terra (bandas 1 y 2, 500 m) bajados de NASA AppEEARS,
algoritmo de Manta et al. (2026, JGR Oceans, en revisión). Sólo se muestran días sin nubes sobre el frente.
Terra deja de tomar datos el 1/12/2026. Última revisión: {checked_utc} UTC.</p>
</body></html>"""
    (out / 'index.html').write_text(html, encoding='utf-8')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--days', type=int, default=CFG['days_back'])
    ap.add_argument('--out', default=str(HERE / 'web'))
    ap.add_argument('--nc-dir', default=None, help='usar .nc locales en vez de AppEEARS')
    ap.add_argument('--force', action='store_true', help='rehacer las figuras aunque no haya fecha nueva')
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    states = {}
    for sat in SATS:
        f = out / f'ultima_{sat.lower()}.json'
        if f.exists():
            states[sat] = json.loads(f.read_text())
    if 'Aqua' not in states and (out / 'ultima.json').exists():   # estado de la versión anterior
        old = json.loads((out / 'ultima.json').read_text())
        if old.get('sat') == 'Aqua':
            states['Aqua'] = old

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

    hist_f = out / 'historial.csv'
    if rows:
        new = pd.DataFrame(rows)
        if hist_f.exists():
            new = pd.concat([pd.read_csv(hist_f), new])
        new = new.drop_duplicates(['Date', 'Sat'], keep='last').sort_values(['Date', 'Sat'])
        new.to_csv(hist_f, index=False)

    checked = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')
    for sat in SATS:
        if sat not in best:
            print(f'{sat}: ningún día despejado en la ventana')
            continue
        d, _, _, res, g = best[sat]
        dstr = d.strftime('%Y-%m-%d')
        prev = states.get(sat)
        if not (a.force or prev is None or dstr >= prev['date'] or
                not (out / f'ft_ultima_{sat.lower()}.png').exists()):
            print(f'{sat}: {dstr} no supera la publicada ({prev["date"]})')
            continue
        fd.make_figure(res, g, dstr, sat, out / f'ft_ultima_{sat.lower()}.png',
                       extra=f'Generado {checked} UTC · NASA AppEEARS {sat} MODIS 09GA.061')
        states[sat] = dict(date=dstr, sat=sat, dist_km=round(res['dist'], 1),
                           area_km2=round(res['area'], 0), lon_cut=round(res['lon_cut'], 4),
                           quality=res['quality'], front_interp=round(res['frac_interp'], 3),
                           generated_utc=checked)
        (out / f'ultima_{sat.lower()}.json').write_text(json.dumps(states[sat], indent=2))
        np.savetxt(out / f'ft_contorno_{sat.lower()}.csv', np.column_stack((res['flon'], res['flat'])),
                   delimiter=',', header='longitude,latitude', comments='', fmt='%.5f')
        print(f'{sat}: nueva detección publicada {dstr}')

    # figura "más reciente de cualquiera de los dos" (compatibilidad con la versión anterior)
    if states:
        sat_new = max(states, key=lambda k: states[k]['date'])
        src = out / f'ft_ultima_{sat_new.lower()}.png'
        if src.exists():
            shutil.copyfile(src, out / 'ft_ultima_deteccion.png')
            (out / 'ultima.json').write_text(json.dumps(states[sat_new], indent=2))
    write_html(out, states, checked)


if __name__ == '__main__':
    main()
