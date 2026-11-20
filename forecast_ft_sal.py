#!/usr/bin/env python3
"""Pronóstico experimental del frente de turbidez (FT) y la salinidad costera del Río de la Plata.

Fuentes en vivo
  - Salinidad de Isla de Flores (EXO, cada 10 min): https://exoflores.fisica.edu.uy/ultimos_4_dias.csv
  - Viento GFS (NOAA) vía Open-Meteo, promediado en el dominio MODIS del paper
    (33.3-36.5°S, 58.7-53.1°W), igual que el ERA5 usado para entrenar.
  - Caudal del RdlP: INA Alerta5 (ina_caudal.py), con el método del manuscrito:
    1.367 x Paraná (ciudad de Paraná, 20 d antes) + media 9 d de (Salto Grande + río Negro).
    En config.json "Q_m3s" puede ser "ina" (defecto), "clim" (mediana mensual) o un número.

Uso
  python forecast_ft_sal.py                     # operacional (necesita internet)
  python forecast_ft_sal.py --q 32000           # fija el caudal (m3/s)
  python forecast_ft_sal.py --replay 2024-05-06 # re-pronóstico con datos históricos (ERA5 + IdF observada)

Salidas en ./salida/: pronostico_ft_sal.png, pronostico_ft_sal.csv, index.html
Sólo requiere numpy, pandas, matplotlib (sin sklearn): los coeficientes están en model_coefs.json.
"""
import argparse, io, json, sys, urllib.request, datetime as dt
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from features import build
import ina_caudal

HERE = Path(__file__).resolve().parent
EXO_URL = 'https://exoflores.fisica.edu.uy/ultimos_4_dias.csv'
LATS = [-34.0, -35.0, -36.0]
LONS = [-58.0, -57.0, -56.0, -55.0, -54.0]
TZ = 'America/Montevideo'
DIAS = ['lun', 'mar', 'mié', 'jue', 'vie', 'sáb', 'dom']
NAMES = {'Punta_del_Tigre': 'Punta del Tigre', 'Pajas_Blancas': 'Pajas Blancas',
         'Playa_del_Cerro': 'Playa del Cerro', 'Pocitos': 'Pocitos', 'Carrasco': 'Carrasco',
         'Isla_de_Flores': 'Isla de Flores', 'La_Paloma': 'La Paloma'}


def http_get(url, timeout=90, intentos=4):
    import time
    req = urllib.request.Request(url, headers={'User-Agent': 'pronostico-ft-sal/1.0'})
    for i in range(intentos):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except Exception as e:
            print(f'[http] intento {i + 1}/{intentos} falló ({url[:60]}...): {e}')
            if i == intentos - 1:
                raise
            time.sleep(15 * (i + 1))


# ------------------------------------------------------------------ datos en vivo
def get_idf_salinity(path=None):
    raw = Path(path).read_bytes() if path else http_get(EXO_URL)
    df = pd.read_csv(io.BytesIO(raw), encoding='utf-8-sig')
    tcol = [c for c in df.columns if c.lower().startswith('timestamp')][0]
    scol = [c for c in df.columns if c.lower().startswith('sal')][0]
    s = pd.Series(pd.to_numeric(df[scol], errors='coerce').values,
                  index=pd.to_datetime(df[tcol], errors='coerce')).dropna().sort_index()
    s = s[(s >= 0) & (s < 40)]
    return s  # hora local, cada 10 min


def get_gfs_wind(past_days=30, forecast_days=8, path=None):
    if path:
        js = json.loads(Path(path).read_text())
    else:
        url = ('https://api.open-meteo.com/v1/gfs?'
               f'latitude={",".join(str(a) for la in LATS for a in [la]*len(LONS))}'
               f'&longitude={",".join(str(o) for _ in LATS for o in LONS)}'
               '&hourly=wind_speed_10m,wind_direction_10m&wind_speed_unit=ms'
               f'&past_days={past_days}&forecast_days={forecast_days}&timezone={TZ}')
        try:
            js = json.loads(http_get(url))
        except Exception:  # respaldo: endpoint general de Open-Meteo con el modelo GFS
            js = json.loads(http_get(url.replace('/v1/gfs?', '/v1/forecast?models=gfs_seamless&')))
    js = js if isinstance(js, list) else [js]
    us, vs = [], []
    for p in js:
        h = pd.DataFrame(p['hourly']); t = pd.to_datetime(h['time'])
        spd, dr = h['wind_speed_10m'].astype(float), np.deg2rad(h['wind_direction_10m'].astype(float))
        us.append(pd.Series((-spd * np.sin(dr)).values, index=t))   # u>0 hacia el E
        vs.append(pd.Series((-spd * np.cos(dr)).values, index=t))   # v>0 hacia el N
    u = pd.concat(us, axis=1).mean(axis=1); v = pd.concat(vs, axis=1).mean(axis=1)
    # media diaria de las 24 h previas a las 13 h locales (pasaje de Aqua), como en el paper
    u = u.shift(11, freq='h').resample('D').mean(); v = v.shift(11, freq='h').resample('D').mean()
    run = js[0].get('generationtime_ms')
    return pd.DataFrame({'u': u, 'v': v}).dropna(), run


# ------------------------------------------------------------------ modelo
def predict_base(coefs, X):
    out = {}
    for k, m in coefs['models'].items():
        Z = (X[coefs['features']].values - np.array(m['mean'])) / np.array(m['scale'])
        y = Z @ np.array(m['coef']) + m['intercept']
        out[k] = pd.Series(y, index=X.index)
    return pd.DataFrame(out)


def run_forecast(coefs, wind, Q, idf_daily, t0):
    """wind: DataFrame diario u,v (pasado+pronóstico); Q: float o Series; idf_daily: Series diaria."""
    df = wind.copy()
    df['Q'] = Q if np.isscalar(Q) else Q.reindex(df.index).ffill().bfill()
    X = build(df)
    base = predict_base(coefs, X)
    r0 = np.nan
    if t0 in idf_daily.index and t0 in base.index:
        r0 = idf_daily[t0] - base.loc[t0, 'S_Isla_de_Flores']
    days = [d for d in base.index if d >= t0][: coefs['hmax'] + 1]
    rows = []
    for h, day in enumerate(days):
        row = {'fecha': day, 'h': h}
        for k, m in coefs['models'].items():
            b = m['beta_IdF'][h] if np.isfinite(r0) else 0.0
            val = base.loc[day, k] + b * (r0 if np.isfinite(r0) else 0)
            lo, hi = m['clip']
            if k.startswith('S_'):
                val = float(np.clip(val, max(lo, 0.1), hi))
            row[k] = val; row[k + '_base'] = base.loc[day, k]; row[k + '_err'] = m['rmse_h'][h] if np.isfinite(r0) else m['cv_rmse']
        rows.append(row)
    fc = pd.DataFrame(rows).set_index('fecha')
    fc['Q_RdlP'] = df['Q'].reindex(fc.index).values
    a, b = coefs['tf_lon_fit']; fc['TF_lon'] = a + b * fc['TF_dist_km']
    from math import erf, sqrt
    Phi = lambda z: 0.5 * (1 + erf(z / sqrt(2)))
    sd = coefs['models']['TF_dist_km']['cv_rmse']
    fc['P_ext'] = [1 - Phi((coefs['tf_p95_km'] - m) / sd) for m in fc['TF_dist_km']]
    fc['P_ret'] = [Phi((coefs['tf_p5_km'] - m) / sd) for m in fc['TF_dist_km']]
    return fc, base, r0


# ------------------------------------------------------------------ figura
def make_figure(fc, wind, idf_raw, idf_daily, coefs, q_src, r0, t0, out_png, mode_txt, obs=None):
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig = plt.figure(figsize=(14, 8.6))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 0.75], width_ratios=[1.25, 1], hspace=0.55, wspace=0.42)
    tmin = t0 - pd.Timedelta(days=5); tmax = fc.index.max() + pd.Timedelta(hours=24)

    # (a) Isla de Flores
    ax = fig.add_subplot(gs[0, 0])
    if idf_raw is not None and len(idf_raw):
        ax.plot(idf_raw.index, idf_raw.values, color='0.55', lw=0.8, label='observada (10 min)')
    idd = idf_daily[(idf_daily.index >= tmin) & (idf_daily.index <= t0)]
    ax.plot(idd.index + pd.Timedelta(hours=12), idd.values, 'ko', ms=4, label='media diaria observada')
    k = 'S_Isla_de_Flores'; x = fc.index + pd.Timedelta(hours=12)
    ax.fill_between(x, (fc[k] - fc[k + '_err']).clip(lower=0), fc[k] + fc[k + '_err'], color='#2b7bba', alpha=0.18, lw=0)
    ax.plot(x, fc[k], '-o', color='#2b7bba', ms=4, label='pronóstico (±1 RMSE)')
    ax.plot(x, fc[k + '_base'], '--', color='#2b7bba', lw=1, alpha=0.7, label='sólo viento + caudal')
    if obs is not None and k in obs:
        o = obs[k].reindex(fc.index); ax.plot(x, o, 'x', color='crimson', ms=7, mew=2, label='verificación')
    ax.axvline(t0 + pd.Timedelta(hours=12), color='k', lw=0.6, ls=':')
    ax.set_xlim(tmin, tmax); ax.set_ylim(0, 35); ax.set_ylabel('Salinidad')
    ax.set_title('a) Salinidad en Isla de Flores', loc='left', fontweight='bold')
    ax.legend(fontsize=7.5, ncol=2, loc='upper left', frameon=False)

    # (b) frente de turbidez
    ax = fig.add_subplot(gs[1, 0])
    p5, p95 = coefs['tf_p5_km'], coefs['tf_p95_km']
    ax.axhspan(p95, 260, color='#d7301f', alpha=0.08); ax.axhspan(-200, p5, color='#fdae61', alpha=0.15)
    ax.text(tmin, p95 + 4, ' extensión extrema (>P95)', fontsize=8, color='#a50f15', va='bottom')
    ax.text(tmin, p5 - 5, ' retracción extrema (<P5)', fontsize=8, color='#b35806', va='top')
    k = 'TF_dist_km'
    ax.fill_between(x, fc[k] - fc[k + '_err'], fc[k] + fc[k + '_err'], color='#8c510a', alpha=0.18, lw=0)
    ax.plot(x, fc[k], '-o', color='#8c510a', ms=4, label='pronóstico (±1 RMSE)')
    if obs is not None and k in obs:
        o = obs[k]; o = o[(o.index >= tmin) & (o.index <= tmax)]
        ax.plot(o.index + pd.Timedelta(hours=12), o.values, 'x', color='crimson', ms=7, mew=2, label='FT observado (MODIS)')
    ax.axhline(0, color='k', lw=0.6); ax.axvline(t0 + pd.Timedelta(hours=12), color='k', lw=0.6, ls=':')
    ax.set_xlim(tmin, tmax); ax.set_ylim(-170, 230)
    ax.set_ylabel('km respecto a la posición media\n(+ = hacia el E)')
    a_, b_ = coefs['tf_lon_fit']
    sec = ax.secondary_yaxis('right', functions=(lambda d: a_ + b_ * d, lambda l: (l - a_) / b_)); sec.set_ylabel('longitud (°)')
    ax.set_title('b) Posición del frente de turbidez sobre la costa norte', loc='left', fontweight='bold')
    for xi, (pe, pr) in zip(x, fc[['P_ext', 'P_ret']].values):
        ax.text(xi, 215, f'{100*pe:.0f}%', ha='center', va='top', fontsize=7.5, color='#a50f15')
        ax.text(xi, -160, f'{100*pr:.0f}%', ha='center', va='bottom', fontsize=7.5, color='#b35806')
    ax.text(t0 - pd.Timedelta(hours=6), 215, 'P(ext.)', ha='right', va='top', fontsize=7.5, color='#a50f15')
    ax.text(t0 - pd.Timedelta(hours=6), -160, 'P(retr.)', ha='right', va='bottom', fontsize=7.5, color='#b35806')
    ax.legend(fontsize=7.5, loc='upper left', frameon=False, bbox_to_anchor=(0.0, 0.66))

    # (c) viento
    ax = fig.add_subplot(gs[2, 0])
    w = wind[(wind.index >= tmin) & (wind.index <= tmax)]
    spd = np.hypot(w.u, w.v); xw = w.index + pd.Timedelta(hours=12)
    ax.bar(xw, spd, width=0.75, color=['#9ecae1' if d <= t0 else '#3182bd' for d in w.index])
    ax.quiver(xw, spd + 1.2, w.u / spd, w.v / spd, angles='uv', scale=28, width=0.004, color='k')
    ax.axvline(t0 + pd.Timedelta(hours=12), color='k', lw=0.6, ls=':')
    ax.set_xlim(tmin, tmax); ax.set_ylim(0, max(10, spd.max() + 3)); ax.set_ylabel('m/s')
    ax.set_title('c) Viento medio en el estuario (flecha: hacia donde sopla)', loc='left', fontweight='bold')
    for a in fig.axes:
        if hasattr(a, 'xaxis') and a.get_subplotspec() is not None and a.get_subplotspec().colspan.start == 0:
            a.xaxis.set_major_locator(matplotlib.dates.DayLocator(interval=2)); a.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%d/%m'))

    # (d) salinidad costera
    ax = fig.add_subplot(gs[0:2, 1])
    st = [s for s in NAMES]  # oeste -> este
    M = np.array([[fc['S_' + s].iloc[h] for h in range(len(fc))] for s in st])
    cmap = LinearSegmentedColormap.from_list('sal', ['#8c510a', '#d8b365', '#f6e8c3', '#c7eae5', '#5ab4ac', '#01665e'])
    im = ax.imshow(M, aspect='auto', cmap=cmap, vmin=0, vmax=32)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f'{M[i, j]:.0f}', ha='center', va='center', fontsize=9,
                    color='white' if (M[i, j] < 6 or M[i, j] > 26) else 'k')
    ax.set_yticks(range(len(st))); ax.set_yticklabels([NAMES[s] for s in st])
    ax.set_xticks(range(len(fc))); ax.set_xticklabels([f"{DIAS[d.dayofweek]}\n{d:%d/%m}" for d in fc.index], fontsize=8)
    ax.set_title('d) Salinidad pronosticada en la costa uruguaya\n(oeste arriba, este abajo)', loc='left', fontweight='bold')
    cb = fig.colorbar(im, ax=ax, orientation='horizontal', fraction=0.05, pad=0.12); cb.set_label('Salinidad')
    for sp in ax.spines.values(): sp.set_visible(False)

    # (e) texto
    ax = fig.add_subplot(gs[2, 1]); ax.axis('off')
    last = idf_daily[idf_daily.index <= t0]
    txt = (f'{mode_txt}\n'
           f'Día de inicio (t0): {t0:%d/%m/%Y}   ·   salinidad IdF t0: {last.iloc[-1]:.1f}\n'
           f'Corrección por dato IdF: {r0:+.1f} (obs − modelo)\n'
           f'Caudal RdlP: {q_src}\n'
           f'Modelo: regresión ridge (viento GFS + caudal + estacionalidad),\n'
           f'entrenada con 2000–2025 (MODIS, ERA5, salinidad costera);\n'
           f'el residuo de Isla de Flores se propaga a las demás estaciones.\n'
           f'P(ext.)/P(retr.): prob. de superar P95/P5 (AUC val. cruzada {coefs["prob"]["P_ext"]["cv_auc"]:.2f}/{coefs["prob"]["P_ret"]["cv_auc"]:.2f}).\n'
           f'Experimental, sin validación operacional.  Manta et al.')
    ax.text(0, 1, txt, va='top', fontsize=8.6, family='DejaVu Sans', linespacing=1.45)
    fig.suptitle('Pronóstico experimental: frente de turbidez y salinidad costera del Río de la Plata',
                 fontsize=14, fontweight='bold', x=0.06, ha='left', y=0.985)
    fig.savefig(out_png, dpi=130, bbox_inches='tight'); plt.close(fig)


def write_html(out_dir, fc, stamp):
    rows = ''.join(f'<tr><td>{DIAS[d.dayofweek]} {d:%d/%m}</td><td>{r.TF_dist_km:+.0f} km ({r.TF_lon:.2f}°)</td><td>{100*r.P_ext:.0f}% / {100*r.P_ret:.0f}%</td>'
                   + ''.join(f'<td>{r["S_" + s]:.1f}</td>' for s in NAMES) + '</tr>' for d, r in fc.iterrows())
    html = f'''<!DOCTYPE html><html lang="es"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Pronóstico FT y salinidad – RdlP</title>
<style>body{{font-family:system-ui,sans-serif;margin:0;padding:12px;color:#222}}img{{width:100%;height:auto}}
table{{border-collapse:collapse;font-size:13px;margin-top:8px}}td,th{{border-bottom:1px solid #ddd;padding:3px 8px;text-align:right}}
th:first-child,td:first-child{{text-align:left}}small{{color:#666}}</style></head><body>
<img src="pronostico_ft_sal.png?v={stamp:%Y%m%d%H}" alt="Pronóstico del frente de turbidez y salinidad costera">
<table><tr><th>Día</th><th>Frente de turbidez</th><th>P(ext.)/P(retr.)</th>{''.join(f'<th>{n}</th>' for n in NAMES.values())}</tr>{rows}</table>
<p><small>Actualizado {stamp:%d/%m/%Y %H:%M} (hora de Uruguay). Datos: EXO Isla de Flores (IF-FCien), viento GFS/NOAA vía Open-Meteo.
Pronóstico experimental.</small></p></body></html>'''
    (out_dir / 'index.html').write_text(html, encoding='utf-8')


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--q', type=float, help='caudal RdlP (m3/s); por defecto config.json')
    ap.add_argument('--replay', help='fecha t0 YYYY-MM-DD para re-pronóstico con la base histórica')
    ap.add_argument('--sal-csv', help='CSV local de exoflores (en lugar de descargar)')
    ap.add_argument('--wind-json', help='JSON local de Open-Meteo (en lugar de descargar)')
    ap.add_argument('--out', default=str(HERE / 'salida'))
    a = ap.parse_args()
    coefs = json.loads((HERE / 'model_coefs.json').read_text())
    cfg = json.loads((HERE / 'config.json').read_text()) if (HERE / 'config.json').exists() else {}
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    now = pd.Timestamp.now(tz=TZ).tz_localize(None)

    if a.replay:
        d = pd.read_csv(HERE / 'data_tfinal_11abr.csv'); d['date'] = pd.to_datetime(d['date'], format='%d-%b-%Y')
        d = d.set_index('date').asfreq('D')
        t0 = pd.Timestamp(a.replay)
        wind = pd.DataFrame({'u': d.u_wnd.interpolate(), 'v': d.v_wnd.interpolate()})
        wind = wind[(wind.index >= t0 - pd.Timedelta(days=30)) & (wind.index <= t0 + pd.Timedelta(days=coefs['hmax']))]
        Q = float(a.q) if a.q else d.Caudal_Rio_de_la_Plata.interpolate()
        q_src = (f'fijado a mano: {Q:,.0f} m³/s' if a.q else f'serie de la base (t0: {Q[t0]:,.0f} m³/s)').replace(',', '.')
        idf_daily = d.Salinidad_Isla_de_Flores.dropna(); idf_raw = None
        obs = {'S_Isla_de_Flores': d.Salinidad_Isla_de_Flores, 'TF_dist_km': d.Dist_km_merged.dropna()}
        mode_txt = 'RE-PRONÓSTICO con datos históricos (viento ERA5 "perfecto"; × = observado)'
    else:
        idf_raw = get_idf_salinity(a.sal_csv)
        idf_daily = idf_raw.resample('D').mean()
        t0 = idf_daily.index.max()          # último día con dato (puede ser parcial)
        wind, _ = get_gfs_wind(path=a.wind_json)
        qcfg = a.q if a.q else cfg.get('Q_m3s', 'ina')
        q_src = None
        if qcfg == 'ina':
            try:
                Q, qinfo = ina_caudal.caudal_rdlp(t0, ids=cfg.get('ina_series'),
                                                  q_negro_defecto=cfg.get('Q_negro_defecto_m3s', 700.0))
                q_src = f"INA (1.367·Paraná[t−20 d] + Uruguay 9 d): {qinfo['Q_hoy']:,.0f} m³/s".replace(',', '.')
                if qinfo.get('Q_INA_virtual'):
                    q_src += f"  ·  INA 'Descarga al RdlP': {qinfo['Q_INA_virtual']:,.0f}".replace(',', '.')
            except Exception as e:
                print('[INA] falló, uso climatología:', e); qcfg = 'clim'
        if qcfg == 'clim':
            Q = float(coefs['Q_clim_monthly'][str(t0.month)]); q_src = f'climatología del mes (INA no disponible): {Q:,.0f} m³/s'.replace(',', '.')
        elif q_src is None:
            Q = float(qcfg); q_src = f'fijado a mano: {Q:,.0f} m³/s'.replace(',', '.')
        obs = None
        mode_txt = f'OPERACIONAL · corrida {now:%d/%m/%Y %H:%M} · viento GFS (Open-Meteo)'

    fc, base, r0 = run_forecast(coefs, wind, Q, idf_daily, t0)
    make_figure(fc, wind, idf_raw, idf_daily, coefs, q_src, r0, t0, out / 'pronostico_ft_sal.png', mode_txt, obs)
    cols = ['h', 'Q_RdlP', 'TF_dist_km', 'TF_lon', 'P_ext', 'P_ret'] + ['S_' + s for s in NAMES]
    fc[cols].round(2).to_csv(out / 'pronostico_ft_sal.csv')
    write_html(out, fc, now)
    print(fc[cols].round(1).to_string())
    print('OK ->', out)


if __name__ == '__main__':
    main()
