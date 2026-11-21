"""Reconstrucción del caudal del Río de la Plata hacia atrás con series del INA y comparación con tfinal.

Esquema del manuscrito: Paraná con 20 días de demora + media de 9 días del Uruguay (día anterior).
  R1 (2006-hoy):  a*SantaFe(t-20) + b*media9[SaltoGrandeAbajo] + c   (coeficientes operacionales, ajustados 2015-2025)
  R2 (1995-2005): a*SantaFe(t-20) + b*media9[SaltoGrandeArriba] + c  (ajustados con 2000-2025)
  R3 (1985-1994): a*SantaFe(t-20) + Uruguay climatológico del día del año + c (sólo orientativo)
Entradas: caudal/series_ina_diarias.csv (bajar con caudal/bajar_series_ina.py) y data_tfinal_11abr.csv.
Salidas: caudal/caudal_rdlp_reconstruido.csv, caudal/fig_comparacion_caudal.png, caudal/comparacion_caudal.md
"""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from pathlib import Path
H = Path(__file__).resolve().parent
i = pd.read_csv(H / 'series_ina_diarias.csv', parse_dates=['fecha']).set_index('fecha').asfreq('D')
d = pd.read_csv(H.parent / 'data_tfinal_11abr.csv'); d['date'] = pd.to_datetime(d['date'], format='%d-%b-%Y')
Q = d.set_index('date').Caudal_Rio_de_la_Plata.rename('tfinal')

sf20 = i.santa_fe.interpolate(limit=5).shift(20)
m9 = lambda s: s.interpolate(limit=5).rolling(9, min_periods=3).mean().shift(1)
sga, sgr = m9(i.salto_grande_abajo), m9(i.salto_grande_arriba)

def ajustar(x1, x2, desde, hasta):
    X = pd.DataFrame({'Q': Q, 'a': x1, 'b': x2}).loc[desde:hasta].dropna()
    c, *_ = np.linalg.lstsq(np.c_[X.a, X.b, np.ones(len(X))], X.Q, rcond=None)
    return c

c1 = np.array([0.985128, 1.196008, -1272.71])                 # operacional (ina_caudal.py), 2015-2025
c2 = ajustar(sf20, sgr, '2000', '2025')
r1 = c1[0] * sf20 + c1[1] * sga + c1[2]
r2 = c2[0] * sf20 + c2[1] * sgr + c2[2]
uclim = sgr.groupby(sgr.index.dayofyear).mean()
c3 = ajustar(sf20, pd.Series(sgr.index.dayofyear.map(uclim).values, index=sgr.index), '2000', '2025')
r3 = c3[0] * sf20 + c3[1] * pd.Series(sf20.index.dayofyear.map(uclim).values, index=sf20.index) + c3[2]
rec = r1.combine_first(r2).combine_first(r3).rename('reconstruido')
fuente = pd.Series(np.where(r1.notna(), 'R1', np.where(r2.notna(), 'R2', np.where(r3.notna(), 'R3', ''))), index=rec.index)
# fórmula original del manuscrito con la serie INA "Paraná" (2015-2025)
manus = (1.367 * i.parana.interpolate(limit=5).shift(20) + m9(i.salto_grande_abajo) + 700).rename('formula_manuscrito')
out = pd.concat(sort=True, objs=[Q, rec, fuente.rename('metodo'), manus, i.ina_descarga_rdlp.rename('INA_descarga_rdlp')], axis=1)
out = out[out[['tfinal', 'reconstruido']].notna().any(axis=1)].loc['1985':]
out.round(0).to_csv(H / 'caudal_rdlp_reconstruido.csv', index_label='fecha')

def stats(a, b):
    x = pd.concat([a, b], axis=1).dropna()
    if len(x) < 30: return None
    e = x.iloc[:, 1] - x.iloc[:, 0]
    return dict(n=len(x), r=np.corrcoef(x.iloc[:, 0], x.iloc[:, 1])[0, 1], sesgo=e.mean(), rmse=np.sqrt((e ** 2).mean()),
                media_tfinal=x.iloc[:, 0].mean(), media_otra=x.iloc[:, 1].mean())
periodos = [('2000-01-01', '2005-12-31', '2000–2005 (R2: Salto Grande aporte)'),
            ('2006-01-01', '2014-12-31', '2006–2014 (R1, fuera del ajuste)'),
            ('2015-01-01', '2019-04-30', '2015–abr 2019 (R1, tfinal con Las Palmas + Guazú)'),
            ('2019-05-01', '2025-12-31', 'may 2019–2025 (R1, tfinal con Paraná)'),
            ('2000-01-01', '2025-12-31', 'Todo 2000–2025')]
filas = []
for a_, b_, nom in periodos:
    for nombre, serie in [('Reconstruido', out.reconstruido), ('Fórmula manuscrito (INA Paraná)', out.formula_manuscrito),
                          ('INA "Descarga al RdlP"', out.INA_descarga_rdlp)]:
        s = stats(out.tfinal.loc[a_:b_], serie.loc[a_:b_])
        if s: filas.append({'período': nom, 'serie': nombre, **s})
tab = pd.DataFrame(filas)
md = ['| Período | Serie | n | r | Sesgo (m³/s) | RMSE (m³/s) |', '|---|---|---|---|---|---|']
for _, r in tab.iterrows():
    f = lambda v, sg='': format(v, sg + ',.0f').replace(',', '.')
    md.append(f"| {r['período']} | {r.serie} | {r.n} | {r.r:.3f} | {f(r.sesgo, '+')} | {f(r.rmse)} |")
(H / 'comparacion_caudal.md').write_text('\n'.join(md) + '\n', encoding='utf-8')
print('\n'.join(md)); print('coef R2', c2.round(4), 'coef R3', c3.round(4))

# ---- figura ----
plt.rcParams.update({'font.size': 9.5, 'axes.spines.top': False, 'axes.spines.right': False})
fig = plt.figure(figsize=(13, 9)); gs = fig.add_gridspec(3, 3, height_ratios=[1.1, 0.8, 1], hspace=0.45, wspace=0.3)
ax = fig.add_subplot(gs[0, :]); mm = out.resample('MS').mean(numeric_only=True)
for m, col in [('R3', '#bdbdbd'), ('R2', '#6baed6'), ('R1', '#08519c')]:
    s = out.reconstruido.where(out.metodo == m).resample('MS').mean()
    ax.plot(s.index, s / 1000, color=col, lw=1.3, label={'R1': 'reconstruido: Santa Fe + Salto Grande (erogado)',
            'R2': 'reconstruido: Santa Fe + Salto Grande (aporte)', 'R3': 'reconstruido: Santa Fe + Uruguay climatológico'}[m])
ax.plot(mm.index, mm.tfinal / 1000, color='#d95f02', lw=1.1, label='tfinal (paper)')
ax.plot(mm.index, mm.INA_descarga_rdlp / 1000, color='0.4', lw=0.8, ls=':', label='INA "Descarga al RdlP"')
ax.axvline(pd.Timestamp('2019-05-01'), color='k', lw=0.6, ls='--'); ax.text(pd.Timestamp('2019-06-01'), 70, 'cambio de estimador\nen tfinal (may-2019)', fontsize=8, va='top')
ax.set_ylabel('Caudal RdlP (10³ m³/s)\nmedias mensuales'); ax.legend(fontsize=7.8, ncol=3, loc='upper left', frameon=False); ax.set_ylim(0, 75)
ax.set_title('a) Caudal del Río de la Plata: reconstrucción con series del INA y serie del paper', loc='left', fontweight='bold')
ax = fig.add_subplot(gs[1, :]); dif = (out.reconstruido - out.tfinal).resample('MS').mean()
ax.bar(dif.index, dif / 1000, width=25, color=np.where(dif > 0, '#b2182b', '#2166ac'))
ax.axhline(0, color='k', lw=0.6); ax.axvline(pd.Timestamp('2019-05-01'), color='k', lw=0.6, ls='--')
ax.set_ylabel('reconstruido − tfinal\n(10³ m³/s)'); ax.set_xlim(fig.axes[0].get_xlim())
ax.set_title('b) Diferencia mensual', loc='left', fontweight='bold')
for k, (a_, b_, nom) in enumerate([periodos[0], periodos[1], periodos[3]]):
    ax = fig.add_subplot(gs[2, k]); x = out.loc[a_:b_, ['tfinal', 'reconstruido']].dropna()
    ax.hexbin(x.tfinal / 1000, x.reconstruido / 1000, gridsize=35, mincnt=1, cmap='Blues', bins='log', extent=(0, 70, 0, 70))
    ax.plot([0, 70], [0, 70], 'k--', lw=0.7); s = stats(x.tfinal, x.reconstruido)
    ax.set_title(f"c{k+1}) {nom.split(' (')[0]}\nr={s['r']:.3f}  sesgo={s['sesgo']:+.0f}  RMSE={s['rmse']:.0f}", loc='left', fontsize=9)
    ax.set_xlabel('tfinal (10³ m³/s)'); ax.set_ylabel('reconstruido (10³ m³/s)') if k == 0 else None
fig.savefig(H / 'fig_comparacion_caudal.png', dpi=140, bbox_inches='tight')
