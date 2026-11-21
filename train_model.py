"""Entrena los modelos de pronóstico de posición del FT y salinidad costera.

Modelo base (por variable): regresión ridge sobre
  - caudal del RdlP (log Q y media de 30 d),
  - viento (u, v con memorias exponenciales de 1, 3 y 7 d; rapidez y rapidez^3),
  - interacciones viento x caudal, y ciclo estacional (seno/coseno).
Corrección con dato en vivo: el residuo del modelo en Isla de Flores al día t0
se propaga a cada variable y plazo h con beta(h), estimado con residuos de
validación cruzada por bloques de años (fuera de muestra).

Salida: model_coefs.json (lo único que necesita el script operacional) y
figuras/tablas de validación.
"""
import json, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from sklearn.linear_model import RidgeCV, Ridge
from sklearn.preprocessing import StandardScaler
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from features import build

STATIONS = {  # nombre: (lon, lat) aproximadas
 'Punta_del_Tigre': (-56.55, -34.77), 'Pajas_Blancas': (-56.37, -34.88),
 'Playa_del_Cerro': (-56.26, -34.89), 'Pocitos': (-56.15, -34.915),
 'Carrasco': (-56.05, -34.89), 'Isla_de_Flores': (-55.93, -34.945),
 'Punta_Ballena': (-55.05, -34.91), 'La_Paloma': (-54.16, -34.66)}
HMAX = 7

d = pd.read_csv('data_tfinal_11abr.csv'); d['date'] = pd.to_datetime(d['date'], format='%d-%b-%Y')
d = d.set_index('date').asfreq('D')
base = pd.DataFrame({'u': d.u_wnd.interpolate(), 'v': d.v_wnd.interpolate(),
                     'Q': d.Caudal_Rio_de_la_Plata.interpolate()}, index=d.index)
X = build(base); FEATS = list(X.columns)
T = {'TF_dist_km': d.Dist_km_merged}
for s in STATIONS: T['S_' + s] = d['Salinidad_' + s]

def ridge_cv_pred(y):
    yrs = y.index.year.values; pred = pd.Series(np.nan, index=y.index)
    if len(np.unique(yrs)) >= 6:   # validación por bloques de años
        bloques = [np.isin(yrs, b) for b in np.array_split(np.unique(yrs), 6)]
    else:                          # registro corto (Punta Ballena, 2025): 6 bloques temporales contiguos
        bloques = [np.isin(np.arange(len(y)), b) for b in np.array_split(np.arange(len(y)), 6)]
    for te in bloques:
        sc = StandardScaler().fit(X.loc[y.index[~te]])
        m = RidgeCV(alphas=np.logspace(-2, 3, 20)).fit(sc.transform(X.loc[y.index[~te]]), y[~te])
        pred[te] = m.predict(sc.transform(X.loc[y.index[te]]))
    return pred

out = {'features': FEATS, 'stations': STATIONS, 'hmax': HMAX,
       'tf_lon_fit': [-56.22077, 0.0076565],  # lon = a + b*dist_km
       'tf_p5_km': -75.1, 'tf_p95_km': 100.5,
       'Q_clim_monthly': d.groupby(d.index.month).Caudal_Rio_de_la_Plata.median().round(-2).to_dict(),
       'models': {}}
resid, preds = {}, {}
for k, y in T.items():
    y = y.dropna(); p = ridge_cv_pred(y); preds[k] = p
    resid[k] = (y - p).reindex(d.index)
    sc = StandardScaler().fit(X.loc[y.index])
    alpha = RidgeCV(alphas=np.logspace(-2, 3, 20)).fit(sc.transform(X.loc[y.index]), y).alpha_
    m = Ridge(alpha=alpha).fit(sc.transform(X.loc[y.index]), y)
    ok = p.notna()
    out['models'][k] = {'mean': sc.mean_.tolist(), 'scale': sc.scale_.tolist(),
        'coef': m.coef_.tolist(), 'intercept': float(m.intercept_), 'alpha': float(alpha),
        'n': int(len(y)), 'period': [str(y.index.min().date()), str(y.index.max().date())],
        'cv_r': float(np.corrcoef(p[ok], y[ok])[0, 1]),
        'cv_rmse': float(np.sqrt(((p[ok] - y[ok])**2).mean())), 'sd_obs': float(y.std()),
        'clip': [float(y.min()), float(y.max())]}

# propagación del residuo de Isla de Flores
R = pd.DataFrame(resid); r0 = R['S_Isla_de_Flores']; rows = []
for k in R:
    beta, sd_base, sd_corr = [], [], []
    for h in range(HMAX + 1):
        a = pd.concat([r0, R[k].shift(-h)], axis=1).dropna()
        b = float(np.polyfit(a.iloc[:, 0], a.iloc[:, 1], 1)[0]) if len(a) > 40 else 0.0
        beta.append(b); e0 = a.iloc[:, 1]; e1 = a.iloc[:, 1] - b * a.iloc[:, 0]
        sd_base.append(float(np.sqrt((e0**2).mean()))); sd_corr.append(float(np.sqrt((e1**2).mean())))
        rows.append({'variable': k, 'h': h, 'n': len(a), 'beta': b, 'rmse_base': sd_base[-1], 'rmse_con_IdF': sd_corr[-1]})
    out['models'][k].update({'beta_IdF': beta, 'rmse_h': sd_corr})
skill = pd.DataFrame(rows); skill.to_csv('validacion_skill_por_plazo.csv', index=False)
json.dump(out, open('model_coefs.json', 'w'), indent=1)

# ---- figura de validación ----
names = {'TF_dist_km': 'Frente de turbidez (km)', **{'S_' + s: s.replace('_', ' ') for s in STATIONS}}
fig, ax = plt.subplots(3, 3, figsize=(13, 10.5))
for a, k in zip(ax.flat, T):
    y = T[k].dropna(); p = preds[k]; ok = p.notna()
    a.hexbin(y[ok], p[ok], gridsize=35, mincnt=1, cmap='Blues', bins='log')
    lo, hi = np.nanpercentile(np.r_[y, p], [0.5, 99.5]); a.plot([lo, hi], [lo, hi], 'k--', lw=.8)
    m = out['models'][k]
    a.set_title(f"{names[k]}\nr={m['cv_r']:.2f}  RMSE={m['cv_rmse']:.1f}  n={m['n']}", fontsize=9.5)
    a.set_xlabel('observado'); a.set_ylabel('modelo (validación cruzada)')
fig.suptitle('Modelo base (viento + caudal), validación cruzada por bloques de años', fontsize=12)
fig.tight_layout(); fig.savefig('fig_validacion_modelo_base.png', dpi=150)

fig, ax = plt.subplots(1, 2, figsize=(12, 4.3))
for k in T:
    s = skill[skill.variable == k]
    if k == 'TF_dist_km':
        ax[1].plot(s.h, s.rmse_base, 'k--', label='sin dato IdF'); ax[1].plot(s.h, s.rmse_con_IdF, 'k-', label='con dato IdF'); continue
    ax[0].plot(s.h, 1 - (s.rmse_con_IdF / s.rmse_base)**2, marker='o', label=names[k])
ax[0].set_xlabel('plazo (días desde la última salinidad en Isla de Flores)'); ax[0].set_ylabel('fracción de varianza residual explicada')
ax[0].set_title('Ganancia por asimilar la salinidad en vivo de Isla de Flores'); ax[0].legend(fontsize=8)
ax[1].set_ylim(0, 60); ax[1].set_ylabel('RMSE (km)'); ax[1].set_title('Frente de turbidez: el dato de IdF casi no mejora (desacople)'); ax[1].set_xlabel('plazo (días)'); ax[1].legend()
fig.tight_layout(); fig.savefig('fig_ganancia_IdF.png', dpi=150)
print(skill.pivot(index='h', columns='variable', values='rmse_con_IdF').round(2))
print({k: (round(v['cv_r'], 2), round(v['cv_rmse'], 2)) for k, v in out['models'].items()})

# ---- probabilidad de desplazamiento extremo del FT ----
# P(dist > P95) y P(dist < P5) a partir del pronóstico y un error gaussiano (RMSE de validación cruzada).
from scipy.stats import norm
from sklearn.metrics import roc_auc_score, brier_score_loss
y = T['TF_dist_km'].dropna(); mu = preds['TF_dist_km'].reindex(y.index); sdv = out['models']['TF_dist_km']['cv_rmse']
out['prob'] = {}
for name, lab, p in [('P_ext', y > out['tf_p95_km'], 1 - norm.cdf((out['tf_p95_km'] - mu) / sdv)),
                     ('P_ret', y < out['tf_p5_km'], norm.cdf((out['tf_p5_km'] - mu) / sdv))]:
    bs0 = brier_score_loss(lab, np.full(len(lab), lab.mean()))
    out['prob'][name] = {'cv_auc': float(roc_auc_score(lab, p)), 'cv_brier_skill': float(1 - brier_score_loss(lab, p) / bs0)}
    print(name, out['prob'][name])
json.dump(out, open('model_coefs.json', 'w'), indent=1)
