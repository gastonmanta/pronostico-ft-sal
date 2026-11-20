"""Construcción de predictores comunes a entrenamiento y operación."""
import numpy as np, pandas as pd

HALFLIVES = (1, 3, 7)  # días: memoria del viento

def wind_feats(u, v):
    """u, v: Series diarias (m/s, hacia donde sopla: u>0 hacia el E). Devuelve EWMs."""
    out = {}
    for h in HALFLIVES:
        out[f'u_ewm{h}'] = u.ewm(halflife=h, min_periods=1).mean()
        out[f'v_ewm{h}'] = v.ewm(halflife=h, min_periods=1).mean()
    spd = np.hypot(u, v)
    out['spd_ewm1'] = spd.ewm(halflife=1, min_periods=1).mean()
    out['spd3_ewm1'] = (spd**3).ewm(halflife=1, min_periods=1).mean()  # proxy de mezcla/resuspensión
    return pd.DataFrame(out)

def build(df):
    """df indexado por fecha diaria con columnas u, v, Q (m3/s)."""
    X = wind_feats(df['u'], df['v'])
    logQ = np.log(df['Q'])
    X['logQ'] = logQ
    X['logQ_30'] = logQ.rolling(30, min_periods=1).mean()
    doy = df.index.dayofyear.values
    X['s1'] = np.sin(2*np.pi*doy/365.25); X['c1'] = np.cos(2*np.pi*doy/365.25)
    # interacciones viento x caudal (la respuesta al viento depende del caudal)
    for h in (1, 3):
        X[f'uQ{h}'] = X[f'u_ewm{h}'] * (logQ - 9.9)
        X[f'vQ{h}'] = X[f'v_ewm{h}'] * (logQ - 9.9)
    return X
