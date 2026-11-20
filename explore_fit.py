import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from features import build
d=pd.read_csv('data_tfinal_11abr.csv'); d['date']=pd.to_datetime(d['date'],format='%d-%b-%Y'); d=d.set_index('date').asfreq('D')
base=pd.DataFrame({'u':d.u_wnd.interpolate(),'v':d.v_wnd.interpolate(),'Q':d.Caudal_Rio_de_la_Plata.interpolate()},index=d.index)
X=build(base)
targets={'TF_dist_km':d.Dist_km_merged}
for s in ['Punta_del_Tigre','Pajas_Blancas','Playa_del_Cerro','Pocitos','Carrasco','Isla_de_Flores','La_Paloma']:
    targets['S_'+s]=d['Salinidad_'+s]
def cv(model,Xm,y,yrs):
    pred=pd.Series(np.nan,index=y.index)
    for blk in np.array_split(np.unique(yrs),6):
        te=np.isin(yrs,blk); tr=~te
        if te.sum()<10 or tr.sum()<50: continue
        model.fit(Xm[tr],y[tr]); pred[te]=model.predict(Xm[te])
    ok=pred.notna(); r=np.corrcoef(pred[ok],y[ok])[0,1]; rmse=np.sqrt(((pred[ok]-y[ok])**2).mean())
    return r,rmse,y[ok].std()
for k,y in targets.items():
    y=y.dropna(); Xm=X.loc[y.index]; yrs=y.index.year.values
    lin=make_pipeline(StandardScaler(),RidgeCV(alphas=np.logspace(-2,3,20)))
    gb=HistGradientBoostingRegressor(max_iter=300,learning_rate=0.05,max_depth=3,min_samples_leaf=30,l2_regularization=1.0)
    r1=cv(lin,Xm.values,y.values if False else y,yrs); r2=cv(gb,Xm.values,y,yrs)
    print(f'{k:22s} n={len(y):5d} sd={r1[2]:6.2f} | Ridge r={r1[0]:.2f} rmse={r1[1]:6.2f} | HGB r={r2[0]:.2f} rmse={r2[1]:6.2f}')
