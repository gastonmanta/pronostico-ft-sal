import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from sklearn.linear_model import RidgeCV, LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from features import build
d=pd.read_csv('data_tfinal_11abr.csv'); d['date']=pd.to_datetime(d['date'],format='%d-%b-%Y'); d=d.set_index('date').asfreq('D')
base=pd.DataFrame({'u':d.u_wnd.interpolate(),'v':d.v_wnd.interpolate(),'Q':d.Caudal_Rio_de_la_Plata.interpolate()},index=d.index)
X=build(base)
T={'TF_dist_km':d.Dist_km_merged}
for s in ['Punta_del_Tigre','Pajas_Blancas','Playa_del_Cerro','Pocitos','Carrasco','Isla_de_Flores','La_Paloma']: T['S_'+s]=d['Salinidad_'+s]
res={}
for k,y in T.items():
    y=y.dropna(); yrs=y.index.year.values; pred=pd.Series(np.nan,index=y.index)
    for blk in np.array_split(np.unique(yrs),6):
        te=np.isin(yrs,blk)
        m=make_pipeline(StandardScaler(),RidgeCV(alphas=np.logspace(-2,3,20))).fit(X.loc[y.index[~te]],y[~te]); pred[te]=m.predict(X.loc[y.index[te]])
    res[k]=(y-pred).reindex(d.index)
R=pd.DataFrame(res); R.to_csv('cv_residuals.csv')
r0=R['S_Isla_de_Flores']
print('lag  '+' '.join(f'{k[:10]:>11s}' for k in R))
for h in range(0,8):
    row=[]
    for k in R:
        a=pd.concat([r0, R[k].shift(-h)],axis=1).dropna()
        if len(a)<40: row.append('   n/a     '); continue
        b=np.polyfit(a.iloc[:,0],a.iloc[:,1],1)[0]; c=a.corr().iloc[0,1]
        row.append(f'{c:5.2f}/{b:5.2f}')
    print(f'{h:3d}  '+' '.join(f'{x:>11s}' for x in row))
print('n overlap h=0:', {k:int(pd.concat([r0,R[k]],axis=1).dropna().shape[0]) for k in R})
