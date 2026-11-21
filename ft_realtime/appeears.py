"""Cliente mínimo de la API de AppEEARS (NASA LP DAAC).

Pide un recorte 'area' de MODIS (MYD09GA / MOD09GA, bandas 1 y 2) en NetCDF4
geográfico, espera a que termine y baja los .nc.

Credenciales Earthdata: variables de entorno EARTHDATA_USER y EARTHDATA_PASS
(en GitHub van como Secrets del repositorio).

Docs de la API: https://appeears.earthdatacloud.nasa.gov/api/
"""
import os
import time
from datetime import date
from pathlib import Path

import requests

API = 'https://appeears.earthdatacloud.nasa.gov/api'


def login(user=None, password=None):
    user = user or os.environ['EARTHDATA_USER']
    password = password or os.environ['EARTHDATA_PASS']
    r = requests.post(f'{API}/login', auth=(user, password), timeout=60)
    r.raise_for_status()
    return r.json()['token']


def logout(token):
    try:
        requests.post(f'{API}/logout', headers=_h(token), timeout=30)
    except Exception:
        pass


def _h(token):
    return {'Authorization': f'Bearer {token}'}


def build_task(name, start: date, end: date, products, layers, bbox):
    """bbox = (lon_min, lat_min, lon_max, lat_max)."""
    lo0, la0, lo1, la1 = bbox
    poly = [[lo0, la0], [lo1, la0], [lo1, la1], [lo0, la1], [lo0, la0]]
    return {
        'task_type': 'area',
        'task_name': name,
        'params': {
            'dates': [{'startDate': start.strftime('%m-%d-%Y'),
                       'endDate': end.strftime('%m-%d-%Y')}],
            'layers': [{'product': p, 'layer': l} for p in products for l in layers],
            'geo': {'type': 'FeatureCollection', 'features': [{
                'type': 'Feature', 'properties': {},
                'geometry': {'type': 'Polygon', 'coordinates': [poly]}}]},
            'output': {'format': {'type': 'netcdf4'}, 'projection': 'geographic'},
        },
    }


def submit(token, task):
    r = requests.post(f'{API}/task', json=task, headers=_h(token), timeout=120)
    if r.status_code >= 400:
        raise RuntimeError(f'AppEEARS rechazó la tarea ({r.status_code}): {r.text}')
    return r.json()['task_id']


def wait(token, task_id, max_min=150, every_s=60):
    t0 = time.time()
    while True:
        r = requests.get(f'{API}/task/{task_id}', headers=_h(token), timeout=60)
        r.raise_for_status()
        st = r.json().get('status')
        print(f'  AppEEARS {task_id}: {st} ({(time.time() - t0) / 60:.0f} min)', flush=True)
        if st == 'done':
            return
        if st == 'error':
            raise RuntimeError(f'La tarea {task_id} terminó con error: {r.json()}')
        if time.time() - t0 > max_min * 60:
            raise TimeoutError(f'La tarea {task_id} no terminó en {max_min} min')
        time.sleep(every_s)


def download_nc(token, task_id, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    r = requests.get(f'{API}/bundle/{task_id}', headers=_h(token), timeout=60)
    r.raise_for_status()
    files = []
    for f in r.json()['files']:
        if not f['file_name'].endswith('.nc'):
            continue
        dest = out_dir / Path(f['file_name']).name
        with requests.get(f"{API}/bundle/{task_id}/{f['file_id']}", headers=_h(token),
                          stream=True, allow_redirects=True, timeout=600) as d:
            d.raise_for_status()
            with open(dest, 'wb') as fh:
                for chunk in d.iter_content(chunk_size=1 << 20):
                    fh.write(chunk)
        print(f'  bajado {dest.name} ({dest.stat().st_size / 1e6:.1f} MB)')
        files.append(dest)
    return files


def fetch(start, end, products, layers, bbox, out_dir, name='ft_rdlp_realtime'):
    token = login()
    try:
        tid = submit(token, build_task(name, start, end, products, layers, bbox))
        print(f'Tarea AppEEARS enviada: {tid} ({start} a {end})')
        wait(token, tid)
        return download_nc(token, tid, out_dir)
    finally:
        logout(token)
