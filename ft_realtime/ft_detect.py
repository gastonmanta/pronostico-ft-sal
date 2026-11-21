"""Detección del frente de turbidez (FT) del Río de la Plata, versión operacional.

Es el algoritmo de deteccion_ft_20260324.py (manuscrito, Sección 2.2) con estos
cambios para que corra solo en GitHub:
  - lee los .nc de AppEEARS con netCDF4 (sin xarray/h5py);
  - la máscara de tierra sale de `global_land_mask` sobre la grilla del .nc
    (o de land_mask_modis.mat si se copia al repo y coincide la grilla);
  - el área estuarina se calcula sumando píxeles (en vez de polígonos shapely),
    lo que evita geopandas/shapely/pyproj; la diferencia es menor a un píxel de borde;
  - la figura no usa los shapefiles GADM (sólo la máscara de tierra).
Los umbrales y parámetros son los mismos del script original.
"""
from math import radians, cos, sin, asin, sqrt
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from scipy.ndimage import gaussian_filter1d, median_filter
from skimage import measure
from skimage.filters import threshold_otsu, sobel
from skimage.morphology import disk, closing

# --- Parámetros (idénticos a deteccion_ft_20260324.py) ---
P = dict(
    nan_threshold=0.10,
    cloud_reflect_thresh=20,
    min_object_size=4000,
    se_disk_radius=10,
    median_ksize=3,
    ancho_total_km=15.0,
    paso_km=1.0,
    lat_sub=(-36.0, -34.5),
    lon_sub=(-57.0, -55.0),
    subregion_threshold=0.80,
    dom=(-58.5, -53.0, -36.5, -33.5),   # lon_min, lon_max, lat_min, lat_max
    lon_cut=-58.4062, lat_cut=-33.9479,
    lon_ref=-56.22,
)


# ----------------------------------------------------------------------------
# utilidades
# ----------------------------------------------------------------------------
def remove_small(mask, min_size, connectivity=1):
    """Igual que skimage.morphology.remove_small_objects(min_size=...)."""
    st = ndi.generate_binary_structure(2, connectivity)
    lab, n = ndi.label(mask, structure=st)
    if n == 0:
        return mask.astype(bool)
    sizes = np.bincount(lab.ravel())
    keep = sizes >= min_size
    keep[0] = False
    return keep[lab]


def haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    a = sin((lat2 - lat1) / 2) ** 2 + cos(lat1) * cos(lat2) * sin((lon2 - lon1) / 2) ** 2
    return 2 * asin(sqrt(a)) * 6371.0


def offset_curve_smooth(path, offset_distance_km, sigma=18):
    off = offset_distance_km / 111.0
    dx = gaussian_filter1d(np.gradient(path[:, 0]), sigma=sigma)
    dy = gaussian_filter1d(np.gradient(path[:, 1]), sigma=sigma)
    mag = np.sqrt(dx ** 2 + dy ** 2)
    p = path.copy()
    p[:, 0] += -dy / mag * off
    p[:, 1] += dx / mag * off
    return p


# ----------------------------------------------------------------------------
# lectura
# ----------------------------------------------------------------------------
def read_appeears_nc(path):
    """Devuelve lon2d, lat2d, fechas (DatetimeIndex) y un lector por día."""
    import netCDF4
    ds = netCDF4.Dataset(path)
    lon = ds['lon'][:].astype(float)
    lat = ds['lat'][:].astype(float)
    lon2d, lat2d = np.meshgrid(lon, lat)
    # units = 'seconds since 2000-01-01'; calendario 'julian' de AppEEARS:
    # entre 1901 y 2099 el conteo de días coincide con el gregoriano.
    t = np.asarray(ds['time'][:], dtype='int64')
    dates = pd.Timestamp('2000-01-01') + pd.to_timedelta(t, unit='s')

    def band(name, i):
        v = ds[name][i, :, :]            # escala y _FillValue aplicados por netCDF4
        return np.ma.filled(v.astype(float), np.nan)

    return ds, lon2d, lat2d, pd.DatetimeIndex(dates).normalize(), band


def land_mask_for_grid(lon2d, lat2d, mat_path=None):
    if mat_path and Path(mat_path).exists():
        try:
            import h5py
            with h5py.File(mat_path, 'r') as f:
                m = np.array(f[list(f.keys())[0]]).astype(bool)
            if m.shape != lon2d.shape and m.T.shape == lon2d.shape:
                m = m.T
            if m.shape == lon2d.shape:
                print('Máscara de tierra: land_mask_modis.mat')
                return m
            print(f'land_mask_modis.mat no coincide con la grilla {lon2d.shape}; uso global_land_mask')
        except Exception as e:
            print('No pude leer land_mask_modis.mat:', e)
    from global_land_mask import globe
    return globe.is_land(lat2d, lon2d)


# ----------------------------------------------------------------------------
# geometría fija (costa y transectas)
# ----------------------------------------------------------------------------
class Geometry:
    def __init__(self, lon2d, lat2d, land_mask):
        self.lon, self.lat = lon2d, lat2d
        self.ocean = remove_small(~land_mask, 4000, connectivity=2)
        self.land = ~self.ocean
        cc = measure.find_contours(self.ocean.astype(float), 0.5)
        cp = max(cc, key=len)
        r, c = cp[:, 0].astype(int), cp[:, 1].astype(int)
        coast = np.column_stack((lon2d[r, c], lat2d[r, c]))
        if np.argmax(coast[:, 1]) > np.argmin(coast[:, 1]):
            coast = coast[::-1]
        i0 = np.argmax(coast[:, 1])
        i1 = np.argmin((coast[:, 0] - P['lon_cut']) ** 2 + (coast[:, 1] - P['lat_cut']) ** 2) + 1
        c_north = coast[i0:i1]
        if len(c_north) < 50:                      # salvaguarda si la máscara cambia el orden
            c_north = coast[min(i0, i1 - 1):max(i0, i1 - 1) + 1]
        offs = np.arange(1.0, P['ancho_total_km'] + P['paso_km'], P['paso_km'])
        self.paths = [offset_curve_smooth(c_north, d) for d in offs]
        self.path_visual = self.paths[-1]
        self.path_middle = self.paths[len(self.paths) // 2]
        self.dacum = []
        for p in self.paths:
            d = np.zeros(len(p))
            for i in range(1, len(p)):
                d[i] = d[i - 1] + haversine(p[i - 1, 0], p[i - 1, 1], p[i, 0], p[i, 1])
            self.dacum.append(d)
        self.sub = ((lat2d >= P['lat_sub'][0]) & (lat2d <= P['lat_sub'][1]) &
                    (lon2d >= P['lon_sub'][0]) & (lon2d <= P['lon_sub'][1]) & self.ocean)
        # área de cada píxel (km²)
        dlat = np.abs(np.gradient(lat2d, axis=0)) * 111.32
        dlon = np.abs(np.gradient(lon2d, axis=1)) * 111.32 * np.cos(np.radians(lat2d))
        self.pix_km2 = dlat * dlon


# ----------------------------------------------------------------------------
# detección de un día
# ----------------------------------------------------------------------------
def detect_day(b1, b2, g: Geometry):
    b1 = b1.copy()
    b1[b1 > P['cloud_reflect_thresh']] = np.nan
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio12 = b2 / np.where(b1 == 0, np.nan, b1)
    ee = b1.copy()
    ee[(ratio12 > 0.9) & (ratio12 < 1.1)] = np.nan
    ee[~g.ocean] = np.nan

    frac_sub = np.count_nonzero(~np.isnan(ee[g.sub])) / max(np.count_nonzero(g.sub), 1)
    quality = 'clear' if frac_sub >= P['subregion_threshold'] else 'cloudy'
    out = dict(quality=quality, frac_sub=frac_sub, area=np.nan, dist=np.nan, lon_cut=np.nan,
               flon=np.array([]), flat=np.array([]), est=None, p_cross=None, ee=ee)

    if np.count_nonzero(~np.isnan(ee)) <= 1000:
        out['quality'] = 'cloudy'
        return out

    een = ee / np.nanmax(ee)
    binary = median_filter(een > threshold_otsu(een[~np.isnan(een)]), size=P['median_ksize'])
    binary = closing(remove_small(binary.astype(bool), P['min_object_size']), disk(P['se_disk_radius']))
    binary[~g.ocean] = False

    edges = sobel(binary.astype(float)) > 0
    edges[ndi.uniform_filter(g.land.astype(float), size=4) > 0] = False
    lab = measure.label(edges)
    props = measure.regionprops(lab)
    if not props:
        out['quality'] = 'cloudy'
        return out

    big = props[np.argmax([p.area for p in props])]
    cidx = max(measure.find_contours((lab == big.label).astype(float), 0.5), key=len)
    r, c = cidx[:, 0].astype(int), cidx[:, 1].astype(int)
    flon, flat = g.lon[r, c], g.lat[r, c]
    frac_nan = np.mean([np.any(np.isnan(ee[max(ri - 1, 0):ri + 2, max(ci - 1, 0):ci + 2]))
                        for ri, ci in zip(r, c)])
    if frac_nan > P['nan_threshold']:
        out['quality'] = 'cloudy'

    # área estuarina: píxeles turbios al oeste/norte del frente (= filtro valid_box original)
    lon_min, _, _, lat_max = P['dom']
    vbox = ((g.lon >= lon_min) & (g.lon <= flon.max() + 0.01) &
            (g.lat >= flat.min() - 0.05) & (g.lat <= lat_max))
    est = binary & vbox
    area = float(np.sum(g.pix_km2[est]))

    # NCL: distancia media a lo largo de las 15 transectas costeras
    pts = np.column_stack((flon, flat))
    lons, dists = [], []
    for p, d in zip(g.paths, g.dacum):
        dm = np.sqrt(((pts[:, None, :] - p[None, :, :]) ** 2).sum(axis=2))
        i_f, i_p = np.unravel_index(np.argmin(dm), dm.shape)
        d_km = -(d[i_p] - d[np.argmin(np.abs(p[:, 0] - P['lon_ref']))])
        lons.append(flon[i_f])
        dists.append(d_km)
    lon_v, dist_v = float(np.mean(lons)), float(np.mean(dists))
    p_cross = g.path_middle[np.argmin(np.abs(g.path_middle[:, 0] - lon_v))]

    out.update(area=area, dist=dist_v, lon_cut=lon_v, flon=flon, flat=flat,
               est=est, p_cross=p_cross, frac_nan=frac_nan)
    return out


# ----------------------------------------------------------------------------
# figura
# ----------------------------------------------------------------------------
def make_figure(res, g: Geometry, date_str, sat, out_png, extra=''):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    lon_min, lon_max, lat_min, lat_max = P['dom']
    fig, axs = plt.subplots(1, 2, figsize=(16, 6.0), constrained_layout=True)
    land_cmap = ListedColormap(['#f5f5dc'])
    ee = res['ee']
    info = (f"{date_str}  ·  MODIS {sat}\n"
            f"Distancia FT (NCL): {res['dist']:.0f} km\n"
            f"Área estuarina: {res['area']:,.0f} km²").replace(',', '.')
    for i, ax in enumerate(axs):
        ax.pcolormesh(g.lon, g.lat, np.where(g.land, 1, np.nan), cmap=land_cmap,
                      shading='auto', zorder=2)
        ax.contour(g.lon, g.lat, g.ocean.astype(float), [0.5], colors='#8c8c8c',
                   linewidths=0.6, zorder=3)
        if i == 0:
            v0, v1 = np.nanpercentile(ee, 2), np.nanpercentile(ee, 98)
            pm = ax.pcolormesh(g.lon, g.lat, ee, vmin=v0, vmax=v1, cmap='turbo',
                               shading='auto', zorder=1)
            fig.colorbar(pm, ax=ax, shrink=0.75, label='Reflectancia banda 1 (645 nm)')
            ax.set_title('Reflectancia en el rojo (proxy de turbidez)')
        else:
            if res['est'] is not None:
                ax.contourf(g.lon, g.lat, res['est'].astype(float), [0.5, 1.5],
                            colors=['#ff9933'], alpha=0.6, zorder=1)
            ax.set_title('Zona estuarina turbia y frente detectado')
        if len(res['flon']):
            ax.plot(res['flon'], res['flat'], 'r-', lw=2.2, zorder=4, label='Frente de turbidez')
        ax.plot(g.path_visual[:, 0], g.path_visual[:, 1], color='hotpink', ls='--', lw=1.3,
                zorder=5, label='Transecta costera')
        if res['p_cross'] is not None:
            ax.scatter(*res['p_cross'], marker='+', color='magenta', s=250, lw=4, zorder=10)
        ax.text(0.50, 0.97, info, transform=ax.transAxes, va='top', ha='center', fontsize=11,
                fontweight='bold', bbox=dict(facecolor='white', alpha=0.9, edgecolor='black'),
                zorder=11)
        ax.set_xlim(lon_min, lon_max)
        ax.set_ylim(lat_min, lat_max)
        ax.set_aspect(1.0 / np.cos(np.radians(-35)))
        ax.grid(True, ls=':', alpha=0.4)
        ax.set_xlabel('Longitud')
    axs[0].set_ylabel('Latitud')
    axs[1].legend(loc='lower right', fontsize=9)
    fig.suptitle('Frente de turbidez del Río de la Plata – última detección válida', fontsize=14,
                 fontweight='bold')
    if extra:
        fig.text(0.995, 0.005, extra, ha='right', va='bottom', fontsize=8, color='0.35')
    fig.savefig(out_png, dpi=130)
    plt.close(fig)
