import os
import warnings
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import gsw
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, MaxNLocator
import numpy as np
import owslib
import pandas as pd
import seawater as sw
import xarray as xr

###  ------------------------------------------------------- ###
def sea_water_density(T=10.0, S=35.0):
  """Ecuación de estado de agua de mar a 1 atm (UNESCO 1983 / OpenDrift).

  S = Salinidad práctica (PSU / ppm) T = Temperatura (°C)
  """
  if np.atleast_1d(T).max() > 100:
    raise ValueError('Temperature should be in celsius, but is > 100')

  R4 = 4.8314e-04
  DR350 = 28.106331

  # Densidad del agua pura a presión atmosférica
  R1 = ((((6.536332e-09 * T - 1.120083e-06) * T + 1.001685e-04) * T - 9.095290e-03) * T + 6.793952e-02) * T - 28.263737

  # Coeficientes con salinidad
  R2 = (((5.3875e-09 * T - 8.2467e-07) * T + 7.6438e-05) * T - 4.0899e-03) * T + 8.24493e-01
  R3 = (-1.6546e-06 * T + 1.0227e-04) * T - 5.72466e-03

  # Ecuación de estado final
  SIG = R1 + (R4 * S + R3 * np.sqrt(S) + R2) * S
  Dens0 = SIG + DR350 + 1000.0
  return Dens0

###  ------------------------------------------------------- ###
def compute_z_r_section(h_sec, zeta_sec, hc, Cs_r, theta_s, theta_b, Vtransform):
    import numpy as np
    N = len(Cs_r)
    pts = h_sec.shape[0]
    z_r_sec = np.zeros((N, pts))
    sc_r = (np.arange(1, N + 1) - N - 0.5) / N
    
    for k in range(N):
        if Vtransform == 1:
            z0 = hc * (sc_r[k] - Cs_r[k]) + Cs_r[k] * h_sec
            z_r_sec[k] = z0 + zeta_sec * (1 + z0 / h_sec)
        elif Vtransform == 2:
            z0 = (hc * sc_r[k] + Cs_r[k] * h_sec) / (hc + h_sec)
            z_r_sec[k] = zeta_sec + (zeta_sec + h_sec) * z0
        z_r_sec[k] = np.minimum(z_r_sec[k], 0)
        z_r_sec[k] = np.maximum(z_r_sec[k], -h_sec)
    return z_r_sec

###  ------------------------------------------------------- ###

print(owslib.__version__)
warnings.filterwarnings("ignore", category=UserWarning, module="argopy")

ruta_base = "/home/adrian/Escritorio/conexion_cigala_2"
especies = ["sardine", "anchovy", "hake"]

part_datasets = {}
global_limits = {}

print("Procesado de datos biologicos")
for esp in especies:
    fichero_simu = f"Galicia_roms_{esp}_n_h_20250515.nc"
    path_part = os.path.join(ruta_base, fichero_simu)
    
    ds = xr.open_dataset(path_part, engine='netcdf4')
    mask_huevo = ds['hatched'] < 1  
    
    # Filtramos la densidad original: se queda el valor si es huevo, si es larva pasa a NaN
    dens_solo_huevo = ds['egg_to_larvae_dens'].where(mask_huevo)
    ds['frozen_egg_dens'] = dens_solo_huevo.ffill(dim='time')
    
    part_datasets[esp] = ds
    
    # Límites absolutos basados en la nueva variable congelada
    global_limits[esp] = {
        'min': float(np.nanmin(ds['frozen_egg_dens'].values)),
        'max': float(np.nanmax(ds['frozen_egg_dens'].values))
    }
    print(f"{esp.upper()} Rango de densidades: [{global_limits[esp]['min']:.2f} a {global_limits[esp]['max']:.2f}]")

tiempos_simu = part_datasets["sardine"].time.values

# Eclosión del último huevo
ultimo_tiempo_con_huevos = tiempos_simu[0]

for esp in especies:
    ds = part_datasets[esp]
    mask_huevo = ds['hatched'] < 1  # (Usa el mismo criterio de arriba)
    # Buscamos los tiempos donde al menos una partícula siga siendo huevo
    tiempos_activos = ds.time.where(mask_huevo.any(dim='trajectory'), drop=True).values
    if len(tiempos_activos) > 0:
        ultimo_tiempo_con_huevos = max(ultimo_tiempo_con_huevos, tiempos_activos[-1])

print(f"\nEl último huevo de la simulación eclosiona el: {pd.to_datetime(ultimo_tiempo_con_huevos)}")

intervalo_horas = 6  


fechas_a_procesar = [tiempos_simu[0]]
delta_ns = np.timedelta64(intervalo_horas, 'h')
    
for t in tiempos_simu[1:]:
    if t <= ultimo_tiempo_con_huevos:
        if t - fechas_a_procesar[-1] >= delta_ns:
            fechas_a_procesar.append(t)
                

def obtener_url_roms(dt64):
    dt = pd.to_datetime(dt64)
    yyyy = dt.strftime('%Y')
    mm = dt.strftime('%m')
    yyyymmdd = dt.strftime('%Y%m%d')
    dt_next = dt + pd.Timedelta(days=1)
    yyyymmdd_next = dt_next.strftime('%Y%m%d')
    return f'http://193.144.42.111:8080/thredds/dodsC/models/Almacen_Datos/Pendeco/MeteoGalicia_ROMS_raw/{yyyy}/{mm}/{yyyymmdd}/00/ocean_history_a{yyyymmdd}_{yyyymmdd_next}.nc'

# Usamos la primera especie de la lista como referencia para detectar la orientación del canal
especie_ref = especies[0]  # Tomará 'sardine'
part_ref = part_datasets[especie_ref]

std_lat = np.nanstd(part_ref['lat'].values)
std_lon = np.nanstd(part_ref['lon'].values)

if std_lat < std_lon:
    section_type = "latitudinal"
    target_coord = np.nanmedian(part_ref['lat'].values)
    print(f"[AUTO-DETECT] Corte LATITUDINAL detectado (Este-Oeste) usando {especie_ref.upper()} como referencia.")
    print(f"              Posición fija: {target_coord:.6f}° Norte")
else:
    section_type = "longitudinal"
    target_coord = np.nanmedian(part_ref['lon'].values)
    print(f"[AUTO-DETECT] Corte LONGITUDINAL detectado (Norte-Sur) usando {especie_ref.upper()} como referencia.")
    print(f"              Posición fija: {target_coord:.6f}° Este/Oeste")

# Mini tolerancia por si las moscas
tol = 1e-4

carpeta_salida = 'render_isopicnas'
os.makedirs(carpeta_salida, exist_ok=True)

max_particulas = 50

TIPO_DENSIDAD = 'egg_dens'

print(f'Plotting (Modo: {TIPO_DENSIDAD}) con fondo OpenDrift...')

shared_cmap = 'turbo'
vmin_iso = 1024.5
vmax_iso = 1027.5
num_isopicnas = 25
depth_min, depth_limit = 0, 100

print('Seleccionando conjunto fijo de partículas por especie...')
indices_fijos_especie = {}

for esp in especies:
  ds = part_datasets[esp]
  candidatos_set = set()

  for fecha_target in fechas_a_procesar:
    part_paso = ds.sel(time=fecha_target, method='nearest')
    lon_inst = part_paso['lon'].values
    lat_inst = part_paso['lat'].values
    z_inst = -part_paso['z'].values
    mask_vivas = ~np.isnan(lon_inst) & ~np.isnan(z_inst)

    if section_type == 'longitudinal':
      mask_seccion = mask_vivas & (np.abs(lon_inst - target_coord) <= tol)
    elif section_type == 'latitudinal':
      mask_seccion = mask_vivas & (np.abs(lat_inst - target_coord) <= tol)

    mask_profundidad = (z_inst >= depth_min) & (z_inst <= depth_limit)
    idx_validos = np.where(mask_seccion & mask_profundidad)[0]
    candidatos_set.update(idx_validos)

  candidatos_arr = np.array(sorted(list(candidatos_set)))

  if len(candidatos_arr) > max_particulas:
    np.random.seed(42)
    indices_fijos_especie[esp] = np.random.choice(
        candidatos_arr, max_particulas, replace=False
    )
  else:
    indices_fijos_especie[esp] = candidatos_arr

for num_progreso, fecha_target in enumerate(fechas_a_procesar, start=1):
  fecha_str = pd.to_datetime(fecha_target).strftime('%Y%m%d_%H%M')
  print(
      f'[{num_progreso}/{len(fechas_a_procesar)}] Fecha:'
      f' {pd.to_datetime(fecha_target)}'
  )

  url_roms = obtener_url_roms(fecha_target)
  ds_roms = xr.open_dataset(url_roms)

  if section_type == 'latitudinal':
    lat_mean = ds_roms['lat_rho'].mean(dim='xi_rho').values
    idx = np.argmin(np.abs(lat_mean - target_coord))
    roms_sec = (
        ds_roms.isel(eta_rho=idx)
        .sel(ocean_time=fecha_target, method='nearest')
        .interpolate_na(dim='xi_rho', method='linear')
    )
    coord_horizontal = roms_sec['lon_rho'].values
    coord_label = 'Longitud (°E)'
  elif section_type == 'longitudinal':
    lon_mean = ds_roms['lon_rho'].mean(dim='eta_rho').values
    idx = np.argmin(np.abs(lon_mean - target_coord))
    roms_sec = (
        ds_roms.isel(xi_rho=idx)
        .sel(ocean_time=fecha_target, method='nearest')
        .interpolate_na(dim='xi_rho', method='linear')
    )
    coord_horizontal = roms_sec['lat_rho'].values
    coord_label = 'Latitud (°N)'

  temp_sec = roms_sec['temp'].values
  salt_sec = roms_sec['salt'].values
  zeta_sec = np.nan_to_num(roms_sec['zeta'].values, nan=0)
  h_sec = roms_sec['h'].values

  theta_s = ds_roms['theta_s'].values
  theta_b = ds_roms['theta_b'].values
  hc = ds_roms['hc'].values
  Cs_r = ds_roms['Cs_r'].values
  Vtransform = ds_roms['Vtransform'].values if 'Vtransform' in ds_roms else 2

  z_r_sec = compute_z_r_section(
      h_sec, zeta_sec, hc, Cs_r, theta_s, theta_b, Vtransform
  )
  depth_sec = -z_r_sec
  coord_2d = np.tile(coord_horizontal, (z_r_sec.shape[0], 1))

  rho_sec = sea_water_density(T=temp_sec, S=salt_sec)
  particulas_filtradas = {}
  promedios_desarrollo = {}
  all_x_presentes = []

  for esp in especies:
    ds = part_datasets[esp]
    part_paso = ds.sel(time=fecha_target, method='nearest')

    lon_inst = part_paso['lon'].values
    lat_inst = part_paso['lat'].values
    z_inst = -part_paso['z'].values
    mask_vivas = ~np.isnan(lon_inst) & ~np.isnan(z_inst)

    if section_type == 'longitudinal':
      mask_seccion = mask_vivas & (np.abs(lon_inst - target_coord) <= tol)
    elif section_type == 'latitudinal':
      mask_seccion = mask_vivas & (np.abs(lat_inst - target_coord) <= tol)

    mask_profundidad = (z_inst >= depth_min) & (z_inst <= depth_limit)
    indices_seccion = np.where(mask_seccion & mask_profundidad)[0]
    indices_seleccionados = np.intersect1d(
        indices_seccion, indices_fijos_especie[esp]
    )

    if len(indices_seleccionados) > 0:
      x_p = (
          lat_inst if section_type == 'longitudinal' else lon_inst
      )[indices_seleccionados]
      y_p = z_inst[indices_seleccionados]

      s_sel = part_paso['sea_water_salinity'].values[indices_seleccionados]
      t_sel = part_paso['sea_water_temperature'].values[indices_seleccionados]
      z_sel = part_paso['z'].values[indices_seleccionados]
      lat_sel = part_paso['lat'].values[indices_seleccionados]
      lon_sel = part_paso['lon'].values[indices_seleccionados]

      p_sel = gsw.p_from_z(z_sel, lat_sel)
      dens_nc = part_paso['SW_dens'].values[indices_seleccionados]

      sa_sel = gsw.SA_from_SP(s_sel, p_sel, lon_sel, lat_sel)
      ct_sel = gsw.CT_from_t(sa_sel, t_sel, p_sel)
      dens_gsw = gsw.rho(sa_sel, ct_sel, p_sel)
      dens_unesco = sw.dens(s_sel, t_sel, np.abs(z_sel))

      if TIPO_DENSIDAD == 'gsw':
        c_p = dens_gsw
      elif TIPO_DENSIDAD == 'unesco_water':
        c_p = dens_unesco
      elif TIPO_DENSIDAD == 'nc_water':
        c_p = dens_nc
      elif TIPO_DENSIDAD == 'egg_dens':
        c_p = part_paso['frozen_egg_dens'].values[indices_seleccionados]
      elif TIPO_DENSIDAD == 'diff_gsw':
        c_p = dens_gsw - dens_nc
      elif TIPO_DENSIDAD == 'diff_unesco':
        c_p = dens_unesco - dens_nc
      elif TIPO_DENSIDAD == 'diff_gsw_unesco':
        c_p = dens_gsw - dens_unesco

      stage_inst = (
          part_paso['stage_fraction'].values
          if 'stage_fraction' in part_paso
          else np.nan
      )
      promedios_desarrollo[esp] = (
          np.nanmean(stage_inst[indices_seleccionados])
          if not np.all(np.isnan(stage_inst))
          else np.nan
      )

      particulas_filtradas[esp] = {'x': x_p, 'y': y_p, 'c': c_p}
      all_x_presentes.extend(x_p)
    else:
      particulas_filtradas[esp] = {
          'x': np.array([]),
          'y': np.array([]),
          'c': np.array([]),
      }
      promedios_desarrollo[esp] = np.nan

  if TIPO_DENSIDAD in ['diff_gsw', 'diff_unesco', 'diff_gsw_unesco']:
    scatter_cmap = 'coolwarm'
    vmin_scatter, vmax_scatter = -0.5, 0.5
    if TIPO_DENSIDAD == 'diff_gsw':
      cbar_label = 'Diff: GSW(TEOS-10) - NetCDF [kg/m³]'
    elif TIPO_DENSIDAD == 'diff_unesco':
      cbar_label = 'Diff: UNESCO(EOS-80) - NetCDF [kg/m³]'
    else:
      cbar_label = 'Diff: GSW(TEOS-10) - UNESCO(EOS-80) [kg/m³]'
  else:
    scatter_cmap = shared_cmap
    vmin_scatter, vmax_scatter = vmin_iso, vmax_iso
    cbar_label = (
        'Egg Density [kg/m³]'
        if TIPO_DENSIDAD == 'egg_dens'
        else 'Density [kg/m³]'
    )

  fig, ax = plt.subplots(figsize=(12, 6.5))
  ax.set_axisbelow(True)
  ax.grid(True, linestyle='--', alpha=0.4, zorder=0)

  if len(all_x_presentes) > 0:
    buffer = 0.03
    x_min_plot = max(
        np.min(all_x_presentes) - buffer, np.nanmin(coord_horizontal)
    )
    x_max_plot = min(
        np.max(all_x_presentes) + buffer, np.nanmax(coord_horizontal)
    )
  else:
    x_min_plot, x_max_plot = np.nanmin(coord_horizontal), np.nanmax(
        coord_horizontal
    )

  mask_visible = (
      (coord_2d >= x_min_plot)
      & (coord_2d <= x_max_plot)
      & (depth_sec >= depth_min)
      & (depth_sec <= depth_limit)
      & ~np.isnan(rho_sec)
  )
  rho_visible = rho_sec[mask_visible]

  if len(rho_visible) > 0:
    rho_min_vis = np.min(rho_visible)
    rho_max_vis = np.max(rho_visible)
    levels = np.linspace(rho_min_vis, rho_max_vis, num_isopicnas)
  else:
    levels = np.linspace(vmin_iso, vmax_iso, num_isopicnas)

  rho_masked = np.ma.masked_invalid(rho_sec)

  ax.contour(
      coord_2d,
      depth_sec,
      rho_masked,
      levels=levels,
      cmap=shared_cmap,
      vmin=vmin_iso,
      vmax=vmax_iso,
      linewidths=1.0,
      zorder=1,
  )

  marcadores = {'sardine': 'o', 'anchovy': 's', 'hake': '^'}
  for esp, datos in particulas_filtradas.items():
    if len(datos['x']) > 0:
      ax.scatter(
          datos['x'],
          datos['y'],
          c=datos['c'],
          cmap=scatter_cmap,
          marker=marcadores[esp],
          s=50,
          edgecolor='black',
          linewidth=0.6,
          vmin=vmin_scatter,
          vmax=vmax_scatter,
          zorder=3,
      )

  ax.set_xlim(x_min_plot, x_max_plot)
  ax.set_ylim(depth_limit, depth_min - 2)
  ax.set_title(
      f'Vertical Section ({section_type.upper()}) | Mode: {TIPO_DENSIDAD.upper()}'
      f' | Coord: {target_coord:.4f}° | Date:'
      f" {pd.to_datetime(fecha_target).strftime('%Y-%d-%m %H:%M')}",
      fontsize=11,
      fontweight='bold',
  )
  ax.set_xlabel(coord_label)
  ax.set_ylabel('Depth (m)')

  elementos_leyenda = []
  nombres_esp = {'sardine': 'Sardine', 'anchovy': 'Anchovy', 'hake': 'Hake'}
  for esp in ['sardine', 'anchovy', 'hake']:
    st = promedios_desarrollo.get(esp, np.nan)
    st_txt = f'{st:.2f}' if not np.isnan(st) else 'N/A'
    elementos_leyenda.append(
        Line2D(
            [0],
            [0],
            marker=marcadores[esp],
            color='w',
            label=f'{nombres_esp[esp]} (Avg Stage: {st_txt})',
            markerfacecolor='gray',
            markeredgecolor='black',
            markersize=8,
        )
    )

  ax.legend(
      handles=elementos_leyenda, loc='lower right', fontsize=9, framealpha=0.9
  ).set_zorder(4)
  plt.subplots_adjust(left=0.08, right=0.95, top=0.88, bottom=0.20)

  cax_unica = fig.add_axes([0.25, 0.07, 0.50, 0.025])
  sm_unificado = cm.ScalarMappable(
      norm=plt.Normalize(vmin=vmin_scatter, vmax=vmax_scatter),
      cmap=scatter_cmap,
  )
  cbar = fig.colorbar(
      sm_unificado, cax=cax_unica, orientation='horizontal', extend='both'
  )

  cbar.formatter.set_useOffset(False)
  cbar.formatter.set_scientific(False)
  cbar.ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
  cbar.update_ticks()

  cbar.set_label(cbar_label, fontweight='bold', fontsize=10)
  cbar.ax.tick_params(labelsize=8)

  nombre_archivo = f'corte_isopicnas_{fecha_str}.png'
  plt.savefig(os.path.join(carpeta_salida, nombre_archivo), dpi=150)
  plt.close(fig)
  print(f'Guardado: {nombre_archivo}')
