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
# Definimos límites fijos para no tener que calcularlos al inicio (puedes ajustarlos si ves que el rango es otro)
global_limits = {
    'sardine': {'min': 1024.0, 'max': 1028.0},
    'anchovy': {'min': 1024.0, 'max': 1028.0},
    'hake':    {'min': 1024.0, 'max': 1028.0}
}

print("Procesado de datos biológicos (carga rápida)...")
for esp in especies:
    fichero_simu = f"Galicia_roms_{esp}_n_h_20250415.nc"
    path_part = os.path.join(ruta_base, fichero_simu)
    
    ds = xr.open_dataset(path_part, engine='netcdf4')
    mask_huevo = ds['hatched'] < 1  
    
    # Solo guardamos lo necesario para filtrar y plotear
    dens_solo_huevo = ds['egg_to_larvae_dens'].where(mask_huevo)
    ds['frozen_egg_dens'] = dens_solo_huevo.ffill(dim='time')
    
    part_datasets[esp] = ds
    print(f"{esp.upper()} cargado.")

tiempos_simu = part_datasets["sardine"].time.values

# Eclosión del último huevo (para saber hasta dónde procesar)
ultimo_tiempo_con_huevos = tiempos_simu[0]
for esp in especies:
    ds = part_datasets[esp]
    mask_huevo = ds['hatched'] < 1
    tiempos_activos = ds.time.where(mask_huevo.any(dim='trajectory'), drop=True).values
    if len(tiempos_activos) > 0:
        ultimo_tiempo_con_huevos = max(ultimo_tiempo_con_huevos, tiempos_activos[-1])

print(f"\nEl último huevo de la simulación eclosiona el: {pd.to_datetime(ultimo_tiempo_con_huevos)}")

# Generar fechas a procesar
intervalo_horas = 6
fechas_a_procesar = [tiempos_simu[0]]
delta_ns = np.timedelta64(intervalo_horas, 'h')
for t in tiempos_simu[1:]:
    if t <= ultimo_tiempo_con_huevos:
        if t - fechas_a_procesar[-1] >= delta_ns:
            fechas_a_procesar.append(t)

def obtener_url_roms(dt64):
    dt = pd.to_datetime(dt64)
    yyyy, mm = dt.strftime('%Y'), dt.strftime('%m')
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

# --- CONMUTADOR DE DENSIDAD ---
# 'nc_water', 'gsw', 'unesco_water', 'egg_dens', 'diff_gsw', 'diff_unesco', 'diff_gsw_unesco'
TIPO_DENSIDAD = 'nc_water'

print(f'Plotting (Modo: {TIPO_DENSIDAD})...')

shared_cmap = 'turbo'
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

print(
    'Calculando límites globales basados EXCLUSIVAMENTE en partículas'
    ' (sin t0)...'
)
valores_globales = []

fechas_evaluacion = (
    fechas_a_procesar[1:]
    if len(fechas_a_procesar) > 1
    else fechas_a_procesar
)

for fecha_target in fechas_evaluacion:
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

      c_validos = c_p[~np.isnan(c_p)]
      if len(c_validos) > 0:
        valores_globales.extend(c_validos)

if len(valores_globales) > 0:
  vmin_global = np.min(valores_globales)
  vmax_global = np.max(valores_globales)
  if vmin_global == vmax_global:
    vmin_global -= 0.05
    vmax_global += 0.05
else:
  vmin_global, vmax_global = 1025.5, 1027.0

print(
    f'Rango Global de Partículas ({TIPO_DENSIDAD.upper()}):'
    f' {vmin_global:.3f} - {vmax_global:.3f} kg/m³'
)

scatter_cmap = (
    'coolwarm'
    if TIPO_DENSIDAD in ['diff_gsw', 'diff_unesco', 'diff_gsw_unesco']
    else shared_cmap
)
cbar_label = (
    f'Water Density at Particle Position ({TIPO_DENSIDAD.upper()}) [kg/m³]'
)

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

      nans_cp = np.isnan(c_p).sum()
      if nans_cp > 0:
        c_p = np.nan_to_num(c_p, nan=vmin_global)

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

  # Densidad de fondo según el tipo activo
  lat0_bg = np.nanmean(coord_horizontal)
  p_bg = gsw.p_from_z(z_r_sec, lat0_bg)

  if TIPO_DENSIDAD in ['nc_water', 'unesco_water', 'diff_unesco']:
    rho_sec = sw.dens(salt_sec, temp_sec, np.abs(z_r_sec))
  else:
    sa_bg = gsw.SA_from_SP(salt_sec, p_bg, lat0_bg, lat0_bg)
    ct_bg = gsw.CT_from_t(sa_bg, temp_sec, p_bg)
    rho_sec = gsw.rho(sa_bg, ct_bg, p_bg)

  fig, ax = plt.subplots(figsize=(12, 6.5))
  ax.set_axisbelow(True)
  ax.grid(True, linestyle='--', alpha=0.4, zorder=0)

  #Determinar primero los límites horizontales del encuadre
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

  # filtrar los datos de densidad estrictamente DENTRO de la ventana visible (X e Y)
  mask_visible = (
      (coord_2d >= x_min_plot)
      & (coord_2d <= x_max_plot)
      & (depth_sec >= depth_min)
      & (depth_sec <= depth_limit)
      & ~np.isnan(rho_sec)
  )
  rho_visible = rho_sec[mask_visible]

  # generar las 25 isopicnas basándose en el rango visible del recuadro
  if len(rho_visible) > 0:
    rho_min_vis = np.min(rho_visible)
    rho_max_vis = np.max(rho_visible)
    levels = np.linspace(rho_min_vis, rho_max_vis, num_isopicnas)
  else:
    levels = np.linspace(vmin_global, vmax_global, num_isopicnas)

  rho_masked = np.ma.masked_invalid(rho_sec)

  # dibujar contornos: levels cubre la ventana 0-100m / X_min-X_max
  ax.contour(
      coord_2d,
      depth_sec,
      rho_masked,
      levels=levels,
      cmap=scatter_cmap,
      vmin=vmin_global,
      vmax=vmax_global,
      linewidths=1.0,
      zorder=1,
  )

  # Partículas (usan la misma escala acotada vmin_global - vmax_global)
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
          vmin=vmin_global,
          vmax=vmax_global,
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

  # Colorbar
  cax_unica = fig.add_axes([0.25, 0.07, 0.50, 0.025])
  sm_unificado = cm.ScalarMappable(
      norm=plt.Normalize(vmin=vmin_global, vmax=vmax_global), cmap=scatter_cmap
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
