import os
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

path_normal = "/home/adrian/Escritorio/conexion_cigala"
path_diff0  = "/home/adrian/Escritorio/conexion_cigala_2"

rutas = [path_normal, path_diff0]
fechas = ["20250415", "20250515"]
especies = ["anchovy", "sardine", "hake"]

estilos_especies = {
    'anchovy': {'color': 'tab:blue',  'label': 'Anchovy'},
    'sardine': {'color': 'tab:red',   'label': 'Sardine'},
    'hake':    {'color': 'tab:green', 'label': 'Hake'}
}

titulos_filas = ["Normal", "Vertical diffusivity = 0"]
titulos_columnas = ["Simulation start date 15-04-2025", "Simulation start date 15-05-2025"]

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(14, 9), sharex='col', sharey=True)

for row_idx, ruta in enumerate(rutas):
    for col_idx, fecha in enumerate(fechas):
        ax = axes[row_idx, col_idx]
        
        for esp in especies:
            nombre_archivo = f"Galicia_roms_{esp}_n_h_{fecha}.nc"
            filepath = os.path.join(ruta, nombre_archivo)
            
            if os.path.exists(filepath):
                with xr.open_dataset(filepath) as ds:
                    dim_particulas = [d for d in ds['egg_to_larvae_dens'].dims if d != 'time'][0]
                    mask_huevo = ds['hatched'] < 1
                    dens_huevos = ds['egg_to_larvae_dens'].where(mask_huevo)
                    dens_mean = dens_huevos.mean(dim=dim_particulas)
                    dens_mean_valid = dens_mean.dropna(dim='time')
                    
                    if len(dens_mean_valid) > 1:
                        densidad_media = dens_mean_valid.values[1:]
                        tiempos = dens_mean_valid['time'].values[1:]
                        
                        ax.plot(tiempos, densidad_media, 
                                label=estilos_especies[esp]['label'], 
                                color=estilos_especies[esp]['color'], 
                                linewidth=2)
            else:
                print(f"Aviso: No se encontró {filepath}")

        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc='upper right', fontsize=9)
        
        if row_idx == 0:
            ax.set_title(titulos_columnas[col_idx], fontsize=12, fontweight='bold', pad=10)
            
        if col_idx == 0:
            ax.set_ylabel(f"[{titulos_filas[row_idx]}]\nDensity (kg/m³)", fontsize=10, fontweight='bold')
            
        if row_idx == 1:
            ax.set_xlabel("Date", fontsize=10)
            ax.tick_params(axis='x', rotation=25)

plt.suptitle("Mean density, egg stage", fontsize=14, fontweight='bold', y=0.98)
plt.tight_layout()

plt.savefig("comparativa_densidades_huevos_sin_inicio_2x2.png", dpi=300)
plt.show()
