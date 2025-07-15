import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import imageio
import pickle
import os
import sys

# Add pygtm path
sys.path.append('/southern/shaipollak')

from pygtm.physical import physical_space
from pygtm.matrix import matrix_space

# ---------------------- User Configuration ----------------------

debug_file_name = "20250604_230117"
res = 240           # spatial resolution (number of bins in the main axis)
T = 1                # transition time in days
duration = 31        # total days to push forward
initial_bin = None   # set to int for delta, or None for uniform
make_gif = True

lon_bounds = [31, 37]
lat_bounds = [31, 37]

mother_dir = f"/southern/shaipollak/parcels_analysis/data/{debug_file_name}"
matrix_dir = os.path.join(mother_dir, "matrix_data")
output_dir = os.path.join(mother_dir, f"Markov_Analysis_Images_bin_{initial_bin}_T_{T}_res_{res}")
os.makedirs(output_dir, exist_ok=True)

# ---------------------- Load Matrix and Domain ----------------------

matrix_path = os.path.join(matrix_dir, f"transition_matrix_T{T}_res={res}.npy")
metadata_path = os.path.join(matrix_dir, f"matrix_metadata_T{T}_res={res}.pkl")

print(f"Loading matrix from: {matrix_path}")
print(f"Loading metadata from: {metadata_path}")

if not os.path.exists(matrix_path) or not os.path.exists(metadata_path):
    raise FileNotFoundError("Matrix or metadata file not found.")

P = np.load(matrix_path)

with open(metadata_path, 'rb') as f:
    meta = pickle.load(f)

bins = meta['domain_bins']
id_og = meta['domain_id_og']
vx = meta['domain_lon']
vy = meta['domain_lat']

# Rebuild physical space
d = physical_space(lon_bounds, lat_bounds, res)
d.bins = bins
d.id_og = id_og
d.lon = vx
d.lat = vy
d.vx = vx
d.vy = vy

# ---------------------- Initial Density ----------------------

if initial_bin is not None:
    print("Setting initial density to a delta function at bin:", initial_bin)
    density = np.zeros(P.shape[0], dtype=np.float32)
    density[initial_bin] = 1.0
    density /= np.sum(density)
else:
    print("Setting initial density to uniform distribution.")
    density = np.ones(P.shape[0], dtype=np.float32) / P.shape[0]

# ---------------------- Plotting Functions ----------------------

def geo_map(ax):
    ax.set_xticks(np.arange(31, 38), crs=ccrs.PlateCarree())
    ax.set_yticks(np.arange(31, 38), crs=ccrs.PlateCarree())
    ax.xaxis.set_major_formatter(LongitudeFormatter())
    ax.yaxis.set_major_formatter(LatitudeFormatter())
    ax.tick_params(axis='x', labelsize=7)
    ax.tick_params(axis='y', labelsize=7)
    ax.add_feature(cfeature.LAND, facecolor='silver', zorder=1)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.25, zorder=1)

# ---------------------- Push Forward Loop ----------------------

steps = int(duration / T)
for i in range(steps + 1):
    dvec = density.copy()
    for _ in range(i):
        dvec = dvec @ P

    dmat = d.vector_to_matrix(dvec)
    dmat = np.nan_to_num(dmat, nan=0.0)

    fig = plt.figure(figsize=(4, 3), dpi=300)
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    geo_map(ax)

    im = ax.pcolormesh(vx, vy, dmat, shading='nearest', cmap="cmo.matter", transform=ccrs.PlateCarree())
    cb = fig.colorbar(im, extend='both')
    cb.set_label(f'Probability (Day {i*T})', size=7, labelpad=2)

    plt.savefig(os.path.join(output_dir, f'density_{i:02d}.png'))
    plt.close()

# ---------------------- Create GIF ----------------------

if make_gif:
    images = [imageio.imread(os.path.join(output_dir, f)) for f in sorted(os.listdir(output_dir)) if f.endswith(".png")]
    gif_path = os.path.join(output_dir, "pushforward.gif")
    imageio.mimsave(gif_path, images, duration=1.6)
    print(f"GIF created at: {gif_path}")
