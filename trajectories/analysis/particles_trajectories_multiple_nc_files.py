import sys
import warnings
import os
import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import LineCollection
import cartopy.feature as cfeature
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable
from datetime import datetime as dt, timedelta as td
import argparse
import glob

# ─────────────────────────────────────────────────────────────────────────────
sys.path.insert(0, '/southern/shaipollak')
from pygtm.physical import physical_space
from pygtm.matrix import matrix_space
from pygtm.dataset2 import trajectory as trajectory2

parser = argparse.ArgumentParser(description="Run Particles Trajectories Simulation.")
parser.add_argument("--simulation_date_and_time", type=str, help="Simulation Date and time in format YYYYMMDD_HHMMSS")
parser.add_argument("--parcels_file_path", type=str, help="Path to the Parcels output NetCDF file.")
parser.add_argument("--follow_particle_id", type=int, help="Follow a single particle [id] trajectory.")
parser.add_argument("--particle_ids", type=int, nargs='+', help="List of particle IDs to plot trajectories for.")
args = parser.parse_args()

# --- Config and paths ---

####################Put the simulation date and time that you want to analyze here####################

debug_name = '20250602_020307' 


######################################################################################################

simulation_time_stamp = args.simulation_date_and_time if args.simulation_date_and_time else debug_name
nc_file_directory = f"/southern/shaipollak/parcels_analysis/data/{simulation_time_stamp}"
file_name = simulation_time_stamp
output_directory = f"{nc_file_directory}/particles_trajectories"
os.makedirs(output_directory, exist_ok=True)

# --- Particle ID ---
debug_particle_id = 15
follow_id = args.follow_particle_id if args.follow_particle_id is not None else debug_particle_id

# --- Load files ---
print(f"Looking for files: {simulation_time_stamp}_proc*.nc")
nc_file_pattern = os.path.join(nc_file_directory, f"{simulation_time_stamp}_proc*.nc")
nc_files = sorted(glob.glob(nc_file_pattern))

if not nc_files:
    raise FileNotFoundError(f"No files found for pattern: {nc_file_pattern}")

x, y, t, d_id = [], [], [], []
for path in nc_files:
    print(f"Reading file: {path}")
    with nc.Dataset(path) as ds:
        lon_par = np.array(ds['lon'][:], dtype=np.float32)
        lat_par = np.array(ds['lat'][:], dtype=np.float32)
        id_par = np.array(ds['trajectory'][:], dtype=np.int32)
        time_par = np.array(ds['time'][:], dtype=np.float32)

        for i in range(time_par[:, 0].size):
            lon_i = np.nan_to_num(lon_par[i], nan=0).flatten()
            lat_i = np.nan_to_num(lat_par[i], nan=0).flatten()
            time_i = np.nan_to_num(time_par[i], nan=0).flatten()
            id_i = id_par[i].flatten()

            x.append(np.array(lon_i, dtype=np.float32))
            y.append(np.array(lat_i, dtype=np.float32))
            t.append(np.array(time_i, dtype=np.float32))
            d_id.append(np.array(id_i, dtype=np.int32))

x_conc = np.concatenate(x)
y_conc = np.concatenate(y)
t_conc = np.concatenate(t)
d_id_conc = np.concatenate(d_id)

# --- Time conversion helpers ---
def datenum(stripped_date):
    days = stripped_date.toordinal()
    hours = (stripped_date - dt.fromordinal(days)).total_seconds() / (24 * 60 * 60)
    return days + hours

def numdate(num):
    days = num - 365
    whole_days = int(days)
    fractional_days = days - whole_days
    date = dt.fromordinal(whole_days)
    time_delta = td(seconds=fractional_days * 24 * 60 * 60)
    return date + time_delta

d = dt.strptime('29/01/2018 01:00', '%d/%m/%Y %H:%M')
start_time = datenum(d)

# --- Compute t_range from data ---
t_all = np.concatenate(t)
t_all = t_all[np.isfinite(t_all) & (t_all < 1e7)]
t_range = [np.min(t_all), np.max(t_all)]

# --- Spatial bounds ---
x_range = [31, 37]
y_range = [31, 37]

# --- Helper: Cartopy styling ---
def style_cartopy_map(ax, extent):
    ax.set_xlim(extent[:2])
    ax.set_ylim(extent[2:])
    ax.set_xticks([30, 32, 34, 36, 38], crs=ccrs.PlateCarree())
    ax.set_yticks([30, 32, 34, 36, 38], crs=ccrs.PlateCarree())
    ax.xaxis.set_major_formatter(LongitudeFormatter())
    ax.yaxis.set_major_formatter(LatitudeFormatter())
    ax.add_feature(cfeature.LAND, facecolor='grey', zorder=1)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.25, zorder=1)


def draw_starting_points(n):
    print("Plotting Starting Points...")
    x_start = np.array([traj[0] for traj in x], dtype=np.float32)
    y_start = np.array([traj[0] for traj in y], dtype=np.float32)
    mask = ~((x_start == 0) & (y_start == 0))
    x_start, y_start = x_start[mask], y_start[mask]
    total = len(x_start)
    if total > n:
        idx = np.random.choice(total, n, replace=False)
        x_start, y_start = x_start[idx], y_start[idx]
    fig, ax = plt.subplots(figsize=(6, 6), dpi=300, subplot_kw={'projection': ccrs.PlateCarree()})
    ax.scatter(x_start, y_start, s=0.5, color='blue', zorder=3)
    style_cartopy_map(ax, extent=x_range + y_range)
    ax.set_title(f"Starting Points of {n} out of {len(d_id)} Particles")
    plt.savefig(f"{output_directory}/starting_points.png", bbox_inches="tight")
    plt.close()

def draw_ending_points(n):
    '''
    Plot the ending points of the trajectories.
    '''

    print("Plotting Ending Points...")
    x_end = np.array([traj[-1] for traj in x], dtype=np.float32)
    y_end = np.array([traj[-1] for traj in y], dtype=np.float32)
    mask = ~((x_end == 0) & (y_end == 0))
    x_end, y_end = x_end[mask], y_end[mask]
    total = len(x_end)
    if total > n:
        idx = np.random.choice(total, n, replace=False)
        x_end, y_end = x_end[idx], y_end[idx]
    fig, ax = plt.subplots(figsize=(6, 6), dpi=300, subplot_kw={'projection': ccrs.PlateCarree()})
    ax.scatter(x_end, y_end, s=0.5, color='red', zorder=3)
    style_cartopy_map(ax, extent=x_range + y_range)
    ax.set_title(f"Ending Points of {n} out of {len(d_id)} Particles")
    plt.savefig(f"{output_directory}/ending_points.png", bbox_inches="tight")
    plt.close()

def draw_manual_trajectories(n=10, particle_ids=None, save=True, point_step=5):
    '''
    'Plot trajectories of particles, optionally following specific particle IDs.'
    '''

    print("Ploting Trajectories...")
    unique_ids = np.unique(d_id_conc)
    if particle_ids is not None:
        sampled_ids = [pid for pid in particle_ids if pid in unique_ids]
    else:
        sampled_ids = np.random.choice(unique_ids, size=min(n, len(unique_ids)), replace=False)

    fig, ax = plt.subplots(figsize=(8, 8), dpi=300, subplot_kw={'projection': ccrs.PlateCarree()})
    style_cartopy_map(ax, extent=x_range + y_range)
    ax.set_title(f"Trajectories of {len(sampled_ids)} particles")

    colors = plt.cm.viridis(np.linspace(0, 1, len(sampled_ids)))
    for i, pid in enumerate(sampled_ids):
        particle_indices = [j for j, ids in enumerate(d_id) if pid in ids]
        for idx in particle_indices:
            lon = x[idx][::point_step]
            lat = y[idx][::point_step]
            ax.plot(lon, lat, '-', color=colors[i], lw=0.2, zorder=2)
            ax.scatter(lon[0], lat[0], s=1, color='blue', zorder=3)
            ax.scatter(lon[-1], lat[-1], s=1, color='red', zorder=3)

    ax.plot([], [], ' ', label=f'Duration: {t_range[1] - t_range[0]:.1f} days')
    ax.legend(fontsize=6, loc='lower left')
    if save:
        plt.savefig(f"{output_directory}/manual_trajectories_custom.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

# === Execute plots ===

particles = 5000

draw_starting_points(particles)
draw_ending_points(particles)
draw_manual_trajectories(n=particles, save=True)