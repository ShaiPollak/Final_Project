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
from matplotlib.collections import LineCollection

# ─────────────────────────────────────────────────────────────────────────────
# Imports from local modules
sys.path.insert(0, '/southern/shaipollak')
from pygtm.physical import physical_space
from pygtm.matrix import matrix_space
from pygtm.dataset2 import trajectory as trajectory2

# -----------------------------------------------Config--------------------------------------------------
# Arguments and settings
parser = argparse.ArgumentParser(description="Run Particles Trajectories Simulation.")
parser.add_argument("--simulation_date_and_time", type=str, help="Simulation Date and time in format YYYYMMDD_HHMMSS")
parser.add_argument("--parcels_file_path", type=str, help="Path to the Parcels output NetCDF file.")
parser.add_argument("--follow_particle_id", type=int, help="Follow a single particle [id] trajectory.")
parser.add_argument("--particle_ids", type=int, nargs='+', help="List of particle IDs to plot trajectories for.")
args = parser.parse_args()

debug_name = '20250604_171817'

simulation_time_stamp = args.simulation_date_and_time if args.simulation_date_and_time is not None else debug_name
nc_file_directory = f"/southern/shaipollak/parcels_analysis/data/{simulation_time_stamp}"
nc_file_path = args.parcels_file_path if args.parcels_file_path else os.path.join(nc_file_directory, f"{debug_name}.nc")
file_name = simulation_time_stamp
output_directory = f"/southern/shaipollak/parcels_analysis/data/{file_name}/particles_trajectories"
os.makedirs(output_directory, exist_ok=True)

print(f"Using NetCDF file: {nc_file_path}")

debug_particle_id = 15
follow_id = args.follow_particle_id if args.follow_particle_id is not None else debug_particle_id

#---------------------------------------------Visualization and time functions------------------------------------------------
# ─────────────────────────────────────────────────────────────────────────────
# Date conversion
'''
def datenum(stripped_date):
    days = 365 + d.toordinal()
    hours = (stripped_date - dt.fromordinal(d.toordinal())).total_seconds() / (24 * 60 * 60)
    return days + hours
'''

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

# ─────────────────────────────────────────────────────────────────────────────
# Helper: styling cartopy map

def style_cartopy_map(ax, extent):
    ax.set_xlim(extent[:2])
    ax.set_ylim(extent[2:])
    ax.set_xticks([30, 32, 34, 36, 38], crs=ccrs.PlateCarree())
    ax.set_yticks([30, 32, 34, 36, 38], crs=ccrs.PlateCarree())
    ax.xaxis.set_major_formatter(LongitudeFormatter())
    ax.yaxis.set_major_formatter(LatitudeFormatter())
    ax.add_feature(cfeature.LAND, facecolor='grey', zorder=1)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.25, zorder=1)

# ─────────────────────────────────────────────────────────────────────────────
#---------------------------------------------Load Data Sets------------------------------------------------

# Load particle data

print(f"Reading file: {nc_file_path}")
ds = nc.Dataset(nc_file_path)
lon_par = np.array(ds['lon'][:], dtype=np.float32)
lat_par = np.array(ds['lat'][:], dtype=np.float32)
id_par = np.array(ds['trajectory'][:], dtype=np.int32)
time_par = np.array(ds['time'][:], dtype=np.float32)

ds.close()

# Convert to list structure
x, y, t, d_id = [], [], [], []
tiks = 450
time_jump = 2 / 24
files_n = tiks

# Generate time array
time_list = np.arange(0, tiks, dtype=np.float32) * time_jump + start_time

for i in range(time_par[:, 0].size):
    lon_i = np.nan_to_num(lon_par[i], nan=0).flatten()
    lat_i = np.nan_to_num(lat_par[i], nan=0).flatten()
    time_i = np.nan_to_num(time_par[i], nan=0).flatten()
    id_i = id_par[i].flatten()

    x.append(lon_i.tolist())
    y.append(lat_i.tolist())
    d_id.append(id_i.tolist())
    t.append(time_i.tolist())

x_conc = np.concatenate(x).astype(np.float32)
y_conc = np.concatenate(y).astype(np.float32)
t_conc = np.concatenate(t).astype(np.float32)
d_id_conc = np.concatenate(d_id).astype(np.int32)

# Trajectory and filtering setup - Does not work at the moment
#data = trajectory2(x, y, t, d_id)

# Debugging: Inspect the data object
#print(f"Data object: {data}")

# Debugging: Check unique IDs in d_id_conc
#unique_ids = np.unique(d_id_conc)
#print(f"Unique particle IDs in d_id_conc: {unique_ids}")

# Filtering
x_range = [31, 37]
y_range = [31, 37]
t_range = [time_list[0], time_list[-1]]
#segs, segs_t, sid = data.filtering(x_range, y_range, t_range, complete_track=False)

# Debugging: Check the filtering output
#print(f"Filtered IDs (sid): {sid}")
#print(f"Filtered segments (segs): {segs}")
#print(f"Filtered time segments (segs_t): {segs_t}")

# ─────────────────────────────────────────────────────────────────────────────
# Functions
def geo_map(ax):
    ax.set_xticks([31, 32, 33, 34, 35, 36, 37], crs=ccrs.PlateCarree())
    ax.set_yticks([31, 32, 33, 34, 35, 36, 37], crs=ccrs.PlateCarree())
    ax.xaxis.set_major_formatter(LongitudeFormatter())
    ax.yaxis.set_major_formatter(LatitudeFormatter())
    ax.tick_params(axis='x', labelsize=7)
    ax.tick_params(axis='y', labelsize=7)
    ax.add_feature(cfeature.LAND, facecolor='silver', zorder=1)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.25, zorder=1)

def draw_manual_trajectories(n=10, particle_ids=None, save=True):
    """
    Plot the trajectories of selected or random particles,
    and show the total simulation duration in the legend.
    """
    print("Ploting Trajectories...")
    unique_ids = np.unique(d_id_conc)
    if particle_ids is not None:
        sampled_ids = [pid for pid in particle_ids if pid in unique_ids]
        if len(sampled_ids) == 0:
            print("None of the provided particle IDs are valid.")
            return
    else:
        if n > len(unique_ids):
            print(f"Only {len(unique_ids)} unique particle IDs available, adjusting n.")
            n = len(unique_ids)
        sampled_ids = np.random.choice(unique_ids, size=n, replace=False)

    # Define the region bounds
    x_min, x_max = x_range[0], x_range[1]
    y_min, y_max = y_range[0], y_range[1]   

    # Create plot
    fig, ax = plt.subplots(figsize=(8, 8), dpi=300, subplot_kw={'projection': ccrs.PlateCarree()})
    style_cartopy_map(ax, extent=[x_min, x_max, y_min, y_max])
    ax.set_title(f"Trajectories of {len(sampled_ids)} particles out of {len(d_id)}")

    colors = plt.cm.viridis(np.linspace(0, 1, len(sampled_ids)))

    for i, pid in enumerate(sampled_ids):
        particle_indices = [j for j, ids in enumerate(d_id) if pid in ids]
        if len(particle_indices) == 0:
            print(f"No data found for particle ID {pid}, skipping.")
            continue

        for particle_index in particle_indices:
            lon_traj = np.array(x[particle_index], dtype=np.float32)
            lat_traj = np.array(y[particle_index], dtype=np.float32)
            t_traj = np.array(t[particle_index], dtype=np.float32)

            sorted_idx = np.argsort(t_traj)
            lon_traj = lon_traj[sorted_idx]
            lat_traj = lat_traj[sorted_idx]
            t_traj = t_traj[sorted_idx]

            in_region = np.logical_and.reduce((
                lon_traj >= x_min,
                lon_traj <= x_max,
                lat_traj >= y_min,
                lat_traj <= y_max
            ))
            lon_traj = lon_traj[in_region]
            lat_traj = lat_traj[in_region]
            t_traj = t_traj[in_region]

            if len(lon_traj) < 2:
                print(f"All points for particle ID {pid} are outside the region, skipping.")
                continue

            ax.plot(lon_traj, lat_traj, '-', color=colors[i], lw=0.05, label=f'ID {pid}', zorder=2)
            ax.scatter(lon_traj[0], lat_traj[0], s=1, color='blue', zorder=3)
            ax.scatter(lon_traj[-1], lat_traj[-1], s=1, color='red', zorder=3)

    # Duration info for legend
    duration_days = (time_list[-1] - time_list[0])
    ax.plot([], [], ' ', label=f'Duration: {duration_days:.1f} days')

    if len(sampled_ids) < 10:
        ax.legend(fontsize=6, loc='lower left')

    if save:
        path = f"{output_directory}/manual_trajectories_custom.png"
        plt.savefig(path, dpi=300, bbox_inches="tight")
        print(f"Manual trajectory plot saved at: {path}")
    plt.close(fig)


def draw_trajectories_starting_near_location(x0, y0, save=True):
    """
    Plot trajectories of particles that START within ~60 km of (x0, y0),
    with color along each trajectory showing progression in time.
    """
    print(f"Plotting trajectories starting near ({x0}, {y0})...")
    delta_lon = 0.1
    delta_lat = 0.1

    start_lon_min = x0 - delta_lon
    start_lon_max = x0 + delta_lon
    start_lat_min = y0 - delta_lat
    start_lat_max = y0 + delta_lat

    selected_indices = []

    for i in range(len(x)):
        lon_start = x[i][0]
        lat_start = y[i][0]

        if (start_lon_min <= lon_start <= start_lon_max and
            start_lat_min <= lat_start <= start_lat_max):
            selected_indices.append(i)

    if not selected_indices:
        print(f"No particles start within ~30 km of point ({x0}, {y0}).")
        return

    # Plot setup
    fig, ax = plt.subplots(figsize=(9, 8), dpi=300, subplot_kw={'projection': ccrs.PlateCarree()})
    style_cartopy_map(ax, extent=[x_range[0], x_range[1], y_range[0], y_range[1]])
    ax.set_title(f"{len(selected_indices)} out of {len(d_id)} Particles Starting Near ({x0:.2f}, {y0:.2f})")

    cmap = plt.cm.plasma
    all_times = []

    # Gather all times for global normalization
    for idx in selected_indices:
        t_arr = np.array(t[idx], dtype=np.float32)
        all_times.extend(t_arr)
    all_times = np.array(all_times)
    norm = plt.Normalize(vmin=np.min(all_times), vmax=np.max(all_times))

    for idx in selected_indices:
        lon_traj = np.array(x[idx], dtype=np.float32)
        lat_traj = np.array(y[idx], dtype=np.float32)
        t_traj = np.array(t[idx], dtype=np.float32)

        sorted_idx = np.argsort(t_traj)
        lon_traj = lon_traj[sorted_idx]
        lat_traj = lat_traj[sorted_idx]
        t_traj = t_traj[sorted_idx]

        if len(lon_traj) < 2:
            continue

        points = np.array([lon_traj, lat_traj]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(segments, cmap=cmap, norm=norm, linewidth=0.1, zorder=2,
                            transform=ccrs.PlateCarree())
        lc.set_array(t_traj[:-1])  # color by time
        ax.add_collection(lc)

        # Start and end points
        ax.scatter(lon_traj[0], lat_traj[0], s=0.1, color='blue', zorder=3)
        ax.scatter(lon_traj[-1], lat_traj[-1], s=0.1, color='red', zorder=3)

    # Add center reference
    ax.scatter(x0, y0, color='black', s=20, zorder=4, label='Start Region Center')

    # Add colorbar for time (converted to readable date format)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    cbar = plt.colorbar(sm, ax=ax, orientation='vertical', pad=0.02)
    
    # Generate readable ticks from numeric time values
    tick_locs = np.linspace(norm.vmin, norm.vmax, 5)
    tick_labels = [numdate(tick).strftime('%d/%m %H:%M') for tick in tick_locs]
    
    cbar.set_ticks(tick_locs)
    cbar.set_ticklabels(tick_labels)
    cbar.set_label('Time (dd/mm HH:MM)', fontsize=8)

    if save:
        path = f'{output_directory}/trajectories_starting_near_{x0}_{y0}.png'
        #path = f"/southern/shaipollak/trajectories/archive/png_shai_archive/trajectories_starting_near_{x0}_{y0}_by_time.png"
        plt.savefig(path, dpi=300, bbox_inches="tight")
        print(f"Time-colored trajectory plot saved at: {path}")
    plt.close(fig)
    print(f"Done plotting trajectories starting near ({x0}, {y0})...")


def draw_starting_points(n):
    print("Plotting Starting Points...")
    x_start = np.array(x, dtype=np.float32)[::files_n]
    y_start = np.array(y, dtype=np.float32)[::files_n]

    # Filter out points where (x, y) == (0, 0)
    mask = ~((x_start == 0) & (y_start == 0))
    x_start = x_start[mask]
    y_start = y_start[mask]

    total = x_start.shape[0]
    if total > n:
        idx = np.random.choice(total, n, replace=False)
        x_start = x_start[idx]
        y_start = y_start[idx]

    fig, ax = plt.subplots(figsize=(6, 6), dpi=300, subplot_kw={'projection': ccrs.PlateCarree()})
    ax.scatter(x_start, y_start, s=0.5, color='blue', zorder=3, label="Starting Points")
    geo_map(ax)
    ax.set_title(f"Starting Points of {n} out of {len(d_id)} Particles")
    ax.legend()
    plt.savefig(f"{output_directory}/starting_points.png", bbox_inches="tight")
    plt.close()
    print("Done plotting starting points.")


def draw_ending_points(n):
    print("Plotting Ending Points...")
    x_end = np.array(x, dtype=np.float32)[files_n - 1::files_n]
    y_end = np.array(y, dtype=np.float32)[files_n - 1::files_n]

    # Filter out points where (x, y) == (0, 0)
    mask = ~((x_end == 0) & (y_end == 0))
    x_end = x_end[mask]
    y_end = y_end[mask]

    total = x_end.shape[0]
    if total > n:
        idx = np.random.choice(total, n, replace=False)
        x_end = x_end[idx]
        y_end = y_end[idx]

    fig, ax = plt.subplots(figsize=(6, 6), dpi=300, subplot_kw={'projection': ccrs.PlateCarree()})
    ax.scatter(x_end, y_end, s=0.5, color='red', zorder=3, label="Ending Points")
    geo_map(ax)
    ax.set_title(f"Ending Points of {n} out of {len(d_id)} Particles")
    ax.legend()
    plt.savefig(f"{output_directory}/ending_points.png", bbox_inches="tight")
    plt.close()
    print("Done plotting ending points.")


def draw_backward_trajectories_near_location(x0, y0, target_datetime, radius_deg=0.1, save=True):
    """
    Plot backward trajectories of particles that passed near (x0, y0) at target_datetime.
    Color along trajectory shows time progression *backward* from target_datetime to start.
    """
    # Convert datetime to numeric time format
    target_time_num = datenum(target_datetime)
    
    # Define radius of interest
    start_lon_min = x0 - radius_deg
    start_lon_max = x0 + radius_deg
    start_lat_min = y0 - radius_deg
    start_lat_max = y0 + radius_deg

    selected_indices = []

    for i in range(len(x)):
        lon_arr = np.array(x[i], dtype=np.float32)
        lat_arr = np.array(y[i], dtype=np.float32)
        #t_arr = np.array(t[i], dtype=np.float32)

        # Find indices where time is close to target
        #print(f"t_check: {t_arr}, target_time_num: {target_time_num}")
        # Sanitize t_arr: mask invalid entries (e.g., NaNs, infs, max int)
        
        t_arr = np.array(t[i], dtype=np.float32)
        t_arr = t_arr[np.isfinite(t_arr) & (t_arr < 1e7)]
        t_arr = t_arr / (24 * 60 * 60) + start_time  # Convert seconds to days, then align with datenum 

        if t_arr.size == 0:
            continue  # skip if array is now empty after cleaning

        time_diff = np.abs(t_arr - target_time_num)
        #print(f"Min time diff: {np.min(time_diff):.6f}, Max time diff: {np.max(time_diff):.6f}")

        time_match = time_diff < 0.05
        
        if not np.any(time_match):
            continue

        #print(f"Found a time match!")

        # Check if any of those points fall in the target region
        in_region = (
            (lon_arr >= start_lon_min) & (lon_arr <= start_lon_max) &
            (lat_arr >= start_lat_min) & (lat_arr <= start_lat_max)
        ) & time_match

        if np.any(in_region):
            print(f"Found particle {i} near ({x0:.2f}, {y0:.2f}) at {target_datetime.strftime('%d/%m %H:%M')}.")
            selected_indices.append(i)

    if not selected_indices:
        print(f"No particles found near ({x0:.2f}, {y0:.2f}) at {target_datetime}.")
        return

    # Plot
    fig, ax = plt.subplots(figsize=(9, 8), dpi=300, subplot_kw={'projection': ccrs.PlateCarree()})
    style_cartopy_map(ax, extent=[x_range[0], x_range[1], y_range[0], y_range[1]])
    ax.set_title(f"{len(selected_indices)} Particles Passed Near ({x0:.2f}, {y0:.2f}) at {target_datetime.strftime('%d/%m %H:%M')}")

    cmap = plt.cm.cividis
    for idx in selected_indices:
        lon_traj = np.array(x[idx], dtype=np.float32)
        lat_traj = np.array(y[idx], dtype=np.float32)
        t_traj = np.array(t[idx], dtype=np.float32)

        sorted_idx = np.argsort(t_traj)
        lon_traj = lon_traj[sorted_idx]
        lat_traj = lat_traj[sorted_idx]
        t_traj = t_traj[sorted_idx]

        # Remove points outside the plotting region
        in_bounds = (
            (lon_traj >= x_range[0]) & (lon_traj <= x_range[1]) &
            (lat_traj >= y_range[0]) & (lat_traj <= y_range[1])
        )
        lon_traj = lon_traj[in_bounds]
        lat_traj = lat_traj[in_bounds]
        t_traj = t_traj[in_bounds]

        # Mask for time before target
        mask = t_traj <= target_time_num
        lon_traj = lon_traj[mask]
        lat_traj = lat_traj[mask]
        t_traj = t_traj[mask]

        if len(lon_traj) < 2:
            continue

        # Plot segment with color by reversed time
        points = np.array([lon_traj, lat_traj]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        norm = plt.Normalize(vmin=np.min(t_traj), vmax=np.max(t_traj))
        lc = LineCollection(segments, cmap=cmap, norm=norm, linewidth=0.1, zorder=2,
                            transform=ccrs.PlateCarree())
        lc.set_array(t_traj)
        ax.add_collection(lc)

        # Start = current time, End = start
        ax.scatter(lon_traj[0], lat_traj[0], s=0.1, color='red', zorder=2)  # position at target time
        ax.scatter(lon_traj[-1], lat_traj[-1], s=0.1, color='blue', zorder=2)  # origin

    ax.scatter(x0, y0, color='black', s=20, zorder=4, label='Query Center')
    '''
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, orientation='vertical', pad=0.02)
    tick_locs = np.linspace(norm.vmin, norm.vmax, 5)
    tick_labels = []
    for tick in tick_locs:
        try:
            tick_labels.append(numdate(tick).strftime('%d/%m %H:%M'))
        except (OverflowError, ValueError):
            tick_labels.append("Invalid")
    cbar.set_ticks(tick_locs)
    cbar.set_ticklabels(tick_labels)
    cbar.set_label('Time (dd/mm HH:MM)', fontsize=8)
    '''
    if save:
        fname = f"{output_directory}/backward_trajectories_near_{x0:.2f}_{y0:.2f}_at_{target_datetime.strftime('%Y%m%d_%H%M')}.png"
        plt.savefig(fname, dpi=300, bbox_inches="tight")
        print(f"Backward trajectory plot saved at: {fname}")
    plt.close(fig)
# ─────────────────────────────────────────────────────────────────────────────

# Run plots
#draw_manual_trajectories(particle_ids=args.particle_ids if args.particle_ids else debug_particle_ids, n=10, save=True)
particles = 5000

#draw_manual_trajectories(n=particles, save=True)

draw_starting_points(particles)
draw_ending_points(particles)
draw_manual_trajectories(n=particles, save=True)

#draw_ending_points(particles)
#draw_trajectories_starting_near_location(33.5, 34.5, save=True)
#draw_backward_trajectories_near_location(x0, y0, dt.strptime('11/02/2018 01:00', '%d/%m/%Y %H:%M'), save=True)
