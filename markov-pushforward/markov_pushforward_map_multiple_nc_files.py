import sys
import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
import cmocean
import cartopy.feature as cfeature
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import argparse
from datetime import datetime as dt, timedelta as td
import os
import imageio
import numpy as np
import os
import pickle
import glob

# =========================================================== Config ===========================================================
# Debug File (for testing)

#THE GOOD ONE (3000000 particles)
#debug = '/southern/shaipollak/parcels_analysis/data/20250329_235344/20250329_235344.nc' THE GOOD ONE (3000000 particles)


#100000 particles
debug_file_name = '20250620_111453'
debug_sampling_rate = 1 #hours
spatial_dis = 120 #Max of bins in the main axis (latitude)

debug = f'/southern/shaipollak/parcels_analysis/data/{debug_file_name}/{debug_file_name}.nc'

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Generate animation from trajectory images.")
parser.add_argument("--parcels_file_timestamp", type=str, help="Parcels file timestamp.")
parser.add_argument("--x0", type=float, help="Initial longitude for the starting position (optional).")
parser.add_argument("--y0", type=float, help="Initial latitude for the starting position (optional).")
parser.add_argument("--show_bins", type=bool, help="Show bins their id in the output images.")
parser.add_argument("--sampling_rate", type=int, help="Sampling rate for the particles trajectories in hours (saved positions, hourly)")
args = parser.parse_args()


sampling_rate = args.sampling_rate if args.sampling_rate else debug_sampling_rate
parcels_file_path_args = f'/southern/shaipollak/parcels_analysis/data/{args.parcels_file_timestamp}/{args.parcels_file_timestamp}.nc'


# Extract the file name without extension
parcels_file_path = parcels_file_path_args if args.parcels_file_timestamp else debug
folder_path = os.path.dirname(parcels_file_path)
file_name = os.path.basename(parcels_file_path).replace('.nc', '')


#=========================================================== Time Functions ===========================================================
# Convert a stripped date to a datenum using datetime as dt
def datenum(stripped_date):
    d = dt.strptime('29/01/2018 01:00','%d/%m/%Y %H:%M')
    days = 365 + d.toordinal()
    hours = (stripped_date - dt.fromordinal(d.toordinal())).total_seconds()/(24*60*60)
    return days + hours

# Reverse function for validation
def numdate(num):
    days = num - 365
    whole_days = int(days)
    fractional_days = days - whole_days
    date = dt.fromordinal(whole_days)
    time_delta = td(seconds=fractional_days * 24 * 60 * 60)
    return date + time_delta


# =========================================================== Load NetCDF Data ===========================================================
# --- Load files ---
print(f"Looking for files: {file_name}_proc*.nc")
nc_file_pattern = os.path.join(folder_path, f"{file_name}_proc*.nc")
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

        num_particles, num_time_steps = lon_par.shape
        d_id_of_file = np.repeat(id_par, num_time_steps)
        x_of_file = lon_par.flatten()
        y_of_file  = lat_par.flatten()
        t_of_file  = time_par.flatten()
        
        x.append(np.array(x_of_file, dtype=np.float32))
        y.append(np.array(y_of_file, dtype=np.float32))
        t.append(np.array(t_of_file, dtype=np.float32))
        d_id.append(np.array(d_id_of_file, dtype=np.int32))

# Concatenate all data from all files
x = np.concatenate(x)
y = np.concatenate(y)
t = np.concatenate(t)
d_id = np.concatenate(d_id)


# Convert d_id, x, y, t to 1D arrays
d_id = d_id.flatten()
x = x.flatten()
y = y.flatten()
t = t.flatten()


print("Data loaded successfully.")

assert d_id.shape == x.shape == y.shape == t.shape # True, "All arrays must have the same shape!"

# ----------------------------------- Parameters for Markov Prediction -------------------------------------


show_bins = args.show_bins if args.show_bins else False
show_bins = False
lon, lat = [31, 37], [31, 37]


#Set Manually the contamination point if provided, otherwise use defaults:
y_contamination = None
x_contamination = None

# Set initial position (x0, y0) if provided, otherwise use defaults
x0, y0 = args.x0 if args.x0 else x_contamination, args.y0 if args.y0 else y_contamination

sys.path.append('/southern/shaipollak')
from pygtm.physical import physical_space
from pygtm.matrix import matrix_space
from pygtm.dataset3 import trajectory as trajectory2

d = physical_space(lon, lat, spatial_dis) #Create Physical space
#trajectory_of_particles_data = trajectory2(x, y, t, d_id)

if x0 is not None and y0 is not None:
    el_id = d.find_element(x0, y0)

#=========================================================== Functions ===========================================================

def get_particle_ids_in_bin(bin_index):
    """
    Return a list of particle IDs whose starting point is inside the given bin index,
    and save them to a text file.
    """
    '''
    trajectory_of_particles_data.create_segments(1)
    tm = matrix_space(d)
    tm.fill_transition_matrix(trajectory_of_particles_data)

    if bin_index >= len(tm.B):
        print(f"Bin index {bin_index} is out of range.")
        return []

    particle_indices = tm.B[bin_index]
    particle_ids = d_id[particle_indices].tolist()

    output_dir = f"/southern/shaipollak/parcels_analysis/data/{file_name}"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"Particle list in bin {bin_index}.txt")

    with open(output_path, "w") as f:
        for pid in particle_ids:
            f.write(f"{pid}\n")

    print(f"Saved particle list for bin {bin_index} to {output_path}")
    return particle_ids
    '''
    
# Save the transition matrix and relevant fields from a matrix_space object
def save_matrix(matrix_obj, base_path, T):
    """
    Save the transition matrix and relevant fields from a matrix_space object.
    
    Args:
        matrix_obj: matrix_space instance
        base_path: directory to save matrix data
        T: timestep (used in filename)
    """
    os.makedirs(base_path, exist_ok=True)
    
    # Save transition matrix as .npy
    np.save(os.path.join(base_path, f'transition_matrix_T{T}_res={spatial_dis}.npy'), matrix_obj.P)
    
    # Save all relevant metadata
    extra_data = {
        'B': matrix_obj.B,
        'M': matrix_obj.M,
        'domain_bins': matrix_obj.domain.bins,
        'domain_id_og': matrix_obj.domain.id_og,
        'domain_lon': matrix_obj.domain.lon,
        'domain_lat': matrix_obj.domain.lat
    }
    with open(os.path.join(base_path, f'matrix_metadata_T{T}_res={spatial_dis}.pkl'), 'wb') as f:
        pickle.dump(extra_data, f)

    print(f"Transition matrix and metadata saved for T={T}")

# Function to load the transition matrix and metadata
def load_matrix(matrix_obj, base_path, T):
    """
    Load the transition matrix and related metadata into an existing matrix_space object.
    
    Args:
        matrix_obj: matrix_space instance to load data into
        base_path: directory where matrix data is saved
        T: timestep (used in filename)
    """
    matrix_file = os.path.join(base_path, f'transition_matrix_T{T}_res={spatial_dis}.npy')
    metadata_file = os.path.join(base_path, f'matrix_metadata_T{T}_res={spatial_dis}.pkl')

    if not os.path.exists(matrix_file) or not os.path.exists(metadata_file):
        raise FileNotFoundError(f"Missing matrix or metadata for T={T} and res={spatial_dis} in {base_path}")

    matrix_obj.P = np.load(matrix_file)

    #Open the metadata file and load the extra data the the matrix object
    with open(metadata_file, 'rb') as f:
        extra_data = pickle.load(f)
        matrix_obj.B = extra_data['B']
        matrix_obj.M = extra_data['M']
        matrix_obj.domain.bins = extra_data['domain_bins']
        matrix_obj.domain.id_og = extra_data['domain_id_og']
        matrix_obj.domain.lon = extra_data['domain_lon']
        matrix_obj.domain.lat = extra_data['domain_lat']
        matrix_obj.N = len(matrix_obj.P)

    print(f"Transition matrix and metadata loaded for T={T} and res={spatial_dis} from {base_path}")

# Function to format the geographical map
def geo_map(ax):
    ax.set_xticks([31, 32, 33, 34, 35, 36, 37], crs=ccrs.PlateCarree())
    ax.set_yticks([31, 32, 33, 34, 35, 36, 37], crs=ccrs.PlateCarree())
    ax.xaxis.set_major_formatter(LongitudeFormatter())
    ax.yaxis.set_major_formatter(LatitudeFormatter())
    ax.tick_params(axis='x', labelsize=7)
    ax.tick_params(axis='y', labelsize=7)
    ax.add_feature(cfeature.LAND, facecolor='silver', zorder=1)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.25, zorder=1)

# =========================================================== Create Matrix Functions ===========================================================
# Function to run Markov prediction
def run_markov_prediction(T_start, T_end, draw_prediction=True, load_matrix_auto=True):
    """
    Run Markov prediction for a range of T values.
    """
    for T in range(T_start, T_end + 1):
        # Load or save the trasition matrix into:
        matrix_dir = f"/southern/shaipollak/parcels_analysis/data/{file_name}/matrix_data"

        tm = matrix_space(d) #Create the matrix space
        
        #Try to find if the Matrix already exists, load it if it does and do not recalculate the matrix
        try:
            print(f"For file {file_name} and T={T}, Trying to load existing matrix...")
            load_matrix(tm, matrix_dir, T)


        ## If the matrix does not exist, create it and save it    
        except FileNotFoundError:
            print("Matrix not found. This is a new matrix! creating it...")
            trajectory_of_particles_data = trajectory2(x, y, t, d_id)
            trajectory_of_particles_data.create_segments(T, mean_hours=sampling_rate)  # Create segments with T days apart
            tm.fill_transition_matrix(trajectory_of_particles_data)
            print("Transition matrix created.")
        
            # --- Augment transition matrix with external absorbing state to make it probability-conserving ---
            # Assume particles enter the domain uniformly (if nothing comes in)
            tm.fi = np.ones(tm.N) / tm.N

            # Expand the matrix by 1 row and 1 column
            tm.P = np.pad(tm.P, ((0, 1), (0, 1)), 'constant', constant_values=0)

            # The new last row: transition from the external state into the domain
            tm.P[-1, :-1] = tm.fi

            # The new last column: transition from domain to external state (exiting)
            tm.P[:-1, -1] = tm.fo

            print("Saving it!")
            save_matrix(tm, matrix_dir, T)

        if not draw_prediction:
            print(f"Skipping image generation for T={T}.")
            continue

        # Determine the initial density and output directory
        if x0 is not None and y0 is not None:
            el_id = d.find_element(x0, y0)
            density = np.zeros(len(d.bins), dtype=np.float32)
            density[el_id] = 1
            density /= np.sum(density)
            
            # Ensure the density vector has the same length as the transition matrix (because of fi and fo)
            if tm.P.shape[0] == len(density) + 1:
                density = np.append(density, 0.0)

            output_directory = f"/southern/shaipollak/parcels_analysis/data/{file_name}/Markov_Analysis_Images_bin_{el_id}_T_{T}_res_{spatial_dis}"
            '''
            # Save particle IDs in the initial bin
            if T == T_start:
                try:
                    updated_index = np.where(d.id_og == el_id)[0][0]
                    particle_indices = tm.B[updated_index]
                    particle_ids = np.unique(d_id[particle_indices]).tolist()
                    output_path = os.path.join(f"/southern/shaipollak/parcels_analysis/data/{file_name}", f"Particle list in bin {el_id}.txt")
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    with open(output_path, "w") as f:
                        f.write(str(particle_ids))
                    print(f"Saved particle list for bin {el_id} to {output_path}")
                except IndexError:
                    print(f"Original bin {el_id} was removed during filtering — no particles.")
            '''
        else:
            
            density = np.ones(len(d.bins), dtype=np.float32) / len(d.bins)

            # Pad density if matrix has absorbing state
            if tm.P.shape[0] == len(density) + 1:
                density = np.append(density, 0.0)

            output_directory = f"/southern/shaipollak/parcels_analysis/data/{file_name}/Markov_Analysis_Images_Full_Grid_T_{T}_res_{spatial_dis}"

        os.makedirs(output_directory, exist_ok=True)

        # Generate probability density images
        month = 31 #in days
        duration = 31 / T
        for i in range(0, int(duration) + 1):
            print(f"Generating image {i} for T={T}...") 
            duration = i * T
            evolved_density = tm.push_forward(density, int(duration / T))
            
            fig = plt.figure(figsize=(4, 3), dpi=300)
            ax1 = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree(), aspect='equal')
            density_matrix = d.vector_to_matrix(evolved_density)
            density_matrix = np.nan_to_num(density_matrix, nan=0.0)

            #Dymamic colorbar
            vmin = np.min(density_matrix)
            vmax = np.max(density_matrix)

            # Avoid having vmin = vmax to prevent plotting errors
            if vmax - vmin < 1e-10:
                vmin, vmax = 0.0, 1e-5

            p1 = ax1.pcolormesh(d.vx, d.vy, density_matrix, vmin=vmin, vmax=vmax,
                        cmap="cmo.matter", alpha=1.0, transform=ccrs.PlateCarree())

            ax1.set_extent([31, 37, 31, 37], crs=ccrs.PlateCarree())
            geo_map(ax1)

            cb = fig.colorbar(p1, extend='both')
            cb.set_label(f'Probability (T={T}d, Day={duration})', size=7, labelpad=2)

            output_file = os.path.join(output_directory, f'density_pushforward_{i:02d}.png')

            #Draw the bins, their id and the probability inside it    
            if show_bins:
                # Show active bins as rectangles
                d.bins_contour(ax1, edgecolor="black", bin_id=range(len(d.bins)), projection=ccrs.PlateCarree())

                for bin_id in range(len(d.bins)):
                    prob = evolved_density[bin_id]
                    if prob > 0:
                        corner_indices = d.bins[bin_id]
                        corners = d.coords[corner_indices]
                        center_lon = np.mean(corners[:, 0])
                        center_lat = np.mean(corners[:, 1])

                        # Compute vertical size of the bin
                        lat_size = np.max(corners[:, 1]) - np.min(corners[:, 1])
                        dy = lat_size * 0.15  

                        # Bin ID (top line)
                        ax1.text(center_lon, center_lat + dy, str(bin_id), fontsize=1.0, ha='center', va='center',
                                transform=ccrs.PlateCarree(), zorder=10, color='green')

                        # Probability
                        ax1.text(center_lon, center_lat - dy, f'{prob:.3f}', fontsize=1, ha='center', va='center',
                                transform=ccrs.PlateCarree(), zorder=10, color='green')

            plt.savefig(output_file, dpi=600 if show_bins else 300)
            plt.close()

        print(f"Image generation complete for T={T}.")


        if not show_bins:
            # Create GIF animation
            files = sorted([file for file in os.listdir(output_directory) if file.endswith('.png')])
            images = [imageio.imread(os.path.join(output_directory, file)) for file in files]

            output_gif = os.path.join(output_directory, "prediction.gif")
            imageio.mimsave(output_gif, images, duration=1.6, loop=0)

            print(f"GIF animation created for T={T}.")

def create_random_matrices(rand=0.8, n_samples=50, T=1):
    """
    Generate N transition matrices using {rand}% of the particles randomly sampled.
    Each matrix is saved in a folder named '80percentMatrix' as matrix_1.npy, ..., matrix_N.npy.
    """
    if abs(rand) > 1 or rand <= 0:
        raise ValueError("rand must be a float between 0 and 1.")
    
    output_dir = f"/southern/shaipollak/parcels_analysis/data/{file_name}/matrix_data/random_matrices_T={T}_res_{spatial_dis}/"
    os.makedirs(output_dir, exist_ok=True)

    total_particles = len(np.unique(d_id))
    unique_ids = np.unique(d_id)

    for i in range(1, n_samples + 1):
        print(f"\n[{dt.now()}] Creating {rand*100}% sample matrix {i}/{n_samples}...")

        # Randomly select 80% of particle IDs
        selected_ids = np.random.choice(unique_ids, size=int(rand * total_particles), replace=False)

        # Filter arrays
        mask = np.isin(d_id, selected_ids)
        sampled_d_id = d_id[mask]
        sampled_x = x[mask]
        sampled_y = y[mask]
        sampled_t = t[mask]

        # Create trajectory object
        trajectory_data = trajectory2(sampled_x, sampled_y, sampled_t, sampled_d_id)
        trajectory_data.create_segments(T)

        # Fill matrix
        tm = matrix_space(d)
        tm.fill_transition_matrix(trajectory_data)

        # Add absorbing state to conserve probability
        tm.fi = np.ones(tm.N) / tm.N
        tm.P = np.pad(tm.P, ((0, 1), (0, 1)), 'constant', constant_values=0)
        tm.P[-1, :-1] = tm.fi
        tm.P[:-1, -1] = tm.fo

        # Save with custom name
        save_matrix(tm, output_dir, f"matrix_{i}")

    print(f"All {n_samples} transition matrices saved to {output_dir}")

# =========================================================== Run Markov Prediction ===========================================================

#run_markov_prediction(1, 1, draw_prediction=True)  # Set draw_prediction to False to skip image generation
create_random_matrices()

