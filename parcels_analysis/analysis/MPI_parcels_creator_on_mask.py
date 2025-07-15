print("----------RUNNING MPI_parcels_creator_on_mask----------")

#TO RUN THIS CODE, USE THE FOLLOWING COMMAND, CHANGE between 300m and 3km, winter and summer, cut_n and dt
# screen -S MPI #Create a new screen session
# mpirun -np 4 python MPI_parcels_creator_on_mask.py --season summer --resolution 300m --cut_n 1 --dt 60 --dt_save 1
# screen -r MPI #Reattach to the screen session

#-------------------------------------IMPORTS AND WARNINGS---------------------------------------

# RUN THE FOLLOWING COMMA
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from parcels import FieldSet, ParticleSet, JITParticle, AdvectionRK4, StatusCode
import numpy as np
import netCDF4 as nc
from datetime import timedelta, datetime
import os
from mpi4py import MPI
import argparse

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#------------------------------------INITIALIZE MPI--------------------------------------------

# MPI initialization
comm = MPI.COMM_WORLD
size = comm.Get_size()  # Total number of processors
rank = comm.Get_rank()  # Rank of this processor
print(f"Running on {size} MPI processes.")

#------------------------------------GRID AND DATA LOCATIONS------------------------------------

uv_grid_dic = {
    '300m': '/indian/vickyverma/EMedCrocoC/INPUT/EMed300m_grd.nc',
    '3km': '/indian/vickyverma/EMedCrocoA/INPUT/EMed3km_grd.nc'
}


uv_data_dic = {
    'winter': {
        '300m': '/indian/vickyverma/EMedCrocoC/WINTER/his/zslices/depth2m/z_EMed300m_his.*.nc',   
        '3km': '/atlantic3/vickyverma/data/EMedCrocoA/WINTER/his/zslices/depth2m/z_EMed3km_his.*.nc'
    }
    ,
    'summer': {
        '300m': '/indian/vickyverma/EMedCrocoC/SUMMER/his/zslices/depth2m/z_EMed300m_his.*.nc',
        '3km': '/atlantic3/vickyverma/data/EMedCrocoA/SUMMER/his/zslices/depth2m/z_EMed3km_his.*.nc'
    }
}

time_line_dic = {
    'winter': '29/01/2018 01:00',
    'summer': ''
}

indices_dic = {
    '3km': {'lon': range(681), 'lat': range(451)},
    '300m': {'lon': range(1599), 'lat': range(2173)}
}

# Generate timestamp if no filename is provided
automated_zarr_dir = f"/southern/shaipollak/parcels_analysis/data/{timestamp}"
automated_zarr_path = os.path.join(automated_zarr_dir, f"{timestamp}.zarr")
automated_zarr_name = f"{timestamp}.zarr"


#For DEBUG:
#uv_grid = '/indian/vickyverma/EMedCrocoC/INPUT/EMed300m_grd.nc'
#uv_data = '/southern/shaipollak/markov-pushforward/300m/data/rvort_winter/z_EMed300m_his.*.nc'
#uv_data_zero = '/southern/shaipollak/markov-pushforward/300m/data/rvort_winter/z_EMed300m_his.00000.nc'

#------------------------------------Config-------------------------------------------

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Run MPI Parcels simulation.")
parser.add_argument("--season", type=str, default='winter', help="Season for the simulation (e.g., 'winter', 'summer')")
parser.add_argument("--resolution", type=str, default='300m', help="Resolution of the grid (e.g., '300m', 3km)")
parser.add_argument("--output", type=str, default=None, help="Path to output Zarr file")
parser.add_argument("--cut_n", type=int, default=5, help="Cutting factor for the grid")
#parser.add_argument("--nend", type=int, default=0, help="End index for particle generation")
parser.add_argument("--ncycle", type=int, help="Simulation time in hours")
parser.add_argument("--dt", type=int, default=60, help="Time step in seconds (default is 30 seconds)")
parser.add_argument("--dt_save", type=int, default=1, help="Trajectory time step to save in hours")
args = parser.parse_args()

season = args.season

resolution = args.resolution
cut_n = args.cut_n
output_zarr_path = args.output if args.output else automated_zarr_path  # Use provided output path or default to automated name
ncycle = args.ncycle if args.output else 900  # Simulation time in hours, can be adjusted as needed, 900 hours is 37.5 days
parallel = True  # Set to True for parallel execution
num_cores = size  # Number of cores to use, set by MPI
dt = args.dt  # Time step in hours, can be adjusted as needed
dt_save = timedelta(hours=args.dt_save) if args.dt_save else timedelta(hours=60*60)  # Time step to save in hours, default is 1 hour


# Configure for later txt file creation
config_params = {
        "timestamp": timestamp,
        "started at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "output_directory": output_zarr_path,
        "output_zarr_file": output_zarr_path,
        "parallel - MPI is used?": parallel,
        "num_cores - number of cores running the proccess": num_cores,
        "cut_n - divide the total amount of particles in factor of": cut_n*cut_n,
        "ncycle - simulation time in hours": ncycle,
        "season - winter or summer": season,
        "resolution - 300m or 3km": resolution,
        "start time - start time of the simulation": time_line_dic[season],
        "uv_grid_dic - grid file location": uv_grid_dic[resolution],
        "uv_data_dic - data file location": uv_data_dic[season][resolution],
        "dt - time resolution of sampling (in seconds)": dt,
        "dt_save - time resolution of saving the trajectory (in hours)": dt_save,
        "number of particles": 0,  # Will be updated later
        "Success": "False"  # Will be updated later
    }

#----------------------------------DEBUG PRINTS--------------------------------------------

print(f"Running with the following parameters:")
for key, value in config_params.items():
    print(f"{key}: {value}")

#-----------------------------------GRID LOADING------------------------------------------

# Loading grid file
if resolution in uv_grid_dic:
    uv_grd = uv_grid_dic[resolution]
else:
    raise ValueError(f"Invalid resolution '{resolution}'. Choose from {list(uv_grid_dic.keys())}.")

ds_uv_grd = nc.Dataset(uv_grd, 'r')
grd_mask = ds_uv_grd['mask_psi'][:]
lon_mask = ds_uv_grd['lon_psi'][:]
lat_mask = ds_uv_grd['lat_psi'][:]
land_mask = (grd_mask == 1)

ds_uv_grd.close()

# Where is the interpolation happening? in the parcles itself! the output of fieldset is velolcity vector field in the f points (psi points)

# Loading data for u and v
if season in uv_data_dic and resolution in uv_data_dic[season]:
    uv_data = uv_data_dic[season][resolution]
    uv_data_zero = uv_data_dic[season][resolution].replace('*', '00000')  # Replace wildcard with a specific file
else:   
    raise ValueError(f"Invalid combination of season '{season}' and resolution '{resolution}'. "
                     f"Choose from {list(uv_data_dic.keys())} seasons and {list(uv_data_dic['winter'].keys())} resolutions.")


#ds_uv_data = nc.Dataset(uv_data_zero, 'r')
#ds_uv_data.close()

print("Done loading grid and data files.")

#-----------------------------------FIELDSET CREATION---------------------------------------------------------

filenames = {'U': {'lon': uv_grd, 'lat': uv_grd, 'data': uv_data},
             'V': {'lon': uv_grd, 'lat': uv_grd, 'data': uv_data}}

variables = {'U': 'u', 'V': 'v'}
dimensions = {'U': {'lon': 'lon_psi', 'lat': 'lat_psi', 'time': 'time'},
              'V': {'lon': 'lon_psi', 'lat': 'lat_psi', 'time': 'time'}}


# DEBUG INDICES (300m resolution)
'''
indices = {'U': {'lon': range(1599), 'lat': range(2173)},
           'V': {'lon': range(1599), 'lat': range(2173)}}
'''

indices = {'U': {'lon': indices_dic[resolution]['lon'], 'lat': indices_dic[resolution]['lat']},
           'V': {'lon': indices_dic[resolution]['lon'], 'lat': indices_dic[resolution]['lat']}}



fieldset = FieldSet.from_nemo(filenames, variables, dimensions, indices, allow_time_extrapolation=False, chunksize='auto')

print("Done creating fieldset.")

#-------------------------- Create particles - Where the particles are initially located?---------------------------

#Filter the land mask to get only water points
lon_water = np.where(land_mask, lon_mask, np.nan)
lat_water = np.where(land_mask, lat_mask, np.nan)


# Cut the number of particles based on cut_n, reducing by {cut_n}^2
cut_lon_water = lon_water[::cut_n, ::cut_n]
cut_lat_water = lat_water[::cut_n, ::cut_n]

[lons, lats] = [cut_lon_water, cut_lat_water]

# Filter out NaN values from lons and lats
valid_mask = ~np.isnan(cut_lon_water) & ~np.isnan(cut_lat_water)
valid_lons = cut_lon_water[valid_mask]
valid_lats = cut_lat_water[valid_mask]
num_of_particles = valid_lons.size
config_params["number of particles"] = num_of_particles

print(f"Valid particle locations: {valid_lons.shape}")
print(f"Total number of particles: {num_of_particles}")

print("Done creating particles.")

# ---------------------------------------- RUN THE SIMULATION ----------------------------------------
# Create ParticleSet
pset = ParticleSet(fieldset, JITParticle, lon=valid_lons, lat=valid_lats)

# Extract directory name from the given path (custom_name)
custom_name = os.path.basename(os.path.dirname(output_zarr_path))  # Gets "custom_name" from "../data/custom_name/****.zarr"
os.makedirs(os.path.dirname(output_zarr_path), exist_ok=True)


#Create Configuration file
config_file_path = os.path.join(automated_zarr_dir, "config.txt")
with open(config_file_path, "w") as f:
        for key, value in config_params.items():
            f.write(f"{key}: {value}\n")

print(f"Configuration file saved to {config_file_path}")

# Set simulation parameters
ncycle = ncycle  # Simulation time in hours
dt_value = dt # Simulation Time step (seconds)
  # Time step to save in hours


# Create output file
output_file = pset.ParticleFile(name=output_zarr_path, outputdt=dt_save)

# Kernel for out-of-bounds particles
def CheckOutOfBounds(particle, fieldset, time):
    if particle.state == StatusCode.ErrorOutOfBounds:
        particle.delete()

# Execute the particle simulation
print(f"Running particle simulation for {ncycle} hours with dt={dt_value}")
pset.execute([AdvectionRK4, CheckOutOfBounds], runtime=timedelta(hours=ncycle), dt=dt_value, output_file=output_file)


print("DONE running the simulation.")
print(f"Progress to convert the Zarr output to NetCDF format.")
config_params["Success"] = "True"  # Update success status
config_params["Finished at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# Save the updated configuration file
with open(config_file_path, "w") as f:
    for key, value in config_params.items():
        f.write(f"{key}: {value}\n")