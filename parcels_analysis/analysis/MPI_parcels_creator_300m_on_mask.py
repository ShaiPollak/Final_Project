print("----------RUNNING MPI_parcels_creator_300m_on_mask----------")


# RUN THE FOLLOWING COMMA
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Imports
from parcels import FieldSet, ParticleSet, JITParticle, AdvectionRK4, StatusCode
import numpy as np
import netCDF4 as nc
from datetime import timedelta, datetime
import os
from mpi4py import MPI
import argparse

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Run MPI Parcels simulation.")
parser.add_argument("--output", type=str, default=None, help="Path to output Zarr file")
parser.add_argument("--cut_n", type=int, default=1, help="Cutting factor for the grid")
#parser.add_argument("--nend", type=int, default=0, help="End index for particle generation")
#parser.add_argument("--ncycle", type=int, default=360, help="Simulation time in hours")
parser.add_argument("--dt", type=int, default=2, help="Time step in hours")
args = parser.parse_args()

# MPI initialization
comm = MPI.COMM_WORLD
size = comm.Get_size()  # Total number of processors
rank = comm.Get_rank()  # Rank of this processor
print(f"Running on {size} MPI processes.")

# Loading grid file
uv_grd = '/indian/vickyverma/EMedCrocoC/INPUT/EMed300m_grd.nc'
ds_uv_grd = nc.Dataset(uv_grd, 'r')
grd_mask = ds_uv_grd['mask_psi'][:]
lon_mask = ds_uv_grd['lon_psi'][:]
lat_mask = ds_uv_grd['lat_psi'][:]

# Where is the interpolation happening?
# Loading data for u and v
uv_data = '/southern/shaipollak/markov-pushforward/300m/data/rvort_winter/z_EMed300m_his.*.nc'
uv_data_zero = '/southern/shaipollak/markov-pushforward/300m/data/rvort_winter/z_EMed300m_his.00000.nc'
ds_uv_data = nc.Dataset(uv_data_zero, 'r')

filenames = {'U': {'lon': uv_grd, 'lat': uv_grd, 'data': uv_data},
             'V': {'lon': uv_grd, 'lat': uv_grd, 'data': uv_data}}

variables = {'U': 'u', 'V': 'v'}
dimensions = {'U': {'lon': 'lon_psi', 'lat': 'lat_psi', 'time': 'time'},
              'V': {'lon': 'lon_psi', 'lat': 'lat_psi', 'time': 'time'}}

indices = {'U': {'lon': range(1599), 'lat': range(2173)},
           'V': {'lon': range(1599), 'lat': range(2173)}}

fieldset = FieldSet.from_nemo(filenames, variables, dimensions, indices, allow_time_extrapolation=False, chunksize='auto')

# Create particles
grd_mask = ds_uv_grd['mask_psi'][:]
lon_mask = ds_uv_grd['lon_psi'][:]
lat_mask = ds_uv_grd['lat_psi'][:]
land_mask = (grd_mask == 1)

lon_water = np.where(land_mask, lon_mask, np.nan)
lat_water = np.where(land_mask, lat_mask, np.nan)

cut_n = args.cut_n
cut_lon_water = lon_water[::cut_n, ::cut_n]
cut_lat_water = lat_water[::cut_n, ::cut_n]
print(f"Particles grid shape: {cut_lon_water.shape}, Total particles: {cut_lon_water.size}")

[lons, lats] = [cut_lon_water, cut_lat_water]

# Filter out NaN values from lons and lats
valid_mask = ~np.isnan(cut_lon_water) & ~np.isnan(cut_lat_water)
valid_lons = cut_lon_water[valid_mask]
valid_lats = cut_lat_water[valid_mask]
print(f"Valid particle locations: {valid_lons.shape}")

# Create ParticleSet
pset = ParticleSet(fieldset, JITParticle, lon=valid_lons, lat=valid_lats)

# Generate timestamp if no filename is provided
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_zarr_name = args.output if args.output else f"/southern/shaipollak/parcels_analysis/data/{timestamp}/{timestamp}.zarr"

# Extract directory name from the given path (custom_name)
custom_name = os.path.basename(os.path.dirname(output_zarr_name))  # Gets "custom_name" from "../data/custom_name/3mil-16.zarr"
print(f"Output Zarr Path: {output_zarr_name}")
print(f"Custom Name Extracted: {custom_name}")

# Ensure the output directory exists
os.makedirs(os.path.dirname(output_zarr_name), exist_ok=True)

# Set simulation parameters
ncycle = 360  # Simulation time in hours
dt_value = timedelta(hours=args.dt) if args.dt else timedelta(hours=2)# Time step

# Create output file
output_file = pset.ParticleFile(name=output_zarr_name, outputdt=dt_value)

# Kernel for out-of-bounds particles
def CheckOutOfBounds(particle, fieldset, time):
    if particle.state == StatusCode.ErrorOutOfBounds:
        particle.delete()

# Execute the particle simulation
print(f"Running particle simulation for {ncycle} hours with dt={dt_value}")
pset.execute([AdvectionRK4, CheckOutOfBounds], runtime=timedelta(hours=ncycle), dt=dt_value, output_file=output_file)

print("DONE")
