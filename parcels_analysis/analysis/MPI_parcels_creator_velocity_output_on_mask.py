print("----------RUNNING MPI_parcels_creator_velocity_output_on_mask----------")

#-------------------------------------IMPORTS AND WARNINGS---------------------------------------
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from parcels import FieldSet, ParticleSet, JITParticle, AdvectionRK4, StatusCode, Variable, ParticleFile
import numpy as np
import netCDF4 as nc
from datetime import timedelta, datetime
import os
from mpi4py import MPI
import argparse

#------------------------------------INITIALIZE MPI--------------------------------------------
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
print(f"Running on {size} MPI processes.")

#------------------------------------TIMESTAMP AND DEFAULTS------------------------------------
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
automated_zarr_dir = f"/southern/shaipollak/parcels_analysis/data/{timestamp}"
automated_zarr_path = os.path.join(automated_zarr_dir, f"{timestamp}.zarr")

#------------------------------------ARGUMENT PARSING------------------------------------------
parser = argparse.ArgumentParser(description="Run MPI Parcels simulation.")
parser.add_argument("--season", type=str, default='winter')
parser.add_argument("--resolution", type=str, default='300m')
parser.add_argument("--output", type=str, default=None)
parser.add_argument("--cut_n", type=int, default=5)
parser.add_argument("--ncycle", type=int, default=900)
parser.add_argument("--dt", type=int, default=60)
parser.add_argument("--dt_save", type=int, default=1)
args = parser.parse_args()

season = args.season
resolution = args.resolution
cut_n = args.cut_n
ncycle = args.ncycle
parallel = True
num_cores = size
dt = args.dt
dt_save = timedelta(hours=args.dt_save)
output_zarr_path = args.output if args.output else automated_zarr_path

#------------------------------------CONFIGURATION---------------------------------------------
uv_grid_dic = {
    '300m': '/indian/vickyverma/EMedCrocoC/INPUT/EMed300m_grd.nc',
    '3km': '/indian/vickyverma/EMedCrocoA/INPUT/EMed3km_grd.nc'
}

uv_data_dic = {
    'winter': {
        '300m': '/indian/vickyverma/EMedCrocoC/WINTER/his/zslices/depth2m/z_EMed300m_his.*.nc',
        '3km': '/atlantic3/vickyverma/data/EMedCrocoA/WINTER/his/zslices/depth2m/z_EMed3km_his.*.nc'
    },
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

config_params = {
    "TYPE" : "VELOCITY OUTPUT",
    "timestamp": timestamp,
    "started at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "output_directory": output_zarr_path,
    "output_zarr_file": output_zarr_path,
    "parallel - MPI is used?": parallel,
    "num_cores - number of cores running the process": num_cores,
    "cut_n - divide the total amount of particles in factor of": cut_n * cut_n,
    "ncycle - simulation time in hours": ncycle,
    "season - winter or summer": season,
    "resolution - 300m or 3km": resolution,
    "start time - start time of the simulation": time_line_dic[season],
    "uv_grid_dic - grid file location": uv_grid_dic[resolution],
    "uv_data_dic - data file location": uv_data_dic[season][resolution],
    "dt - time resolution of sampling (in seconds)": dt,
    "dt_save - time resolution of saving the trajectory (in hours)": dt_save,
    "number of particles": 0,
    "Success": "False"
}

#------------------------------------LOAD DATA------------------------------------------------
dataset = nc.Dataset(uv_grid_dic[resolution], 'r')
gr_mask = dataset['mask_psi'][:]
lon_mask = dataset['lon_psi'][:]
lat_mask = dataset['lat_psi'][:]
land_mask = (gr_mask == 1)
dataset.close()

uv_data = uv_data_dic[season][resolution]
filenames = {
    'U': {'lon': uv_grid_dic[resolution], 'lat': uv_grid_dic[resolution], 'data': uv_data},
    'V': {'lon': uv_grid_dic[resolution], 'lat': uv_grid_dic[resolution], 'data': uv_data}
}
variables = {'U': 'u', 'V': 'v'}
dimensions = {
    'U': {'lon': 'lon_psi', 'lat': 'lat_psi', 'time': 'time'},
    'V': {'lon': 'lon_psi', 'lat': 'lat_psi', 'time': 'time'}
}
indices = {
    'U': {'lon': indices_dic[resolution]['lon'], 'lat': indices_dic[resolution]['lat']},
    'V': {'lon': indices_dic[resolution]['lon'], 'lat': indices_dic[resolution]['lat']}
}

fieldset = FieldSet.from_nemo(filenames, variables, dimensions, indices, allow_time_extrapolation=False)

#------------------------------------PARTICLE INITIALIZATION----------------------------------
lon_water = np.where(land_mask, lon_mask, np.nan)
lat_water = np.where(land_mask, lat_mask, np.nan)
cut_lon_water = lon_water[::cut_n, ::cut_n]
cut_lat_water = lat_water[::cut_n, ::cut_n]
valid_mask = ~np.isnan(cut_lon_water) & ~np.isnan(cut_lat_water)
valid_lons = cut_lon_water[valid_mask]
valid_lats = cut_lat_water[valid_mask]
config_params["number of particles"] = valid_lons.size

#------------------------------------CUSTOM PARTICLE (INHERITS THE JITPARTICLE CLASS)----------
class MyParticle(JITParticle):
    u = Variable('u', dtype=np.float32)
    v = Variable('v', dtype=np.float32)

#------------------------------------KERNELS--------------------------------------------------
def SampleVelocity(particle, fieldset, time):
    particle.u = fieldset.U[time, particle.depth, particle.lat, particle.lon]
    particle.v = fieldset.V[time, particle.depth, particle.lat, particle.lon]

def CheckOutOfBounds(particle, fieldset, time):
    if particle.state == StatusCode.ErrorOutOfBounds:
        particle.delete()

#------------------------------------SIMULATION EXECUTION-------------------------------------
os.makedirs(os.path.dirname(output_zarr_path), exist_ok=True)
pset = ParticleSet(fieldset, MyParticle, lon=valid_lons, lat=valid_lats)
output_file = pset.ParticleFile(name=output_zarr_path, outputdt=dt_save)

config_file_path = os.path.join(os.path.dirname(output_zarr_path), "config.txt")
with open(config_file_path, "w") as f:
    for key, value in config_params.items():
        f.write(f"{key}: {value}\n")

print(f"Running simulation for {ncycle} hours on rank {rank}...")
pset.execute([AdvectionRK4, SampleVelocity, CheckOutOfBounds],
             runtime=timedelta(hours=ncycle),
             dt=dt,
             output_file=output_file)

config_params["Success"] = "True"
config_params["Finished at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
with open(config_file_path, "w") as f:
    for key, value in config_params.items():
        f.write(f"{key}: {value}\n")

print("DONE running the simulation.")
