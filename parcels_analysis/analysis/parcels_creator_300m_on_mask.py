# Imports
from parcels import FieldSet, ParticleSet, JITParticle,ScipyParticle, AdvectionRK4, StatusCode
from parcels import DiffusionUniformKh, Kernel



import numpy as np
import netCDF4 as nc
from datetime import timedelta 
import matplotlib.pyplot as plt
import math
import os

# Loading grid file
uv_grd    = '/indian/vickyverma/EMedCrocoC/INPUT/EMed300m_grd.nc'
ds_uv_grd = nc.Dataset(uv_grd, 'r')
grd_mask = ds_uv_grd['mask_psi'][:]
lon_mask = ds_uv_grd['lon_psi'][:]
lat_mask = ds_uv_grd['lat_psi'][:]


# Loading all the data for u and v, basis for the particle movement (* - joker for all matching files, make sure it's sorted and limited to what you want)
uv_data  ='/southern/shaipollak/markov-pushforward/300m/data/rvort_winter/z_EMed300m_his.*.nc'
uv_data_zero = '/southern/shaipollak/markov-pushforward/300m/data/rvort_winter/z_EMed300m_his.00000.nc'
ds_uv_data = nc.Dataset(uv_data_zero, 'r')

# Definitions for using FieldSet
filenames = {'U': {'lon': uv_grd, 'lat': uv_grd, 'data': uv_data},
        'V': {'lon': uv_grd, 'lat': uv_grd, 'data': uv_data}}

variables = {'U': 'u', 'V': 'v'}

dimensions = {'U': {'lon': 'lon_psi', 'lat': 'lat_psi', 'time': 'time'},
              'V': {'lon': 'lon_psi', 'lat': 'lat_psi', 'time': 'time'}}

# Select indices based on the size of the lon,lat pixel range - should be the smallest value for each parameter
indices = {'U': {'lon': range(1599), 'lat': range(2173)},
              'V': {'lon': range(1599), 'lat': range(2173)}}

fieldset = FieldSet.from_nemo(filenames, variables, dimensions, indices, 
       allow_time_extrapolation=False)


# Parcles to make and where, creates parcels over the water area and cut the number based on cut_n to control the amount of particles
grd_mask = ds_uv_grd['mask_psi'][:]
lon_mask = ds_uv_grd['lon_psi'][:]
lat_mask = ds_uv_grd['lat_psi'][:]

land_mask = (grd_mask == 1)

lon_water = np.where(land_mask, lon_mask, np.nan)
lat_water = np.where(land_mask, lat_mask, np.nan)

# print(np.count_nonzero(lat_water), lon_water.size)

cut_n = 10
cut_lon_water = lon_water[::cut_n, ::cut_n]
cut_lat_water = lat_water[::cut_n, ::cut_n]
print(cut_lon_water.shape, cut_lon_water.size)

[lons, lats] = [cut_lon_water, cut_lat_water]


# Calculate cycles to be used for runtime
nstart = 0
# nend = 1252
# ncycle needs to be the difference between the values and the general time difference (ie 4 hours) minus 1 so there won't be an out of time error
# ncycle = nend-nstart+3
ncycle = 75
# print('ncycle: ', ncycle)

# Setting the ParticleSet
pset=ParticleSet(fieldset, JITParticle, lon=lons, lat=lats)


# Creating an additional kernal for deleting particles that are out of bounds, else the run crashes
def CheckOutOfBounds(particle, fieldset, time):
    if particle.state == StatusCode.ErrorOutOfBounds:
        particle.delete()


output_zarr_name =  '../data/test72.zarr'
# Setting the output_file and running the pset with the two kernels - AdvectionRK4 and the particles OOB remover
output_file = pset.ParticleFile(name=output_zarr_name, outputdt=timedelta(hours=2))

pset.execute([AdvectionRK4, CheckOutOfBounds], runtime=timedelta(hours=ncycle), 
             dt=timedelta(hours=2), output_file=output_file, 
             )
# pset.execute([AdvectionRK4, Sample_land, CheckOutOfBounds], runtime=timedelta(hours=ncycle),
#              dt=timedelta(seconds=30), output_file=output_file,
#              )
