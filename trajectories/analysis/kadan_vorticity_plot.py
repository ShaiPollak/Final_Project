
import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os
import imageio.v2 as imageio

#Grid file, needed soley for creating the limits for the plot
input_dir_grid = '/indian/vickyverma/EMedCrocoC/INPUT'
Med_grid = '%s/EMed300m_grd.nc' % input_dir_grid
ds_Med_grid = nc.Dataset(Med_grid)

f_mask = ds_Med_grid['mask_psi'][:]
f_lat = ds_Med_grid['lat_psi'][:]
f_lon = ds_Med_grid['lon_psi'][:]
# print(f_mask.shape) >>> (2173, 1599)

lon_d = f_lon.max() - f_lon.min()
lat_d = f_lat.max() - f_lat.min()
ratio = lat_d/lon_d # >>> 1.127058470020234 it's more for correting the relations view between the 6 lat and 5 lon - need to ask Roy abt it

def select_particle_location(file_indx, frame):

    #Loading the file and reading the dataset for the particle locations
    Med_particle='../data/EMed300m_his_prt_data_bc_winter.nc'
    ds_Med_particle = nc.Dataset(Med_particle)

    #Making the variables usable
    lon_par = ds_Med_particle['lon'][:]
    lat_par = ds_Med_particle['lat'][:]

    #Selected time, count starts from zero and jumps are by 1 based on a combo of file_indx and frame (0/1)
    selected_time=int(file_indx/2 +frame)

    #Creating lists for xy location from the selected time
    selected_time_part_lon = lon_par[:, selected_time]
    selected_time_part_lat = lat_par[:, selected_time]
    
    return (selected_time_part_lon, selected_time_part_lat)

def plot_vorticity(file_indx, frame,  par_lon, par_lat):
    #Plot setup
    plt.figure(figsize=(5, 5)) #Make the plot a bit bigger
    plt.imshow(dd_rvort_array, cmap='bwr', extent=[f_lon.min(), f_lon.max(), f_lat.max(), f_lat.min()]) #lat order is inversed as I need to flip the y axis
    plt.gca().invert_yaxis()
    plt.clim(-2, 2) # Setting the limit for the surface vorticity 
    colorbar = plt.colorbar()
    plt.gca().set_aspect(ratio) #Correct the image size based on the ratio of the lat lon total distance ratio

    plt.scatter(par_lon, par_lat, color='seagreen', s=1, alpha=0.8)
    
    #Time
    given_date = datetime.strptime('29/01/2018 01:00:00', '%d/%m/%Y %H:%M:%S')
    curr_date = given_date + timedelta(hours=int(file_indx + 2*frame))

    #Labelling
    plt.title(curr_date) 
    plt.ylabel('Latitude [°]')
    plt.xlabel('Longitude [°]')
    colorbar.ax.set_title(r'$\zeta/f$')
    
    #Save file
    output_file='png_kadan/prt_rvort_bc_%05d_%d.png'  % (file_indx, frame)
    print('OUTPUT:', output_file)
    plt.savefig(output_file, dpi=200)
    plt.close()

    #Plot display should be after the save file if needed


base_name = 'z_EMed300m_his_zvort'
data_dir = '/southern/shaipollak/markov-pushforward/300m/data/rvort_winter'  #data directory

# Loop through file numbers from 0 to limit
for file_indx in range(0, 836, 4):
    # Format the file name
    file_name = f"{base_name}.{file_indx:05d}.nc"
    
    # Create the full file path
    file_path = os.path.join(data_dir, file_name)
    
    for frame in range(0,2):
   	 # Open and modify the file
   	 ds_current_vort = nc.Dataset(file_path)

   	 rvort = ds_current_vort['rvort'][:]
   	 rvort_array = np.array(rvort)
   	 dd_rvort_array = rvort_array[(2*frame)] * f_mask #Mask Multipication
    
   	 par_lon, par_lat = select_particle_location(file_indx, frame)
   	 plot_vorticity(file_indx, frame,  par_lon, par_lat)


### Creating and saving the animation
# Directory containing PNG files
directory = '../data/png_kadan/'

# Get a list of PNG files in the directory
files = [file for file in os.listdir(directory) if file.endswith('.png')]
files.sort()  # Sort files in alphabetical order

# Create a list to store images
images = []

# Read images from files and append to the list
for file in files:
    img_path = os.path.join(directory, file)
    images.append(imageio.imread(img_path))

# Save the animation
output_file = '../data/animation.gif'
imageio.mimsave(output_file, images, duration=10, loop=0)  # Adjust duration as needed

print('Done')
