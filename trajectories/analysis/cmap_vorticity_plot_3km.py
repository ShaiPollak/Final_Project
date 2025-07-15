
import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os
import imageio.v2 as imageio

#Grid file, needed solely for creating the limits for the plot
Med_grid    = '/indian/vickyverma/EMedCrocoA/INPUT/EMed3km_grd.nc'
ds_Med_grid = nc.Dataset(Med_grid)

f_mask = ds_Med_grid['mask_psi'][:]
f_lat = ds_Med_grid['lat_psi'][:]
f_lon = ds_Med_grid['lon_psi'][:]
# print(f_mask.shape) >>> (2173, 1599)

# Defining Z for 100 different values
n = np.arange(0,200)
z = np.repeat(n,200)
z_sorted = np.sort(z)


def particle_loop_size():

    # Loading the file and reading the dataset for the particle locations
    Med_particle='/southern/kadanshinyuk/markov-pushforward/3km/data/EMed3km_his_prt_data_bc_winter_extd2.nc'
    ds_Med_particle = nc.Dataset(Med_particle)
    obs_size = ds_Med_particle.dimensions['obs'].size 
    ds_Med_particle.close()
    return (obs_size)

obs_size = particle_loop_size()
print(obs_size)

def select_particle_location(file_indx, frame):
    #Loading the file and reading the dataset for the particle locations
    Med_particle='/southern/kadanshinyuk/markov-pushforward/3km/data/EMed3km_his_prt_data_bc_winter_extd2.nc'
    ds_Med_particle = nc.Dataset(Med_particle)

    #Making the variables usable
    lon_par = ds_Med_particle['lon'][:]
    lat_par = ds_Med_particle['lat'][:]

    #Selected time, count starts from zero and jumps are by 1 based on a combo of file_indx and frame (0/1)
    selected_time=int(file_indx/2 +frame)
    ########### The problem is here, for 82 items times 12 plus 1 we get 493 which is oob for the particles
    ####### I need to remove the last data for the particles or adjust the loop to end with just the first file (try)
    #Creating lists for xy location from the selected time
    selected_time_part_lon = lon_par[:, selected_time]
    selected_time_part_lat = lat_par[:, selected_time]
    
    return (selected_time_part_lon, selected_time_part_lat)

def plot_vorticity(file_indx, frame,  par_lon, par_lat):
    #Plot setup
    plt.figure(figsize=(8, 4.5)) #Make the plot a bit bigger
    color_map = plt.imshow(masked_heatmap, cmap='bwr', extent=[f_lon.min(), f_lon.max(), f_lat.max(), f_lat.min()]) #lat order is inversed as I need to flip the y axis
    plt.gca().invert_yaxis()
    plt.clim(-2, 2) # Setting the limit for the surface vorticity 
    plt.colorbar(color_map, fraction = 0.025, label=r'$\zeta/f$')
    plt.gca().set_aspect('equal') # Ratio

    # Land contour line
    plt.gca().set_facecolor("white") #enables to change to color of the land
    plt.contour(masked_heatmap.mask,  colors='black', linewidths=0.2, extent=[f_lon.min(), f_lon.max(), f_lat.min(), f_lat.max()])


    # Plot current location
    plt.scatter(par_lon, par_lat, s=1, c=z_sorted, cmap='plasma_r', alpha=0.8)
        
    #Time
    given_date = datetime.strptime('29/01/2018 01:00:00', '%d/%m/%Y %H:%M:%S')
    curr_date = given_date + timedelta(hours=int(file_indx + 2*frame))

    #Labelling
    plt.title(curr_date) 
    plt.ylabel('Latitude [°]')
    plt.xlabel('Longitude [°]')
    
    #Save file
    output_file='../data/cmap_png_files_3km/prt_rvort_bc_%05d_%d.png'  % (file_indx, frame)
    print('OUTPUT:', output_file)
    plt.savefig(output_file, dpi=200)
    plt.close()

    #Plot display should be after the save file if needed


base_name = 'z_EMed3km_his_zvort'
data_dir = '/southern/kadanshinyuk/markov-pushforward/3km/data/rvort_winter_3km'

# # Loop through file numbers from 0 to limit
# for file_indx in range(0, int(obs_size*2), 12):
#     # Format the file name
#     temp_indx = int(1524+file_indx)
#     file_name = f"{base_name}.{temp_indx:05d}.nc"
    
#     # Create the full file path
#     file_path = os.path.join(data_dir, file_name)
    
#     for frame in range(0,2):
#         # Open and modify the file
#         ds_current_vort = nc.Dataset(file_path)

#         rvort = ds_current_vort['rvort'][:]
#         rvort_array = np.array(rvort)
#         dd_rvort_array = rvort_array[(2*frame)] * f_mask #Mask Multipication

#         # Heatmap resize with mask and clearing mask area
#         heatmap=dd_rvort_array[:f_mask.shape[0],:f_mask.shape[1]]
#         masked_heatmap = np.ma.masked_where(f_mask == 0, heatmap)
        
#         # Running functions
#         par_lon, par_lat = select_particle_location(file_indx, frame)
#         plot_vorticity(file_indx, frame,  par_lon, par_lat)

# print("Done plotting. Working on GIF")

# ### Creating and saving the animation
# # Directory containing PNG files
# directory = '../data/cmap_png_files_3km/'

# # Get a list of PNG files in the directory
# files = [file for file in os.listdir(directory) if file.endswith('.png')]
# files.sort()  # Sort files in alphabetical order

# # Create a list to store images
# images = []

# # Read images from files and append to the list
# for file in files:
#     img_path = os.path.join(directory, file)
#     images.append(imageio.imread(img_path))

# # Save the animation
# output_file = '../data/cmap_animation_3km.gif'
# imageio.mimsave(output_file, images, duration=10, loop=0)  # Adjust duration as needed

# print('Done')
