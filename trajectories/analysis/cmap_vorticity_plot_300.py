
import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os
import imageio.v2 as imageio
from tqdm import tqdm
import argparse
'''   
    #Grid file, needed solely for creating the limits for the plot
    input_dir_grid = '/indian/vickyverma/EMedCrocoC/INPUT'
    Med_grid = '%s/EMed300m_grd.nc' % input_dir_grid
    ds_Med_grid = nc.Dataset(Med_grid)

    f_mask = ds_Med_grid['mask_psi'][:]
    f_lat = ds_Med_grid['lat_psi'][:]
    f_lon = ds_Med_grid['lon_psi'][:]
    # print(f_mask.shape) >>> (2173, 1599)

    # Defining Z for ### different values, check the lons1d lats1d definitions at parcels_creator_300m.py
    n = np.arange(0,218) #create
    z = np.repeat(n,160) #repeat
    z_sorted = np.sort(z)

    # /southern/kadanshinyuk/parcels/data/testing-2.nc
    parcels_file_name = "testing-4"

    def particle_loop_size():
        # Loading the file and reading the dataset for the particle locations
        # Med_particle='../data/Med300m_his_prt_data_bc_winter.nc'
        Med_particle = f'/southern/kadanshinyuk/parcels/data/{parcels_file_name}.nc'
        ds_Med_particle = nc.Dataset(Med_particle)
        obs_size = ds_Med_particle.dimensions['obs'].size 
        ds_Med_particle.close()
        return (obs_size)

    obs_size = particle_loop_size()
    print(obs_size)

    def select_particle_location(file_indx, frame):
        #Loading the file and reading the dataset for the particle locations
        # Med_particle='../data/EMed300m_his_prt_data_bc_winter.nc'
        Med_particle = f'/southern/kadanshinyuk/parcels/data/{parcels_file_name}.nc'
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

    def plot_vorticity(file_indx, frame,  par_lon, par_lat, output_directory):
        corr_v = 0.01
        #Plot setup
        plt.figure(figsize=(5, 5)) #Make the plot a bit bigger
        plt.imshow(masked_heatmap, cmap='bwr', extent=[f_lon.min()-corr_v, f_lon.max(), f_lat.max(), f_lat.min()-corr_v]) #lat order is inversed as I need to flip the y axis
        # plt.imshow(masked_heatmap, cmap='bwr')

        plt.gca().invert_yaxis()
        plt.clim(-2, 2) # Setting the limit for the surface vorticity 
        colorbar = plt.colorbar()
        plt.gca().set_aspect("equal") #Correct the image size based on the ratio of the lat lon total distance ratio

        # Land contour line
        plt.gca().set_facecolor("white") #enables to change to color of the land
        plt.contour(masked_heatmap.mask,  colors='black', linewidths=0.2, extent=[f_lon.min()-corr_v, f_lon.max(), f_lat.min()-corr_v, f_lat.max()])


        # Plot current location
        # plt.scatter(par_lon, par_lat, s=0.2, c=z_sorted, cmap='plasma_r', alpha=1)
        plt.scatter(par_lon, par_lat, s=0.2, color ='midnightblue', alpha=1)

            
        #Time
        given_date = datetime.strptime('29/01/2018 01:00:00', '%d/%m/%Y %H:%M:%S')
        curr_date = given_date + timedelta(hours=int(file_indx + 2*frame))

        #Labelling
        plt.title(curr_date) 
        plt.ylabel('Latitude [°]')
        plt.xlabel('Longitude [°]')
        colorbar.ax.set_title(r'$\zeta/f$')
        
        # Save file
        output_file= output_directory + 'prt_rvort_bc_%05d_%d.png'  % (file_indx, frame)
        # print('OUTPUT:', output_file)
        plt.savefig(output_file, dpi=300)
        plt.close()

        #Plot display should be after the save file if needed


    # Directory containing the vorticity files
    base_name = 'z_EMed300m_his_zvort'
    data_dir = '/southern/kadanshinyuk/markov-pushforward/300m/data/rvort_winter'

    # Directory containing PNG files
    output_directory = f'../data/{parcels_file_name}/'
    os.makedirs(output_directory, exist_ok=True)


    # Loop through file numbers from 0 to limit
    for file_indx in tqdm(range(0, int((obs_size-1) * 2), 4), desc="Creating files"):
    # for file_indx in range(0, int(obs_size*2), 4):
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

            # Heatmap resize with mask and clearing mask area
            heatmap=dd_rvort_array[:f_mask.shape[0],:f_mask.shape[1]]
            masked_heatmap = np.ma.masked_where(f_mask == 0, heatmap)
            
            # Running functions
            par_lon, par_lat = select_particle_location(file_indx, frame)
            plot_vorticity(file_indx, frame,  par_lon, par_lat, output_directory)

    ### Creating and saving the animation
    print("Starting to work on the animation...")
    # Get a list of PNG files in the directory
    files = [file for file in os.listdir(output_directory) if file.endswith('.png')]
    files.sort()  # Sort files in alphabetical order

    # Create a list to store images
    images = []

    # Read images from files and append to the list
    for file in files:
        img_path = os.path.join(output_directory, file)
        images.append(imageio.imread(img_path))

    # Save the animation
    output_file = f'../data/{parcels_file_name}_b.gif'
    imageio.mimsave(output_file, images, duration=10, loop=0)  # Adjust duration as needed

    # Save the animation as MP4
    output_file = f'../data/{parcels_file_name}_b.mp4'
    imageio.mimsave(output_file, images, fps=24)  # Adjust fps (frames per second) as needed


    print('Done')
    '''



# Parse command-line arguments
parser = argparse.ArgumentParser(description="Plot particle vorticity and create animations.")
parser.add_argument("--simulation_date_and_time", type=str, help="Simulation Date and time in format YYYYMMDD_HHMMSS")
parser.add_argument("--parcels_file", type=str, help="Path to the Parcels output NetCDF file.")
parser.add_argument("--output_dir", type=str, help="Directory to save PNG and animation files.")
#parser.add_argument("--vorticity_data_dir", type=str, help="Path to vorticity data directory.")
args = parser.parse_args()

debug_file = "20250601_134703"

# Extract arguments
file_name = args.simulation_date_and_time if args.simulation_date_and_time else debug_file
parcels_file_name = args.parcels_file if args.parcels_file else f"{file_name}.nc"
output_directory = args.output_dir if args.output_dir else f"/southern/shaipollak/parcels_analysis/data/{file_name}/particles_trajectories"
output_images_dir = os.path.join(output_directory, "vorticity_images")  # New folder for images
data_dir = args.vorticity_data_dir

# Ensure directories exist
os.makedirs(output_directory, exist_ok=True)
os.makedirs(output_images_dir, exist_ok=True)  # Create "images" folder

# Grid file (needed to set plot limits)
input_dir_grid = '/indian/vickyverma/EMedCrocoC/INPUT'
Med_grid = f"{input_dir_grid}/EMed300m_grd.nc"
ds_Med_grid = nc.Dataset(Med_grid)

f_mask = ds_Med_grid['mask_psi'][:]
f_lat = ds_Med_grid['lat_psi'][:]
f_lon = ds_Med_grid['lon_psi'][:]

# Define Z values
n = np.arange(0, 218)
z = np.repeat(n, 160)
z_sorted = np.sort(z)

def particle_loop_size():
    """Get the total number of observations from the Parcels output file."""
    ds_Med_particle = nc.Dataset(args.parcels_file)
    obs_size = ds_Med_particle.dimensions['obs'].size
    ds_Med_particle.close()
    return obs_size

obs_size = particle_loop_size()
print(f"Total observations: {obs_size}")

def select_particle_location(file_indx, frame):
    """Select particle locations from the Parcels output NetCDF file."""
    ds_Med_particle = nc.Dataset(args.parcels_file)
    lon_par = ds_Med_particle['lon'][:]
    lat_par = ds_Med_particle['lat'][:]
    selected_time = int(file_indx / 2 + frame)
    selected_time_part_lon = lon_par[:, selected_time]
    selected_time_part_lat = lat_par[:, selected_time]
    ds_Med_particle.close()
    return selected_time_part_lon, selected_time_part_lat

def plot_vorticity(file_indx, frame, par_lon, par_lat):
    """Plot vorticity and particle positions."""
    corr_v = 0.01
    plt.figure(figsize=(5, 5))
    plt.imshow(masked_heatmap, cmap='bwr', extent=[f_lon.min()-corr_v, f_lon.max(), f_lat.max(), f_lat.min()-corr_v])
    plt.gca().invert_yaxis()
    plt.clim(-2, 2)
    colorbar = plt.colorbar()
    plt.gca().set_aspect("equal")
    plt.gca().set_facecolor("white")
    plt.contour(masked_heatmap.mask, colors='black', linewidths=0.2, extent=[f_lon.min()-corr_v, f_lon.max(), f_lat.min()-corr_v, f_lat.max()])
    plt.scatter(par_lon, par_lat, s=0.2, color='midnightblue', alpha=1)

    given_date = datetime.strptime('29/01/2018 01:00:00', '%d/%m/%Y %H:%M:%S')
    curr_date = given_date + timedelta(hours=int(file_indx + 2 * frame))
    plt.title(curr_date)
    plt.ylabel('Latitude [°]')
    plt.xlabel('Longitude [°]')
    colorbar.ax.set_title(r'$\zeta/f$')

    # Save the file inside "images" folder
    output_file = os.path.join(output_images_dir, f'prt_rvort_bc_{file_indx:05d}_{frame}.png')
    plt.savefig(output_file, dpi=300)
    plt.close()

base_name = 'z_EMed300m_his_zvort'

for file_indx in tqdm(range(0, int((obs_size - 1) * 2), 4), desc="Creating files"):
    file_name = f"{base_name}.{file_indx:05d}.nc"
    file_path = os.path.join(data_dir, file_name)

    for frame in range(0, 2):
        ds_current_vort = nc.Dataset(file_path)
        rvort = ds_current_vort['rvort'][:]
        rvort_array = np.array(rvort)
        dd_rvort_array = rvort_array[(2 * frame)] * f_mask
        heatmap = dd_rvort_array[:f_mask.shape[0], :f_mask.shape[1]]
        masked_heatmap = np.ma.masked_where(f_mask == 0, heatmap)
        
        par_lon, par_lat = select_particle_location(file_indx, frame)
        plot_vorticity(file_indx, frame, par_lon, par_lat)

# Create animation
print("Creating animation...")

# Get a list of PNG files in the "images" folder
files = sorted([file for file in os.listdir(output_images_dir) if file.endswith('.png')])
images = [imageio.imread(os.path.join(output_images_dir, file)) for file in files]

# Save the animation as GIF
output_gif = os.path.join(output_directory, f"cmap_vorticity_plot_{parcels_file_name}_b.gif")
imageio.mimsave(output_gif, images, duration=10, loop=0)

# Save the animation as MP4
#output_mp4 = os.path.join(output_directory, f"{parcels_file_name}_b.mp4")
#imageio.mimsave(output_mp4, images, fps=24)

print("Animation created successfully!")