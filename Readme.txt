Hi there!

In order to use these codes with ease, please read below.




Use the following scripts in this order:






1. In parcels_analysis/analysis

1.1. Run Parcels Simulation:
     
     Run MPI_parcels_creator_on_mask.py
     
     mpirun -np {number_of_cores (int)} python MPI_parcels_creator_on_mask.py --season {summer or winter (str)} --resolution {300m or 3km (str)} --cut_n {particles_cut_factor (int)} --dt {interpolation_time_in_sec (int)} --dt_save {trajectory_save_in_hours (int)}

    This should save the simulation into a zarr file filled with proc*.zarr files in parcels_analysis/data/{simulation_time_stamp}


1.2  Extract zarr files into separate nc files:
     
     Run tool_zarr_to_nc_GitHubScript.py: 
     
     python tool_zarr_to_nc_GitHubScript.py --simulation_date_and_time {simulation_time_stamp or simulation_zarr_file_name (str)}

     This should save nc files into parcels_analysis/data/{simulation_time_stamp}


2.  To check trajectories: in /trajectories/analysis
    
    Run: python particles_trajectories_multiple_nc_files.py --simulation_date_and_time {simulation_time_stamp or simulation_zarr_file_name (str)}

    This should save trajectories of 5000 random particles, starting points and ending point to parcels_analysis/data/{simulation_time_stamp}/trajectories/


3. To run markov prediction: in /markov-pushforward
    
   Run: python markov_pushforward_map_multiple_nc_files.py --parcels_file_timestamp {simulation_time_stamp or simulation_zarr_file_name (str)} --x0 {31 to 37 (float)} --y0 {31 to 37 (float)}

   This should:

   1. Create transition matrix and save it as a pkl file in parcels_analysis/data/{simulation_time_stamp}/matrix_data...

   2. Create a prediction of spreading contamination from starting point x0,y0, and save it into parcels_analysis/data/{simulation_time_stamp}/Markov_Analysis_Images_bin....

   3. There are many other options and functions inside the script (including Multi-TM creation for different sets of particles), have a look.

4. To create lagrangian geography clusters (both K means and SEBA): in /markov-pushforward

   Run: python lagrangian_geography.py --simulation_date_and_time {simulation_time_stamp or simulation_zarr_file_name (str)}

   This should output a K-means maps, SEBA maps into parcels_analysis/data/{simulation_time_stamp}/lagrangian_geography...

   There are many other options and functions inside the script, have a look.



    



