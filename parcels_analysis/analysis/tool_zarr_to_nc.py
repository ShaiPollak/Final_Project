# To run this script, use the command:
# screen -S myscript #Create a new screen session
# python tool_zarr_to_nc_MPI.py --simulation_date_and_time YYYYMMDD_HHMMSS
# screen -r myscript #Reattach to the screen session


import os
import xarray as xr
import logging
import glob
from datetime import datetime
import argparse
from tqdm import tqdm  # Progress bar


#------------------------------Config------------------------------------------

debug_name = "20250604_230117"

# Parse command-line arguments

parser = argparse.ArgumentParser(description="Convert Zarr files to NetCDF.")
parser.add_argument("--simulation_date_and_time", type=str, help="Simulation Date and time in format YYYYMMDD_HHMMSS")
parser.add_argument("--zarr_path", type=str, help="Other path to the Zarr files (use pattern for multiple files).")
parser.add_argument("--nc_output", type=str, help="Other path to save the output NetCDF file.")
parser.add_argument("--split", type=bool, default=True, help="Split output: create one NetCDF per proc*.zarr file.")

args = parser.parse_args()

simulation_time_stamp = args.simulation_date_and_time if args.simulation_date_and_time is not None else debug_name
zarr_directory = args.zarr_path if args.zarr_path else f"/southern/shaipollak/parcels_analysis/data/{simulation_time_stamp}"
mother_zarr_path = os.path.join(zarr_directory, os.path.basename(zarr_directory) + ".zarr")
zarr_file_pattern = os.path.join(mother_zarr_path, "proc*.zarr")
nc_file_path = args.nc_output if args.nc_output else os.path.join(zarr_directory, f"{os.path.basename(zarr_directory)}.nc")
split_output = args.split

# Check if proc*.zarr files exist inside the mother Zarr
proc_files = glob.glob(zarr_file_pattern)

#------------------------------Logging Setup-----------------------------------
# Setup logging
now = datetime.now().strftime('%Y%m%d_%H%M%S')


log_file = f"{zarr_directory}/zarr_to_nc_{now}.log"
logging.basicConfig(
    filename=log_file, level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
console_handler = logging.StreamHandler()  # Also log to console
console_handler.setLevel(logging.INFO)
logging.getLogger().addHandler(console_handler)
logging.info(f"Processing Zarr files in: {mother_zarr_path}")
logging.info(f"Output NetCDF file: {nc_file_path}")


#-------------------------------------------Main Processing--------------------------------------------
try:
    datasets = []

    if proc_files:
        total_files = len(proc_files)
        logging.info(f"Found {total_files} proc*.zarr files to process.")

        if split_output:
            logging.info("Split output is enabled. Each proc*.zarr file will be saved as a separate NetCDF file.")
            for i, file in enumerate(tqdm(proc_files, desc="Processing Zarr files", unit="file")):
                logging.info(f"[{(i+1)/total_files*100:.2f}%] Loading file: {file}")
                ds = xr.open_dataset(file, engine='zarr', chunks={}, decode_timedelta=False)

                # Handle time variable
                if 'time' in ds.variables:
                    min_time, max_time = ds['time'].min().values, ds['time'].max().values
                    nan_count = ds['time'].isnull().sum().values
                    logging.info(f"Time in {file} | Min: {min_time}, Max: {max_time}, NaN count: {nan_count}")
                    ds['time'] = ds['time'].fillna(0)
                    ds['time'] = xr.decode_cf(ds)['time']
                    ds['time'] = ds['time'].astype('int64')

                # Save to individual NetCDF
                proc_name = os.path.basename(file).replace(".zarr", ".nc")
                nc_out_file = os.path.join(zarr_directory, proc_name)
                ds.to_netcdf(nc_out_file, engine="netcdf4", format="NETCDF4", encoding={'time': {'dtype': 'int64'}})
                logging.info(f"✅ Saved {nc_out_file}")

        else:
            logging.info("Split output is disabled. Merging all proc*.zarr files into a single NetCDF file.")
            for i, file in enumerate(tqdm(proc_files, desc="Processing Zarr files", unit="file")):
                logging.info(f"[{(i+1)/total_files*100:.2f}%] Loading file: {file}")
                ds = xr.open_dataset(file, engine='zarr', chunks={}, decode_timedelta=False)

                # Debug: Check time variable
                if 'time' in ds.variables:
                    min_time, max_time = ds['time'].min().values, ds['time'].max().values
                    nan_count = ds['time'].isnull().sum().values
                    logging.info(f"Time in {file} | Min: {min_time}, Max: {max_time}, NaN count: {nan_count}")
                    ds['time'] = ds['time'].fillna(0)
                    ds['time'] = xr.decode_cf(ds)['time']

                datasets.append(ds)

        logging.info("Attempting to merge datasets... This may take a very long time for large datasets (1 hour+)")
        ds = xr.concat(datasets, dim="trajectory", coords="minimal", compat="override", join="outer")
        ds['time'] = ds['time'].astype('int64')
        ds.to_netcdf(nc_file_path, engine="netcdf4", format="NETCDF4", encoding={'time': {'dtype': 'int64'}})
        print(f"✅ Successfully saved NetCDF file: {nc_file_path}")
        logging.info(f"✅ NetCDF file saved successfully: {nc_file_path}")

except Exception as e:
    logging.error(f"❌ Error during Zarr extraction: {str(e)}", exc_info=True)
    raise  # Re-raise the exception for debugging
