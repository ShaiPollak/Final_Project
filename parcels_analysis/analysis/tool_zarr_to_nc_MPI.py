import os
import xarray as xr
from datetime import datetime
import argparse
import glob

debug_name = "20250312_110827"

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Convert Zarr files to NetCDF.")
parser.add_argument("--zarr_path", type=str, help="Path to the Zarr files (use pattern for multiple files).")
parser.add_argument("--nc_output", type=str, help="Path to save the output NetCDF file.")
args = parser.parse_args()

# Extract arguments
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
zarr_directory = args.zarr_path if args.zarr_path else f"/southern/shaipollak/parcels_analysis/data/{debug_name}"
mother_zarr_path = os.path.join(zarr_directory, os.path.basename(zarr_directory) + ".zarr")
zarr_file_pattern = os.path.join(mother_zarr_path, "proc*.zarr")
nc_file_path = args.nc_output if args.nc_output else os.path.join(zarr_directory, f"{os.path.basename(zarr_directory)}.nc")

print(f"Processing Zarr files in: {mother_zarr_path}")
print(f"Saving to NetCDF: {nc_file_path}")

# Check if proc*.zarr files exist inside the mother Zarr
proc_files = glob.glob(zarr_file_pattern)

if proc_files:
    print(f"Found multiple proc*.zarr files: {proc_files}")
    ds = xr.open_mfdataset(proc_files, engine='zarr', parallel=True, concat_dim='trajectory', combine='nested')
elif os.path.exists(mother_zarr_path):
    print("No proc*.zarr files found. Using the mother Zarr file directly...")
    ds = xr.open_dataset(mother_zarr_path, engine='zarr', chunks={})
else:
    raise FileNotFoundError(f"Neither proc*.zarr files nor the mother Zarr file found at {mother_zarr_path}")

# Ensure output directory exists
os.makedirs(os.path.dirname(nc_file_path), exist_ok=True)

# Save the combined dataset to NetCDF
ds.to_netcdf(nc_file_path)
print(f"NetCDF file saved successfully: {nc_file_path}")