
#python tool_zarr_to_nc_GitHubScript.py --simulation_date_and_time 20250605_020323 --split True --verbose

import xarray as xr
import os
import glob
import dask
from dask.diagnostics import ProgressBar
import argparse


def zarr_to_netcdf(zarr_path, output_path=None, use_dask=False, verbose=False, encoding=False, encoding_level=3):
    """
    Convert a .zarr store to a netCDF (.nc) file.

    Parameters
    ----------
    zarr_path : str
        Path to the .zarr store.
    output_path : str, optional
        Path to save the output .nc file. If not provided, the output will have the same name as the input with a .nc extension.
    use_dask : bool, optional
        Use dask parallelization. Enables progress bar.
    verbose : bool, optional
        Print progress messages.
    encoding : bool, optional
        Compress the output file using zlib.
    encoding_level : int, optional
        Compression level for the output file.
    """
    if not os.path.exists(zarr_path):
        raise FileNotFoundError(f"The provided path {zarr_path} does not exist.")

    if verbose:
        print(f"Converting {zarr_path} to .nc")

    # Load the .zarr store into an xarray dataset
    if use_dask:
        ds = xr.open_zarr(zarr_path, chunks='auto')
    else:
        ds = xr.open_zarr(zarr_path, chunks=None)

    if output_path is None:
        output_path = os.path.splitext(zarr_path)[0] + '.nc'

    encoding_dict = None
    if encoding:
        encoding_dict = {key: {"zlib": True, "complevel": encoding_level} for key in ds.data_vars}

    if use_dask:
        with ProgressBar():
            ds.to_netcdf(output_path, compute=True, encoding=encoding_dict)
    else:
        ds.to_netcdf(output_path, encoding=encoding_dict)

    if verbose:
        print(f"Converted {zarr_path} to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Convert Zarr files to NetCDF.")
    parser.add_argument("--simulation_date_and_time", type=str, help="Simulation Date and time in format YYYYMMDD_HHMMSS")
    parser.add_argument("--zarr_path", type=str, help="Other path to the Zarr files (use pattern for multiple files).")
    parser.add_argument("--nc_output", type=str, help="Other path to save the output NetCDF file.")
    parser.add_argument("--split", type=bool, default=True, help="Split output: create one NetCDF per proc*.zarr file.")
    parser.add_argument("--use_dask", "-d", action="store_true", help="Use dask parallelization. Enables progress bar.")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print progress messages.")
    parser.add_argument("--encoding", "-e", action="store_true", help="Compress the output file using zlib.")
    parser.add_argument("--encoding_level", "-l", type=int, default=3, help="Compression level for the output file.")
    args = parser.parse_args()

    debug_name = "debug_timestamp"
    simulation_time_stamp = args.simulation_date_and_time if args.simulation_date_and_time else debug_name
    zarr_directory = args.zarr_path if args.zarr_path else f"/southern/shaipollak/parcels_analysis/data/{simulation_time_stamp}"
    mother_zarr_path = os.path.join(zarr_directory, os.path.basename(zarr_directory) + ".zarr")
    zarr_file_pattern = os.path.join(mother_zarr_path, "proc*.zarr")
    nc_file_path = args.nc_output if args.nc_output else os.path.join(zarr_directory, f"{os.path.basename(zarr_directory)}.nc")

    if args.split:
        proc_files = sorted(glob.glob(zarr_file_pattern))
        if args.verbose:
            print(f"Found {len(proc_files)} files matching pattern {zarr_file_pattern}")
        for proc in proc_files:
            proc_id = os.path.basename(proc).replace(".zarr", "")
            out_path = nc_file_path.replace(".nc", f"_{proc_id}.nc")
            zarr_to_netcdf(proc, out_path, use_dask=args.use_dask, verbose=args.verbose, encoding=args.encoding, encoding_level=args.encoding_level)
    else:
        # Expecting full dataset in mother_zarr_path
        zarr_to_netcdf(mother_zarr_path, nc_file_path, use_dask=args.use_dask, verbose=args.verbose, encoding=args.encoding, encoding_level=args.encoding_level)


if __name__ == "__main__":
    main()