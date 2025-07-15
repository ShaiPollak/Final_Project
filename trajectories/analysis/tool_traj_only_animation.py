import imageio.v2 as imageio
import os
from tqdm import tqdm
import argparse

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Generate animation from trajectory images.")
parser.add_argument("--parcels_file_path", type=str, help="Path to the Parcels output NetCDF file.")
parser.add_argument("--output_dir", type=str, help="Directory where images are stored and animation will be saved.")
args = parser.parse_args()

# Extract the file name
parcels_file_name = os.path.basename(args.parcels_file_path).replace(".nc", "")
output_directory = os.path.abspath(args.output_dir)  # Ensure absolute path

# Define the correct images directory
images_dir = os.path.join(output_directory, "images")

# Ensure the images directory exists
if not os.path.exists(images_dir):
    print(f"Error: The directory {images_dir} does not exist. Cannot create animation.")
    exit(1)

# Debugging: Print all files inside images directory
print(f"\nChecking files in {images_dir}:")
all_files = os.listdir(images_dir)
print(all_files)

# Find PNG files inside `images/`
png_files = sorted([file for file in all_files if file.endswith('.png')])
print(f"Found {len(png_files)} PNG files in images directory.")

if not png_files:
    print("Error: No PNG files found in 'images' directory. GIF creation aborted.")
    exit(1)

# Load images for animation
images = [imageio.imread(os.path.join(images_dir, file)) for file in tqdm(png_files, desc="Loading images")]

# Check if images were successfully loaded
if not images:
    print("Error: No images were loaded. Animation aborted.")
    exit(1)

print("\nCreating GIF animation...")

# Save the animation as a GIF
gif_output_path = os.path.join(output_directory, f"trajectory_{parcels_file_name}.gif")
imageio.mimsave(gif_output_path, images, duration=10, loop=0)

print(f"Created GIF: {gif_output_path}")
print("Done!")
