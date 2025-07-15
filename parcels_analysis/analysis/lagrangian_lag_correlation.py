import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import netCDF4 as nc



############ --- Configuration --- #################
file_name = "20250629_000412"
folder_path = f'/southern/shaipollak/parcels_analysis/data/{file_name}'
output_dir = f"{folder_path}/lag_correlation"
os.makedirs(output_dir, exist_ok=True)
dt = 900  # resolution time, seconds, change according to config file






#------------------------------------------ Functions -------------------------------------------
def autocorrelation(u_series):
    """Compute autocorrelation R(τ) of a 1D signal u(t)."""
    u_series = u_series - np.mean(u_series)  # subtract mean
    n = len(u_series)
    result = np.correlate(u_series, u_series, mode='full')
    result = result[n-1:] / np.arange(n, 0, -1)  # normalize by number of overlapping points
    result = result / result[0]  # normalize autocorrelation so R(0) = 1
    return result


#--------------------------------------Main Scipt--------------------------------------------------


# --- Load files ---
print(f"Looking for files: {file_name}_proc*.nc")
nc_file_pattern = os.path.join(folder_path, f"{file_name}_proc*.nc")
nc_files = sorted(glob.glob(nc_file_pattern))

if not nc_files:
    raise FileNotFoundError(f"No files found for pattern: {nc_file_pattern}")

x, y, t, d_id, u, v = [], [], [], [], [], []
for path in nc_files:
    print(f"Reading file: {path}")
    with nc.Dataset(path) as ds:
        lon_par = np.array(ds['lon'][:], dtype=np.float32)
        lat_par = np.array(ds['lat'][:], dtype=np.float32)
        id_par = np.array(ds['trajectory'][:], dtype=np.int32)
        time_par = np.array(ds['time'][:], dtype=np.float32)
        u_par = np.array(ds['u'][:], dtype=np.float32)
        v_par = np.array(ds['v'][:], dtype=np.float32)

        x.append(lon_par)
        y.append(lat_par)
        t.append(time_par)
        d_id.append(id_par)
        u.append(u_par)
        v.append(v_par)

# --- Stack into full arrays ---
x = np.concatenate(x)
y = np.concatenate(y)
t = np.concatenate(t)
d_id = np.concatenate(d_id)
u = np.concatenate(u)
v = np.concatenate(v)

print(u[0:500, 0])

# --- Check shapes ---
assert u.shape == v.shape, "u and v must have the same shape"


n_particles = len(np.unique(d_id)) #number of particles
n_lags = u.shape[1] #number of timesteps

# --- Compute autocorrelations ---
print("Computing autocorrelation!")
autocorr_u = np.array([autocorrelation(u[p, :]) for p in range(n_particles)])
autocorr_v = np.array([autocorrelation(v[p, :]) for p in range(n_particles)])

mean_autocorr_u = np.nanmean(autocorr_u, axis=0)
std_autocorr_u = np.nanstd(autocorr_u, axis=0)

mean_autocorr_v = np.nanmean(autocorr_v, axis=0)
std_autocorr_v = np.nanstd(autocorr_v, axis=0)

# --- Lags in hours ---
lags = np.arange(n_lags) * dt / 3600  # convert seconds to hours

# --- Plot both u and v ---
print("Plotting...")
plt.figure(figsize=(10, 6))
plt.plot(lags, mean_autocorr_u, label='u velocity', color='blue')
plt.fill_between(lags, mean_autocorr_u - std_autocorr_u, mean_autocorr_u + std_autocorr_u, alpha=0.2, color='blue')

plt.plot(lags, mean_autocorr_v, label='v velocity', color='red')
plt.fill_between(lags, mean_autocorr_v - std_autocorr_v, mean_autocorr_v + std_autocorr_v, alpha=0.2, color='red')

plt.xlabel('Lag time [hours]')
plt.ylabel('Autocorrelation R(τ)')
plt.title('Lagrangian Velocity Autocorrelation (u and v)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(f"{output_dir}/lagrangian_velocity_autocorrelation.png", dpi=300)
plt.close()

print("Autocorrelation plot saved successfully.")
