import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import sys
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import argparse
import seba
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, ScalarFormatter
import scipy.linalg as sla
import seba as seba_module

# Add pygtm path
sys.path.append(f'/southern/shaipollak')

from pygtm.physical import physical_space
from pygtm.matrix import matrix_space

# =========================================================== USER CONFIG ===========================================================

parser = argparse.ArgumentParser(description="Lagrangian Geography Analysis")
parser.add_argument("--simulation_date_and_time", type=str, help="Simulation Date and time in format YYYYMMDD_HHMMSS")
args = parser.parse_args()


debug_file = "20250605_020323"
simulation_date_and_time = args.simulation_date_and_time if args.simulation_date_and_time else debug_file

T = 1 # SET THE TRANSITION TIME (IN DAYS)
spatial_resolution = 120

print(f"Creating Lagrangian Geography Analysis for T={T} days, simulation date and time: {simulation_date_and_time}")
matrix_dir = f'/southern/shaipollak/parcels_analysis/data/{simulation_date_and_time}/matrix_data'
matrix_file = os.path.join(matrix_dir, f'transition_matrix_T{T}_res={spatial_resolution}.npy')
pkl_path = os.path.join(matrix_dir, f'matrix_metadata_T{T}_res={spatial_resolution}.pkl')
output_dir = f'/southern/shaipollak/parcels_analysis/data/{simulation_date_and_time}/lagrangian_geography/T={T}_res={spatial_resolution}'
os.makedirs(output_dir, exist_ok=True)

random_matrix_dir = f'/southern/shaipollak/parcels_analysis/data/{simulation_date_and_time}/matrix_data/random_matrices_T={T}_res_{spatial_resolution}'
os.makedirs(random_matrix_dir, exist_ok=True)


# ======================================= Define spatial bounds and resolution ===================================================

lon_bounds = [31, 37]
lat_bounds = [31, 37]
selected_index = 1 

# =========================================================== Load data ===========================================================
print(f"Loading transition matrix from {matrix_file} and metadata from {pkl_path}...")
P = np.load(matrix_file)
with open(pkl_path, 'rb') as f:
    metadata = pickle.load(f)


bins = metadata['domain_bins']
id_og = metadata['domain_id_og']

# === Rebuild physical domain ===
d = physical_space(lon_bounds, lat_bounds, spatial_resolution)
d.bins = bins
d.id_og = id_og

# === Matrix Object ===
m = matrix_space(d)
m.P = P
m.left_and_right_eigenvectors(n=100)  # Compute eigenvectors (adjust n as needed)

print(f"Transition matrix loaded with shape {m.P.shape} and {m.P.size} elements.")


# =========================================================== Sanity Checks ===========================================================
def stochastic_check():
    """
    Check if the transition matrix P is stochastic (each row sums to 1).
    """
    # Check if the transition matrix is stochastic
    row_sums = np.sum(m.P, axis=1)
    tolerance = 1e-6  # acceptable numerical error margin
    non_conserving_rows = np.where(np.abs(row_sums - 1) > tolerance)[0]
    print(f"\n--- Row Conservation Check (T={T}) ---")
    print(f"Total rows: {m.P.shape[0]}")
    print(f"Min row sum: {np.min(row_sums):.6f}")
    print(f"Max row sum: {np.max(row_sums):.6f}")
    print(f"Rows that do NOT sum to 1 (tolerance={tolerance}): {len(non_conserving_rows)}")
    print("Indices of non-conserving rows:", non_conserving_rows[:])

def sanity_check():

    """
    Perform sanity checks on the eigenvalues and eigenvectors of the matrix.
    This includes checking the eigenvalue spectrum, normalizing the right eigenvector (stationary distribution),
    and normalizing the left eigenvector.
    Returns:
        v_right: Normalized right eigenvector (stationary distribution).
        v_left: Normalized left eigenvector.
    """
    
    
    # === SANITY CHECKS ===
    print("\n--- Eigenvalue & Eigenvector Sanity Checks ---")
    idx_R = np.argmin(np.abs(m.eigR - 1))
    idx_L = np.argmin(np.abs(m.eigL - 1))

    print(f"Index of λ ≈ 1 in right spectrum: {idx_R}, eigenvalue: {m.eigR[idx_R]}")
    print(f"Index of λ ≈ 1 in left  spectrum: {idx_L}, eigenvalue: {m.eigL[idx_L]}")

    # Normalize right eigenvector (stationary distribution)
    v_right = m.R[:, idx_R]
    if np.all(np.abs(v_right.imag) < 1e-10):
        v_right = v_right.real
    else:
        print("Warning: Right eigenvector λ≈1 is significantly complex. Taking abs().")
        v_right = np.abs(v_right)

    v_right = np.abs(v_right)
    v_right = v_right / np.sum(v_right)
    print(f"Sum of right eigenvector (stationary distribution): {v_right.sum():.6f}")

    # Normalize left eigenvector
    v_left = m.L[:, idx_L]
    if np.all(np.abs(v_left.imag) < 1e-10):
        v_left = v_left.real
    else:
        print("Warning: Left eigenvector λ≈1 is significantly complex. Taking abs().")
        v_left = np.abs(v_left)

    v_left = np.abs(v_left)
    v_left = v_left / np.sum(v_left)

    print("Right eigenvector stats (stationary distribution):")
    print(f"   min: {v_right.min():.3e}, max: {v_right.max():.3e}, sum: {v_right.sum():.6f}")
    print("Left eigenvector stats:")
    print(f"   min: {v_left.min():.3e}, max: {v_left.max():.3e}, sum: {v_left.sum():.6f}")

    return v_right, v_left

# =========================================================== Functions ===========================================================
def geo_map(ax):
    '''
    Plot Helper - Set up a Cartopy map with the specified bounds and features.
    '''
    ax.set_xticks(np.arange(31, 38), crs=ccrs.PlateCarree())
    ax.set_yticks(np.arange(31, 38), crs=ccrs.PlateCarree())
    ax.xaxis.set_major_formatter(LongitudeFormatter())
    ax.yaxis.set_major_formatter(LatitudeFormatter())
    ax.tick_params(axis='x', labelsize=7)
    ax.tick_params(axis='y', labelsize=7)
    ax.add_feature(cfeature.LAND, facecolor='silver', zorder=1)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.25, zorder=1)

def get_discrete_cmap(n):
    """Plot Helprt - Returns a ListedColormap with `n` unique colors."""
    if n <= 10:
        return ListedColormap(plt.cm.tab10.colors[:n])
    elif n <= 20:
        return ListedColormap(plt.cm.tab20.colors[:n])
    else:
        base = cm.get_cmap('gist_ncar', n)
        return ListedColormap(base(np.linspace(0, 1, n)))

def plot_eigenvalue_spectrum():
    '''
    Plot the eigenvalue spectrum of the transition matrix.
    '''

    print("Plotting eigenvalues spectrum...")

    eigvals = m.eigR
    real_vals = eigvals.real
    imag_vals = eigvals.imag

    fig, ax = plt.subplots(figsize=(6, 6))  # square aspect
    ax.scatter(real_vals, imag_vals, c='blue', s=10, label='Eigenvalues', zorder=2)

    # Axis settings
    ax.set_xlabel('Real Part')
    ax.set_ylabel('Imaginary Part')
    ax.set_title('Eigenvalue Spectrum')
    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

    # Set limits with margin
    rmax = np.max(np.abs(real_vals)) * 1.1
    imax = np.max(np.abs(imag_vals)) * 1.1
    ax.set_xlim(-rmax, rmax)
    ax.set_ylim(-imax, imax)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/eigenvalue_spectrum.png", dpi=300)
    plt.close()

def plot_eigenvector(d, vec, title, fname="eigenvector"):
    '''
    Plot a single eigenvector on the geographical domain.
    '''

    print(f"Plotting eigenvector: {title}")
    
    vec_matrix = d.vector_to_matrix(vec)
    masked = np.ma.masked_where(np.isnan(vec_matrix), vec_matrix)

    fig = plt.figure(figsize=(6, 5), dpi=200)
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    geo_map(ax)

    # Main plot
    c = ax.pcolormesh(
        d.vx, d.vy, masked,
        shading='auto',
        cmap='viridis',
        transform=ccrs.PlateCarree(),
        alpha=0.75
    )

    # Colorbar at bottom
    cbar = fig.colorbar(c, ax=ax, orientation='horizontal', pad=0.07, aspect=40)
    cbar.set_label("Eigenvector Value", fontsize=7)

    # Format: scientific notation (x10^), no offset, small ticks
    sci_formatter = ScalarFormatter(useMathText=True)
    sci_formatter.set_powerlimits((-3, 3))  # use sci notation outside this range
    sci_formatter.set_scientific(True)
    sci_formatter.set_useOffset(False)
    cbar.ax.tick_params(labelsize=6)
    cbar.formatter = sci_formatter
    cbar.update_ticks()

    ax.set_title(title, fontsize=9)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{fname}.png", dpi=300)
    plt.close()
    
def plot_lagrangian_geography(m, d, selected_vec=[1, 2, 3], n_clusters=4):
    """
    Plot Lagrangian Geography Clustering using KMeans on selected eigenvectors.
    Args:
        m: matrix_space object with transition matrix and eigenvectors.
        d: physical_space object with domain geometry.
        selected_vec: list of indices of eigenvectors to use for clustering.
        n_clusters: number of clusters to form.
    """

    print("Plotting Lagrangian Geography Clustering...")
    labels = m.lagrangian_geography(selected_vec, n_clusters)
    label_matrix = d.vector_to_matrix(labels)

    cmap = get_discrete_cmap(n_clusters)
    bounds = np.arange(n_clusters + 1) - 0.5
    norm = BoundaryNorm(bounds, cmap.N)

    fig = plt.figure(figsize=(6, 5), dpi=200)
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    geo_map(ax)

    mesh = ax.pcolormesh(d.vx, d.vy, label_matrix, shading='auto', cmap=cmap, norm=norm, transform=ccrs.PlateCarree())
    handles = [Patch(color=cmap(i), label=f"Province {i}") for i in range(n_clusters)]
    ax.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

    ax.set_title(f"(KMeans) Clustering, for {len(selected_vec)+1} first eigenvectors and k={n_clusters}", fontsize=9)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/kmeans_clustering_vecs={len(selected_vec)+1}_k={n_clusters}.png", dpi=300)
    plt.close()
    return labels

# === Plot Residence or Hitting Time ===
def plot_time_map(d, vec, label="Hitting Time Map Between Regions", fname="Hitting_Time_Map"):
    print("Plotting Hitting Time Map...")
    vec_matrix = d.vector_to_matrix(vec)
    fig = plt.figure(figsize=(6, 5), dpi=200)
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    geo_map(ax)
    c = ax.pcolormesh(d.vx, d.vy, vec_matrix, shading='auto', cmap='magma', transform=ccrs.PlateCarree())
    fig.colorbar(c, ax=ax, label=label)
    ax.set_title(f"{label} Map")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{fname}.png", dpi=300)
    plt.close()

# === Plot SEBA Labels ===
def plot_seba_labels(A, d, fname="seba_labels", title="SEBA Provinces"):

    print("Plotting SEBA provinces...")
    A_matrix = d.vector_to_matrix(A)
    unique_labels = np.unique(A)
    n_clusters = len(unique_labels)

    cmap = get_discrete_cmap(n_clusters)
    bounds = np.arange(n_clusters + 1) - 0.5
    norm = BoundaryNorm(bounds, cmap.N)

    fig = plt.figure(figsize=(6, 5), dpi=200)
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    geo_map(ax)

    mesh = ax.pcolormesh(d.vx, d.vy, A_matrix, shading='auto', cmap=cmap, norm=norm, transform=ccrs.PlateCarree())
    handles = [Patch(color=cmap(i), label=f"Province {i}") for i in range(n_clusters)]
    ax.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(f"{fname}.png", dpi=300)
    plt.close()

def plot_max_likelihood_provinces(A, d, fname="seba_max_likelihood", title="SEBA Provinces (Max Likelihood)"):
    """
    Plot province labels obtained from SEBA's max_likelihood clustering.
    
    Args:
        A: 1D array of cluster labels (from max_likelihood)
        d: physical_space instance (contains domain geometry)
        fname: output filename (without extension)
        title: plot title
    """
    print("Plotting SEBA Max Likelihood provinces...")
    A_matrix = d.vector_to_matrix(A)
    unique_labels = np.unique(A)
    n_clusters = len(unique_labels)

    cmap = get_discrete_cmap(n_clusters)
    bounds = np.arange(n_clusters + 1) - 0.5
    norm = BoundaryNorm(bounds, cmap.N)

    fig = plt.figure(figsize=(6, 5), dpi=200)
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    geo_map(ax)

    mesh = ax.pcolormesh(d.vx, d.vy, A_matrix, shading='auto', cmap=cmap, norm=norm, transform=ccrs.PlateCarree())
    handles = [Patch(color=cmap(i), label=f"Province {i+1}") for i in range(n_clusters)]
    ax.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(f"{fname}.png", dpi=300)
    plt.close()

def create_eigenvalue_certainty_map(m, full_eigvals, matrix_dir=random_matrix_dir, n=20):
    """
    Plot a certainty map (mean and std) of real parts of eigenvalues across sampled 80% matrices.

    Args:
        m: matrix_space object of full matrix (already has eigenvalues computed).
        matrix_dir: path to random matrix folder.
        full_eigvals: array of full matrix eigenvalues (right), used for baseline.
        n: number of leading eigenvalues to consider.
    """

    print("Computing eigenvalue certainty map...")
    eig_matrix = []

    for i in range(1, 51):
        matrix_path = os.path.join(matrix_dir, f'transition_matrix_Tmatrix_{i}.npy')
        if not os.path.exists(matrix_path):
            print(f"Missing matrix {i}")
            continue

        print(f"Finding eigenvalues of matrix {i}...")    

        # Load sampled matrix and compute eigenvalues
        P_sampled = np.load(matrix_path)
        m_sample = matrix_space(m.domain)
        m_sample.P = P_sampled
        m_sample.left_and_right_eigenvectors(n*3) # Compute more than n to ensure we have enough
        eig_real = np.real(m_sample.eigR[:n])
        if len(eig_real) == n:
            eig_matrix.append(eig_real)
        else:
            print(f"Skipping matrix {i}: only {len(eig_real)} eigenvalues returned.")
            continue

    eig_matrix = np.array(eig_matrix)

    # Compute mean and std across matrices
    mean_vals = np.mean(eig_matrix, axis=0)
    std_vals = np.std(eig_matrix, axis=0)
    full_vals = np.real(full_eigvals[:n])

    x = np.arange(1, n + 1)

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x, mean_vals, label='Mean', color='blue')
    ax.fill_between(x, mean_vals - std_vals, mean_vals + std_vals, color='blue', alpha=0.2, label='±1 Std Dev')
    ax.plot(x, full_vals, label='Full Matrix Eigenvalues', color='red', linestyle='--')

    ax.set_xlabel("Eigenvalue Index")
    ax.set_ylabel("Eigenvalue")
    ax.set_title(f"First {n} Eigenvalues Certainty Map of T={T} Transition Matrix")
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()

    outpath = os.path.join(output_dir, "eigenvalue_certainty_map.png")
    plt.savefig(outpath, dpi=300)
    plt.close()
    print(f"Certainty map saved to {outpath}")

def run_seba_reliability_diagnostic(V_full, max_r=20, plot=True):
    """
    Run SEBA diagnostics using your seba.py module to choose the optimal number of input eigenvectors.

    Parameters
    ----------
    V_full : np.ndarray
        Full matrix of eigenvectors (shape: n_points x n_vectors)
    max_r : int
        Maximum number of eigenvectors to test
    plot : bool
        Whether to plot the diagnostic curve

    Returns
    -------
    r_vals : list of int
        Number of eigenvectors used
    reliability_scores : list of float
        Diagnostic score for each r: -Σ min(s_j)
    """
    r_vals = []
    reliability_scores = []

    for r in range(2, max_r + 1):
        V = V_full[:, :r]
        S = seba.seba(V)
        score = -np.sum(np.min(S, axis=0))
        r_vals.append(r)
        reliability_scores.append(score)

    if plot:
        os.makedirs(output_dir, exist_ok=True)
        fig = plt.figure(figsize=(8, 4))
        plt.plot(r_vals, reliability_scores, marker='o')
        plt.title("SEBA Reliability Diagnostic")
        plt.xlabel("Number of Input Eigenvectors (r)")
        plt.ylabel("Score = -Σ min(s_j)")
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()

        outpath = os.path.join(output_dir, "seba_reliability_diagnostic.png")
        plt.savefig(outpath, dpi=300)
        plt.close()
        print(f"[SEBA Reliability] Saved plot to: {outpath}")

    return r_vals, reliability_scores

def seba_sparse_reliability_diagnostic_with_plot(V_full, r_max=20):
    """
    Computes SEBA sparse reliability diagnostic and saves both a heatmap and line plot.

    For each r = 2 to r_max, run SEBA with the first r eigenvectors.
    For each k ≤ r, compute the diagnostic score:
        Score(k) = -sum of min(S_ij) over the k most localized SEBA vectors
    """
    output_dir = f"/southern/shaipollak/parcels_analysis/data/{simulation_date_and_time}/T={T}_res={spatial_resolution}/lagrangian_geography/SEBA"
    os.makedirs(output_dir, exist_ok=True)

    p, total_r = V_full.shape
    r_max = min(r_max, total_r)
    score_matrix = np.full((r_max - 1, r_max - 1), np.nan)

    for r in range(2, r_max + 1):
        V = V_full[:, :r]
        S = seba(V)

        # Compute min value for each SEBA vector (localization diagnostic)
        reliability = np.min(S, axis=0)
        sorted_indices = np.argsort(reliability)  # most localized first

        for k in range(1, r):  # k = number of provinces to keep
            top_k = sorted_indices[:k]
            score = -np.sum(reliability[top_k])
            score_matrix[r - 2, k - 1] = score

    # --- Plot Heatmap ---
    fig, ax = plt.subplots(figsize=(10, 6))
    cax = ax.imshow(score_matrix, origin='lower', cmap='viridis', aspect='auto')
    fig.colorbar(cax, ax=ax, label='-Σ min(S_ij)')
    ax.set_xlabel('k (Number of Sparse Vectors)', fontsize=12)
    ax.set_ylabel('r (Number of Input Eigenvectors)', fontsize=12)
    ax.set_title("SEBA Diagnostics (Top-k Sorted)", fontsize=14)

    ax.set_xticks(np.arange(r_max - 1))
    ax.set_xticklabels(np.arange(1, r_max), fontsize=10)  # integers on x-axis
    ax.set_yticks(np.arange(r_max - 1))
    ax.set_yticklabels(np.arange(2, r_max + 1), fontsize=10)  # integers on y-axis

    plt.tight_layout()
    heatmap_path = os.path.join(output_dir, "seba_diagnostics_advanced.png")
    plt.savefig(heatmap_path, dpi=300)
    plt.close()

    # --- Plot Cumulative Curves for Each k ---
    fig, ax = plt.subplots(figsize=(10, 6))
    for k in range(1, r_max - 1):
        x_vals = np.arange(k + 1, r_max + 1)
        y_vals = [score_matrix[r - 2, k - 1] for r in x_vals]
        ax.plot(x_vals, y_vals, marker='o', label=f"k = {k}")
    ax.set_xlabel("r (Number of Eigenvectors)", fontsize=12)
    ax.set_ylabel("Score = -Σ min(S_ij)", fontsize=12)
    ax.set_title("SEBA Reliability: Sorted Top-k", fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(title="k = Number of Provinces", fontsize=9)
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))

    plt.tight_layout()
    curves_path = os.path.join(output_dir, "seba_reliability_all_k_curves.png")
    plt.savefig(curves_path, dpi=300)
    plt.close()

    return score_matrix

def plot_seba_stacked_diagnostic(score_matrix, save_path="seba_diagnostics_stacked.png"):
    """
    Plot the SEBA stacked diagnostic as in the paper: one line per k (cumulative min(S_ij) for j=1..k).
    
    Parameters
    ----------
    score_matrix : 2D np.ndarray
        Matrix of shape (r_max - 1, r_max - 1), where [r-2, k-1] = -Σ min(S_ij) for first k features using r eigenvectors
    save_path : str
        Output file path to save the figure
    """
    r_max = score_matrix.shape[0] + 1  # since it starts from r=2
    r_vals = np.arange(2, r_max + 1)

    plt.figure(figsize=(10, 6))
    colors = plt.cm.tab20(np.linspace(0, 1, r_max - 1))

    # Plot cumulative min(S_ij) lines for each k
    for k in range(1, r_max):
        values = score_matrix[k-1:r_max-1, k-1]  # from r=k to r_max
        r_range = r_vals[k-1:]
        plt.plot(r_range, values, marker='o', linewidth=1, label=f'k={k}', color=colors[k % len(colors)])

    # Plot envelope (max for each r)
    envelope = np.max(score_matrix, axis=1)
    plt.plot(r_vals, envelope, linestyle='--', color='black', label='Envelope')

    # Optional: r_min = argmax gradient drop
    diffs = np.diff(envelope)
    r_min = np.argmin(diffs) + 2  # shift because r starts at 2
    plt.scatter(r_min, envelope[r_min - 2], color='black', s=50, marker='D', label=r'$r_{\mathrm{min}}$')

    plt.xlabel("r (Number of Eigenvectors)")
    plt.ylabel(r"$|\min \, S_{ij}|$ (stacked sum)")
    plt.title("SEBA Stacked Diagnostic Plot")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='upper left', fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved stacked SEBA diagnostic to: {save_path}")
# ===================================================== Examples =================================================

#score_matrix = seba_sparse_reliability_diagnostic_with_plot(m.R[:, 1:20], r_max=20, output_dir=output_dir) 
#plot_seba_stacked_diagnostic(score_matrix, save_path=os.path.join(output_dir, "seba_diagnostics_stacked.png"))

def create_k_means_and_eigenvectors_graphs():

    print("=== Running Lagrangian Geography Analysis ===")
    
    # Check if the transition matrix is stochastic
    stochastic_check()

    # Plot Eigenvalue Spectrum
    #plot_eigenvalue_spectrum()

    # Sanity Checks on eigenvalues and eigenvectors, # and normalize the right and left eigenvectors
    v_right, v_left = sanity_check()


    # Plot properly normalized first eigenvectors (1st is stationary)
    plot_eigenvector(d, v_right, "Right Eigenvector (Stationary Distribution)", "eigvec_right_stationary")
    plot_eigenvector(d, v_left, "Left Eigenvector (λ ≈ 1)", "eigvec_left_stationary")

    # Plot other top eigenvectors as-is
    for i in range(1, 15):
        plot_eigenvector(d, m.R[:, i], f"Right Eigenvector {i+1} (λ = {m.eigR[i]:.3f})", f"eigvec_right_{i+1}")
        plot_eigenvector(d, m.L[:, i], f"Left Eigenvector {i+1} (λ = {m.eigL[i]:.3f})", f"eigvec_left_{i+1}")

def run_seba_grid(m, d, seba_module, max_k=11, max_r=20):
    
    output_seba_dir = f"/southern/shaipollak/parcels_analysis/data/{simulation_date_and_time}/T={T}_res={spatial_resolution}/lagrangian_geography/SEBA"
    os.makedirs(output_seba_dir, exist_ok=True)

    for k in range(2, max_k + 1):  # number of clusters you want to extract
        k_dir = os.path.join(output_seba_dir, f"k={k}")
        os.makedirs(k_dir, exist_ok=True)

        for r in range(k + 1, max_r + 1):  # number of eigenvectors to use
            print(f"\n[SEBA] Running for r={r}, k={k}")

            eigvecs = m.R[:, 1:r + 1]  # Skip stationary (index 0), take r vectors
            S = seba_module.seba(eigvecs)
            
            _, A = seba_module.max_likelihood(S, k=k)
            
            print(f"Unique provinces: {np.unique(A)}")

            # Save all plots in the k_dir (no per-r folder)
            plot_seba_labels(A, d, fname=os.path.join(k_dir, f"seba_labels_r{r}"), title=f"SEBA Provinces (r={r}, k={k})")
            plot_max_likelihood_provinces(A, d, fname=os.path.join(k_dir, f"seba_max_likelihood_r{r}_k{k}"), title=f"SEBA Max Likelihood (r={r}, k={k})")

            # Optional: plot time maps to provinces, etc. here


run_seba_grid(m, d, seba_module, max_k=11, max_r=15)

#create_k_means_and_eigenvectors_graphs()
#create_eigenvalue_certainty_map(m, m.eigR) #m is already computed with left_and_right_eigenvectors(n=100)

