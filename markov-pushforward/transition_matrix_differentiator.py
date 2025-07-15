import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class MatrixComparator:
    def __init__(self, matrix_a, matrix_b):
        """
        Initialize the MatrixComparator with two matrices.

        Args:
            matrix_a (numpy.ndarray): First matrix.
            matrix_b (numpy.ndarray): Second matrix.
        """
        self.matrix_a = matrix_a
        self.matrix_b = matrix_b

    def truncate_to_overlap(self):
        """
        Truncate the matrices to their overlapping region.

        Returns:
            Truncated matrices (matrix_a, matrix_b).
        """
        rows = min(self.matrix_a.shape[0], self.matrix_b.shape[0])
        cols = min(self.matrix_a.shape[1], self.matrix_b.shape[1])
        return self.matrix_a[:rows, :cols], self.matrix_b[:rows, :cols]

    def pad_to_match(self):
        """
        Pad the smaller matrix to match the size of the larger matrix.

        Returns:
            Padded matrices (matrix_a, matrix_b).
        """
        max_rows = max(self.matrix_a.shape[0], self.matrix_b.shape[0])
        max_cols = max(self.matrix_a.shape[1], self.matrix_b.shape[1])

        padded_a = np.zeros((max_rows, max_cols))
        padded_b = np.zeros((max_rows, max_cols))

        padded_a[:self.matrix_a.shape[0], :self.matrix_a.shape[1]] = self.matrix_a
        padded_b[:self.matrix_b.shape[0], :self.matrix_b.shape[1]] = self.matrix_b

        return padded_a, padded_b

    def frobenius_difference(self, use_overlap=True):
        """
        Compute the Frobenius norm of the difference between the matrices.

        Args:
            use_overlap (bool): Whether to use the truncated overlapping region.

        Returns:
            Frobenius norm of the difference.
        """
        if use_overlap:
            a, b = self.truncate_to_overlap()
        else:
            a, b = self.pad_to_match()
        return np.linalg.norm(a - b, ord='fro')

    def difference_statistics(self, use_overlap=True):
        """
        Compute statistics of the element-wise differences.

        Args:
            use_overlap (bool): Whether to use the truncated overlapping region.

        Returns:
            dict: Statistics of differences (mean, std, max, min).
        """
        if use_overlap:
            a, b = self.truncate_to_overlap()
        else:
            a, b = self.pad_to_match()

        diff = a - b
        return {
            "mean_difference": np.mean(diff),
            "std_difference": np.std(diff),
            "max_difference": np.max(diff),
            "min_difference": np.min(diff),
        }

    def plot_difference_heatmap(self, use_overlap=True):
        """
        Plot a heatmap of the differences between the matrices.

        Args:
            use_overlap (bool): Whether to use the truncated overlapping region.
        """
        if use_overlap:
            a, b = self.truncate_to_overlap()
            title_suffix = " (Overlapping Region)"
        else:
            a, b = self.pad_to_match()
            title_suffix = " (Padded Matrices)"

        diff = a - b
        plt.figure(figsize=(10, 8))
        sns.heatmap(diff, cmap="coolwarm", center=0, cbar=True)
        plt.title(f"Difference Heatmap{title_suffix}")
        plt.show()

    def plot_absolute_difference_heatmap(self, use_overlap=True):
        """
        Plot a heatmap of the absolute differences between the matrices.

        Args:
            use_overlap (bool): Whether to use the truncated overlapping region.
        """
        if use_overlap:
            a, b = self.truncate_to_overlap()
            title_suffix = " (Overlapping Region)"
        else:
            a, b = self.pad_to_match()
            title_suffix = " (Padded Matrices)"

        abs_diff = np.abs(a - b)
        plt.figure(figsize=(10, 8))
        sns.heatmap(abs_diff, cmap="viridis", cbar=True)
        plt.title(f"Absolute Difference Heatmap{title_suffix}")
        plt.show()

    def plot_difference_histogram(self, use_overlap=True):
        """
        Plot a histogram of the element-wise differences between the matrices.

        Args:
            use_overlap (bool): Whether to use the truncated overlapping region.
        """
        if use_overlap:
            a, b = self.truncate_to_overlap()
        else:
            a, b = self.pad_to_match()

        diff = a - b
        plt.figure(figsize=(8, 6))
        plt.hist(diff.flatten(), bins=50, color='blue', alpha=0.7)
        plt.title("Histogram of Differences")
        plt.xlabel("Difference")
        plt.ylabel("Frequency")
        plt.show()

    def visualize_matrices(self):
        """
        Visualize the two matrices side by side to highlight size differences.
        """
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        sns.heatmap(self.matrix_a, cmap="viridis", cbar=True)
        plt.title("Matrix A")

        plt.subplot(1, 2, 2)
        sns.heatmap(self.matrix_b, cmap="viridis", cbar=True)
        plt.title("Matrix B")
        plt.show()

    def visualize_transition_matrix(self, matrix, max_size=1000):
        """
        Visualize a transition matrix by downsampling if it's too large.

        Args:
            matrix (numpy.ndarray): The transition matrix to visualize.
            max_size (int): Maximum size (rows or columns) for visualization.
        """
        rows, cols = matrix.shape

        if rows > max_size or cols > max_size:
            row_indices = np.linspace(0, rows - 1, max_size, dtype=int)
            col_indices = np.linspace(0, cols - 1, max_size, dtype=int)
            reduced_matrix = matrix[np.ix_(row_indices, col_indices)]
            title_suffix = " (Downsampled)"
        else:
            reduced_matrix = matrix
            title_suffix = ""

        plt.figure(figsize=(10, 8))
        sns.heatmap(reduced_matrix, cmap="viridis", cbar=True)
        plt.title(f"Transition Matrix Visualization{title_suffix}")
        plt.xlabel("Columns")
        plt.ylabel("Rows")
        plt.show()

# Example Usage
# matrix_a = np.random.rand(10, 8)
# matrix_b = np.random.rand(8, 10)
# comparator = MatrixComparator(matrix_a, matrix_b)
# print(comparator.frobenius_difference())
# print(comparator.difference_statistics())
# comparator.plot_difference_heatmap()
# comparator.visualize_matrices()
# comparator.visualize_transition_matrix(matrix_a)
