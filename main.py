from sklearn.decomposition import PCA
import numpy as np
from sklearn.neighbors import KernelDensity
from scipy.ndimage import maximum_filter
from scipy.spatial.distance import cdist
from imblearn.datasets import fetch_datasets

class Undersampler:
    def __init__(self, bandwidth=0.1, grid_size=100, top_percentage=5):
        """
        Args:
            bandwidth (float): Bandwidth for the Gaussian kernel. Width of the kernel.
                - A smaller bandwidth value will result in a more sensitive density estimation.
                - A larger bandwidth value will result in a smoother density estimation.
            grid_size (int): Size of the grid for density evaluation.
            top_percentage (float): Percentage of closest prototypes to find (e.g., 10 for top 10%).
        """
        self.bandwidth = bandwidth
        self.grid_size = grid_size
        self.top_percentage = top_percentage


    def fit(self, X, y):
        """
        Fit the undersampling process on the given data.

        Args:
            X (np.ndarray): The dataset to perform undersampling on.
            y (np.ndarray): Target labels (optional). If provided, will be used for splitting into classes.
        
        Returns:
            The undersampled dataset.
        """

        self.X = X
        self.y = y

        # Perform PCA
        self.X_pca, self.pca = self.perform_pca(X)

        # Identify majority and minority classes based on class distribution
        majority_class, minority_class = self.get_majority_minority_classes(self.y)

        #  Split data into majority and minority classes
        X_majority_pca, X_minority_pca = self.split_by_class(self.X_pca, self.y, majority_class, minority_class)
        
        # Estimate the density of the majority class using Kernel Density Estimation (KDE).
        density, xx, yy = self.kde_density(X_majority_pca, bandwidth=self.bandwidth, grid_size=self.grid_size)

        # Find high-density centers in the estimated density map.
        self.centers = self.find_high_density_centers(density, xx, yy)
        print(len(self.centers))

        # Identify the closest data points (prototypes) to the high-density centers
        self.closest_prototypes = self.find_closest_prototypes(X_majority_pca, self.centers, self.top_percentage)
        print(len(self.closest_prototypes))

        # Extract the selected prototypes from the majority class
        self.prototypes = self.extract_prototypes(X_majority_pca, self.closest_prototypes)
        
        # Merge the selected prototypes with the minority class to form the undersampled dataset
        self.merged_data, self.merged_labels = self.merge_prototypes_and_minority(X_minority_pca, self.prototypes)
        
        return self.merged_data
    

    def perform_pca(self, X, n_components=2):
        """ Perform PCA to reduce dimensionality. """
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X)
        return X_pca, pca
    
    
    def get_majority_minority_classes(self, y):
        """ Get the majority and minority classes in the dataset. """
        unique, counts = np.unique(y, return_counts=True)
        majority_class = unique[np.argmax(counts)]
        minority_class = unique[np.argmin(counts)]
        return majority_class, minority_class
    
    
    def split_by_class(self, X, y, majority_class, minority_class):
        """Split data by majority and minority classes."""
        X_majority = X[y == majority_class]
        X_minority = X[y == minority_class]
        return X_majority, X_minority
    
    
    def kde_density(self, X, bandwidth=0.1, grid_size=100):
        """
        Compute the KDE density using sklearn's KernelDensity.
        
        Args:
            X (np.ndarray): 2D data array of shape (n_samples, n_features).
            bandwidth (float): Bandwidth for the Gaussian kernel. Width of the kernel.
                - A smaller bandwidth value will result in a more sensitive density estimation.
                - A larger bandwidth value will result in a smoother density estimation.
            grid_size (int): Size of the grid for density evaluation.
        
        Returns:
            density (np.ndarray): 2D array of density values.
            xx (np.ndarray): X-coordinates of the grid.
            yy (np.ndarray): Y-coordinates of the grid.
        """
        kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth)
        kde.fit(X)
        
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, grid_size), np.linspace(y_min, y_max, grid_size))
        grid_points = np.vstack([xx.ravel(), yy.ravel()]).T
        
        log_density = kde.score_samples(grid_points)
        density = np.exp(log_density).reshape(xx.shape)
        
        return density, xx, yy
    
    
    def find_high_density_centers(self, density, xx, yy, num_centers=5, filter_size=5):
        """
        Find centers of high-density regions in a 2D density map.
        
        Args:
            density (np.ndarray): 2D array of density values.
            xx (np.ndarray): X-coordinates of the density grid.
            yy (np.ndarray): Y-coordinates of the density grid.
            num_centers (int): Number of high-density centers to detect.
            filter_size (int): Size of the filter for local maxima detection.
        
        Returns:
            List of (x, y) tuples representing high-density centers.
        """
        local_max = maximum_filter(density, size=filter_size) == density
        maxima_coords = np.argwhere(local_max)
        maxima_values = density[local_max]
        
        sorted_indices = np.argsort(maxima_values)[::-1]
        maxima_coords_sorted = maxima_coords[sorted_indices]

        centers = []
        for i in range(min(num_centers, len(maxima_coords_sorted))):
            row, col = maxima_coords_sorted[i]
            centers.append((xx[0, col], yy[row, 0]))
        
        return centers
    
    
    def find_closest_prototypes(self, X, centers, top_percentage=10):
        """
        Find the top N% closest prototypes to each high-density center.
        
        Args:
            X (np.ndarray): Original dataset of prototypes.
            centers (list): List of (x, y) tuples representing high-density centers.
            top_percentage (float): Percentage of closest prototypes to find (e.g., 10 for top 10%).
        
        Returns:
            dict: A dictionary where keys are center indices, and values are arrays of indices of the closest prototypes.
        """
        num_points = len(X)
        num_to_select = max(1, int((top_percentage / 100) * num_points))

        distances = cdist(X, centers, metric="euclidean")
        closest_prototypes = {}
        
        for center_idx in range(len(centers)):
            center_distances = distances[:, center_idx]
            sorted_indices = np.argsort(center_distances)
            closest_prototypes[center_idx] = sorted_indices[:num_to_select]
        
        return closest_prototypes
    

    def extract_prototypes(self, X_majority_pca, closest_prototypes):
        """Extract prototypes based on closest points to high-density centers."""
        prototypes = np.vstack([X_majority_pca[closest_prototypes[i]] for i in range(len(closest_prototypes))])
        return prototypes
    
    
    def merge_prototypes_and_minority(self, X_minority_pca, prototypes):
        """Merge prototypes with the minority class."""
        merged_data_pca = np.vstack((X_minority_pca, prototypes))
        merged_labels = np.hstack((np.zeros(len(X_minority_pca)), np.ones(len(prototypes))))

        merged_data = self.pca.inverse_transform(merged_data_pca) if self.pca else merged_data_pca

        return merged_data, merged_labels
    

def main():
    dataset = fetch_datasets()['wine_quality']
    X, y = dataset.data, dataset.target
    print(X.shape, y.shape)

    undersampler = Undersampler()
    X_new = undersampler.fit(X, y)
    print(X_new.shape)

    original_size = X.shape[0]
    new_size = X_new.shape[0]
    percentage_reduction = ((original_size - new_size) / original_size) * 100
    print(f"Reduced dataset size by {percentage_reduction:.2f}%")


if __name__ == "__main__":
    main()