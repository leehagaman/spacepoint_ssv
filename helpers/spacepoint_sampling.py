import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.neighbors import NearestNeighbors, KernelDensity

def fps_sampling(points, n_samples, rng=None):
    """
    Perform an optimized Farthest Point Sampling (FPS) using NumPy.

    :param points_maybe_with_charge: numpy array of shape (N, 3)
    :param n_samples: number of points to sample
    :param rng: random number generator instance
    :return: indices of sampled points
    """
    if rng is None:
        rng = np.random.default_rng()
        
    N = points.shape[0]
    if N == 0: return np.empty((0, 3))
    sampled_indices = np.zeros(n_samples, dtype=int)
    distances = np.full(N, np.inf)

    # Choose the first point randomly
    sampled_indices[0] = rng.integers(N)
    
    # Efficiently compute distances using vectorized operations
    for i in range(1, n_samples):
        # Update minimum distances
        diff = points - points[sampled_indices[i - 1]]
        new_distances = np.einsum('ij,ij->i', diff, diff)  # Faster squared Euclidean distance
        distances = np.minimum(distances, new_distances)
        
        # Select the point farthest from the existing sampled set
        sampled_indices[i] = np.argmax(distances)

    sampled_points = points[sampled_indices]

    return sampled_points


def fps_clustering_downsample(points_maybe_with_charge, n_samples, rng=None):
    """
    Downsample the point cloud using FPS and clustering.
    
    :param points_maybe_with_charge: numpy array of shape (N, 3) or (N, 4) representing the point cloud, maybe including charge
    :param n_samples: number of points in the downsampled cloud
    :param rng: random number generator instance
    :return: downsampled point cloud
    """

    if points_maybe_with_charge.shape[1] == 3: # just 3D points
        points = points_maybe_with_charge
        point_charges = None
    elif points_maybe_with_charge.shape[1] == 4: # 3D points with charge
        points = points_maybe_with_charge[:, :3]
        point_charges = points_maybe_with_charge[:, 3]
    else:
        raise ValueError(f"Points must be of shape (3,) or (4,), got {points_maybe_with_charge.shape}")

    if rng is None:
        rng = np.random.default_rng()

    if len(points) == 0 or n_samples == 0: return np.empty((0, 3))

    # Remove duplicate points to avoid KMeans convergence issues
    rounded_points = np.round(points, decimals=5)
    unique_indices = np.unique(rounded_points, axis=0, return_index=True)[1]
    points = points[unique_indices]

    if len(points) < n_samples:
        n_samples = len(points)

    # Perform FPS to get initial samples
    sampled_points = fps_sampling(points, n_samples, rng)

    # Use K-means clustering to associate other points with the samples
    kmeans = KMeans(n_clusters=n_samples, init=sampled_points, n_init=1)
    kmeans.fit(points)

    downsampled_points = kmeans.cluster_centers_

    if point_charges is None:
        return downsampled_points

    #Find the nearest downsampled point for each original point
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(downsampled_points)
    _, indices = nbrs.kneighbors(points)
    
    # Initialize charge sums array
    charge_sums = np.zeros(len(sampled_points))
    
    # Sum charges for each downsampled point
    for i, nearest_idx in enumerate(indices.flatten()):
        charge_sums[nearest_idx] += point_charges[i]

    downsampled_points_with_charge = np.concatenate([downsampled_points, charge_sums[:, np.newaxis]], axis=1)
        
    return downsampled_points_with_charge

def get_min_dists(points_A, points_B):
    """
    Get the minimum distance between each point in points_A and all the points in points_B.
    """

    if len(points_A) == 0 or len(points_B) == 0:
        return np.array([])

    nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(points_B)
    distances, _ = nbrs.kneighbors(points_A)
    return distances


def remove_outliers(points, min_neighbors=5, radius=None, k=10):
    """
    Remove outliers from point cloud based on local density.
    
    Args:
        points: numpy array of shape (N, D) where N is number of points, D is dimensionality
        min_neighbors: minimum number of neighbors required to keep a point
        radius: radius for neighborhood search. If None, use k-nearest neighbors approach
        k: number of neighbors to consider if radius is None
        
    Returns:
        Filtered point cloud without outliers
    """
    # Choose neighborhood calculation approach
    if radius is not None:
        # Radius-based approach
        nbrs = NearestNeighbors(radius=radius).fit(points)
        distances, indices = nbrs.radius_neighbors(points)
        # Count neighbors for each point (excluding the point itself)
        neighbor_count = np.array([len(idx) - 1 for idx in indices])
    else:
        # k-nearest neighbors approach
        nbrs = NearestNeighbors(n_neighbors=k+1).fit(points)  # +1 because point is its own neighbor
        distances, indices = nbrs.kneighbors(points)
        # Calculate average distance to k nearest neighbors
        avg_distances = np.mean(distances[:, 1:], axis=1)  # Exclude self (first neighbor)
        # Points with large average distances are outliers
        threshold = np.mean(avg_distances) + 2 * np.std(avg_distances)
        neighbor_count = np.where(avg_distances <= threshold, k, 0)
    
    # Keep points with sufficient neighbors
    mask = neighbor_count >= min_neighbors
    filtered_points = points[mask]
    
    return filtered_points



def energy_weighted_density_sampling(points, energies, n_samples=1000, rng=None):

    if rng is None:
        rng = np.random.default_rng()

    if len(points) == 0 or n_samples == 0: return np.empty((0, 3))

    if n_samples > len(points):
        n_samples = len(points)

    # Sample directly proportional to energy values
    # Normalize energies to use as probabilities
    probs = energies / energies.sum()
    
    # Sample based on energy values
    sampled_indices = rng.choice(len(points), n_samples, replace=False, p=probs)
    
    sampled_points = points[sampled_indices]
    
    return sampled_points
