import numpy as np

# making detector boundary points

tpc_min_x = -1.
tpc_max_x = 254.3
tpc_min_y = -115.
tpc_max_y = 117.
tpc_min_z = 0.6
tpc_max_z = 1036.4

def generate_box_edge_points(x_min, x_max, y_min, y_max, z_min, z_max, num_points_per_edge=10):

    # Generate points along edges parallel to X-axis
    t = np.linspace(0, 1, num_points_per_edge)
    x_edges = [
        np.column_stack([
            x_min + t * (x_max - x_min),
            np.full_like(t, y_min),
            np.full_like(t, z_min)
        ]),
        np.column_stack([
            x_min + t * (x_max - x_min),
            np.full_like(t, y_max),
            np.full_like(t, z_min)
        ]),
        np.column_stack([
            x_min + t * (x_max - x_min),
            np.full_like(t, y_min),
            np.full_like(t, z_max)
        ]),
        np.column_stack([
            x_min + t * (x_max - x_min),
            np.full_like(t, y_max),
            np.full_like(t, z_max)
        ])
    ]
    
    # Generate points along edges parallel to Y-axis
    y_edges = [
        np.column_stack([
            np.full_like(t, x_min),
            y_min + t * (y_max - y_min),
            np.full_like(t, z_min)
        ]),
        np.column_stack([
            np.full_like(t, x_max),
            y_min + t * (y_max - y_min),
            np.full_like(t, z_min)
        ]),
        np.column_stack([
            np.full_like(t, x_min),
            y_min + t * (y_max - y_min),
            np.full_like(t, z_max)
        ]),
        np.column_stack([
            np.full_like(t, x_max),
            y_min + t * (y_max - y_min),
            np.full_like(t, z_max)
        ])
    ]
    
    # Generate points along edges parallel to Z-axis
    z_edges = [
        np.column_stack([
            np.full_like(t, x_min),
            np.full_like(t, y_min),
            z_min + t * (z_max - z_min)
        ]),
        np.column_stack([
            np.full_like(t, x_max),
            np.full_like(t, y_min),
            z_min + t * (z_max - z_min)
        ]),
        np.column_stack([
            np.full_like(t, x_min),
            np.full_like(t, y_max),
            z_min + t * (z_max - z_min)
        ]),
        np.column_stack([
            np.full_like(t, x_max),
            np.full_like(t, y_max),
            z_min + t * (z_max - z_min)
        ])
    ]
    
    # Combine all edges
    all_points = np.vstack(x_edges + y_edges + z_edges)
    return all_points


detector_boundary_points = generate_box_edge_points(tpc_min_x, tpc_max_x, tpc_min_y, tpc_max_y, tpc_min_z, tpc_max_z, num_points_per_edge=100)
x_width = tpc_max_x - tpc_min_x
expanded_detector_boundary_points = generate_box_edge_points(tpc_min_x - x_width, tpc_max_x + x_width, tpc_min_y, tpc_max_y, tpc_min_z, tpc_max_z, num_points_per_edge=100)


from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

def fps_sampling(points, n_samples):
    """
    Perform an optimized Farthest Point Sampling (FPS) using NumPy.

    :param points: numpy array of shape (N, 3) representing the point cloud
    :param n_samples: number of points to sample
    :return: indices of sampled points
    """
    N = points.shape[0]
    if N == 0: return np.array([])
    sampled_indices = np.zeros(n_samples, dtype=int)
    distances = np.full(N, np.inf)

    # Choose the first point randomly
    sampled_indices[0] = np.random.randint(N)
    
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

def fps_clustering_downsample(points, n_samples, debug=False):
    """
    Downsample the point cloud using FPS and clustering.
    
    :param points: numpy array of shape (N, 3) representing the point cloud
    :param n_samples: number of points in the downsampled cloud
    :return: downsampled point cloud
    """

    if len(points) == 0 or n_samples == 0: return np.array([])

    if debug:
        print(f"downsampling {points.shape[0]} points to {n_samples} points")

    if debug:
        print(f"performing FPS...", end="", flush=True)

    # Perform FPS to get initial samples
    sampled_points = fps_sampling(points, n_samples)
    
    if debug:
        print(f"done", flush=True)

    if debug:
        print(f"performing KMeans...", end="", flush=True)

    # Use K-means clustering to associate other points with the samples
    kmeans = KMeans(n_clusters=n_samples, init=sampled_points, n_init=1)
    kmeans.fit(points)

    if debug:
        print(f"done", flush=True)
        
    return kmeans.cluster_centers_

def get_min_dists(points_A, points_B):
    """
    Get the minimum distance between each point in points_A and all the points in points_B.
    """

    if len(points_A) == 0 or len(points_B) == 0:
        return np.array([])

    nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(points_B)
    distances, _ = nbrs.kneighbors(points_A)
    return distances

