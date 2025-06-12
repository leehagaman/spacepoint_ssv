import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.neighbors import NearestNeighbors, KernelDensity


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


def plot_event(index, 
               Tcluster_spacepoints, Trec_spacepoints, TrueEDep_spacepoints, true_gamma1_EDep_spacepoints, 
               true_gamma2_EDep_spacepoints, other_particles_EDep_spacepoints,
               downsampled_Tcluster_spacepoints, downsampled_Trec_spacepoints, downsampled_TrueEDep_spacepoints, 
               downsampled_true_gamma1_EDep_spacepoints, downsampled_true_gamma2_EDep_spacepoints, downsampled_other_particles_EDep_spacepoints,
               real_nu_reco_nu_downsampled_spacepoints, real_nu_reco_cosmic_downsampled_spacepoints, real_cosmic_reco_nu_downsampled_spacepoints, real_cosmic_reco_cosmic_downsampled_spacepoints, 
               real_gamma1_downsampled_spacepoints, real_gamma2_downsampled_spacepoints, real_other_particles_downsampled_spacepoints, real_cosmic_downsampled_spacepoints,
               reco_nu_vtx, true_nu_vtx,
               include_non_downsampled_points=True):

    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    detector_boundary_points = generate_box_edge_points(tpc_min_x, tpc_max_x, tpc_min_y, tpc_max_y, tpc_min_z, tpc_max_z, num_points_per_edge=100)
    x_width = tpc_max_x - tpc_min_x
    expanded_detector_boundary_points = generate_box_edge_points(tpc_min_x - x_width, tpc_max_x + x_width, tpc_min_y, tpc_max_y, tpc_min_z, tpc_max_z, num_points_per_edge=100)


    fig = make_subplots(rows=1, cols=1, specs=[[{'type': 'scene'}]])

    # these are only added to set the camera at a better position
    fig.add_trace(go.Scatter3d(
        x=expanded_detector_boundary_points[:, 2],
        y=expanded_detector_boundary_points[:, 0],
        z=expanded_detector_boundary_points[:, 1],
        mode='markers',
        marker=dict(
            size=0.2,
            color='black',
            opacity=0.8
        ),
        name='Expanded TPC Boundary'
    ))

    fig.add_trace(go.Scatter3d(
        x=detector_boundary_points[:, 2],
        y=detector_boundary_points[:, 0],
        z=detector_boundary_points[:, 1],
        mode='markers',
        marker=dict(
            size=1,
            color='black',
            opacity=0.8
        ),
        name='TPC Boundary'
    ))

    fig.add_trace(go.Scatter3d(
        x=[reco_nu_vtx[index][2]],
        y=[reco_nu_vtx[index][0]],
        z=[reco_nu_vtx[index][1]],
        mode='markers',
        marker=dict(size=10, color='purple', opacity=1),
        name='Reco Neutrino Vertex',
        visible='legendonly'
    ))

    fig.add_trace(go.Scatter3d(
        x=[true_nu_vtx[index][2]],
        y=[true_nu_vtx[index][0]],
        z=[true_nu_vtx[index][1]],
        mode='markers',
        marker=dict(size=10, color='green', opacity=1),
        name='True Neutrino Vertex',
        visible='legendonly'

    ))


    if include_non_downsampled_points:
        fig.add_trace(go.Scatter3d(
            x=Tcluster_spacepoints[index][:, 2],
            y=Tcluster_spacepoints[index][:, 0],
            z=Tcluster_spacepoints[index][:, 1],
            mode='markers',
            marker=dict(
                size=1,
                color="blue",
                opacity=0.8
            ),
            name='Tcluster Spacepoints',
            visible='legendonly'
        ))

    fig.add_trace(go.Scatter3d(
        x=downsampled_Tcluster_spacepoints[index][:, 2],
        y=downsampled_Tcluster_spacepoints[index][:, 0],
        z=downsampled_Tcluster_spacepoints[index][:, 1],
        mode='markers',
        marker=dict(
            size=3,
            color="blue",
            opacity=0.8
        ),
        name='Downsampled Tcluster Spacepoints',
        visible='legendonly'
    ))

    if include_non_downsampled_points:
        fig.add_trace(go.Scatter3d(
            x=Trec_spacepoints[index][:, 2],
            y=Trec_spacepoints[index][:, 0],
            z=Trec_spacepoints[index][:, 1],
            mode='markers',
            marker=dict(
                size=1,
                color='red',
                opacity=0.8
            ),
            name='Trec Spacepoints',
            visible='legendonly'
        ))

    fig.add_trace(go.Scatter3d(
        x=downsampled_Trec_spacepoints[index][:, 2],
        y=downsampled_Trec_spacepoints[index][:, 0],
        z=downsampled_Trec_spacepoints[index][:, 1],
        mode='markers',
        marker=dict(
            size=3,
            color='red',
            opacity=0.8
        ),
        name='Downsampled Trec Spacepoints',
        visible='legendonly'
    ))

    if include_non_downsampled_points:
        fig.add_trace(go.Scatter3d(
            x=TrueEDep_spacepoints[index][:, 2],
            y=TrueEDep_spacepoints[index][:, 0],
            z=TrueEDep_spacepoints[index][:, 1],
            mode='markers',
            marker=dict(
                size=1,
                color='orange',
                opacity=0.8
            ),
            name='TrueEDep Spacepoints',
            visible='legendonly'
        ))

        fig.add_trace(go.Scatter3d(
            x=true_gamma1_EDep_spacepoints[index][:, 2],
            y=true_gamma1_EDep_spacepoints[index][:, 0],
            z=true_gamma1_EDep_spacepoints[index][:, 1],
            mode='markers',
            marker=dict(
                size=1,
                color='lightgreen',
                opacity=0.8
            ),
            name='Real Gamma 1 EDep Spacepoints',
            visible='legendonly'
        ))

        fig.add_trace(go.Scatter3d(
            x=true_gamma2_EDep_spacepoints[index][:, 2],
            y=true_gamma2_EDep_spacepoints[index][:, 0],
            z=true_gamma2_EDep_spacepoints[index][:, 1],
            mode='markers',
            marker=dict(
                size=1,
                color='green',
                opacity=0.8
            ),
            name='Real Gamma 2 EDep Spacepoints',
            visible='legendonly'
        ))

        fig.add_trace(go.Scatter3d(
            x=other_particles_EDep_spacepoints[index][:, 2],
            y=other_particles_EDep_spacepoints[index][:, 0],
            z=other_particles_EDep_spacepoints[index][:, 1],
            mode='markers',
            marker=dict(
                size=1,
                color='brown',
                opacity=0.8
            ),
            name='Real Other Particles EDep Spacepoints',
            visible='legendonly'
        ))

    fig.add_trace(go.Scatter3d(
        x=downsampled_TrueEDep_spacepoints[index][:, 2],
        y=downsampled_TrueEDep_spacepoints[index][:, 0],
        z=downsampled_TrueEDep_spacepoints[index][:, 1],
        mode='markers',
        marker=dict(
            size=3,
            color='orange',
            opacity=0.8
        ),
        name='Downsampled TrueEDep Spacepoints',
        visible='legendonly'
    ))

    fig.add_trace(go.Scatter3d(
        x=real_nu_reco_nu_downsampled_spacepoints[index][:, 2],
        y=real_nu_reco_nu_downsampled_spacepoints[index][:, 0],
        z=real_nu_reco_nu_downsampled_spacepoints[index][:, 1],
        mode='markers',
        marker=dict(
            size=3,
            color='orange',
            opacity=0.8
        ),
        name='Real Nu Reco Nu Spacepoints',
        visible='legendonly'
    ))

    fig.add_trace(go.Scatter3d(
        x=real_nu_reco_cosmic_downsampled_spacepoints[index][:, 2],
        y=real_nu_reco_cosmic_downsampled_spacepoints[index][:, 0],
        z=real_nu_reco_cosmic_downsampled_spacepoints[index][:, 1],
        mode='markers',
        marker=dict(
            size=3,
            color='red',
            opacity=0.8
        ),
        name='Real Nu Reco Cosmic Spacepoints',
        visible='legendonly'
    ))

    fig.add_trace(go.Scatter3d(
        x=real_cosmic_reco_nu_downsampled_spacepoints[index][:, 2],
        y=real_cosmic_reco_nu_downsampled_spacepoints[index][:, 0],
        z=real_cosmic_reco_nu_downsampled_spacepoints[index][:, 1],
        mode='markers',
        marker=dict(
            size=3,
            color='brown',
            opacity=0.8
        ),
        name='Real Cosmic Reco Nu Spacepoints',
        visible='legendonly'
    ))

    fig.add_trace(go.Scatter3d(
        x=real_cosmic_reco_cosmic_downsampled_spacepoints[index][:, 2],
        y=real_cosmic_reco_cosmic_downsampled_spacepoints[index][:, 0],
        z=real_cosmic_reco_cosmic_downsampled_spacepoints[index][:, 1],
        mode='markers',
        marker=dict(
            size=3,
            color='blue',
            opacity=0.8
        ),
        name='Real Cosmic Reco Cosmic Spacepoints',
        visible='legendonly'
    ))

    fig.add_trace(go.Scatter3d(
        x=real_gamma1_downsampled_spacepoints[index][:, 2],
        y=real_gamma1_downsampled_spacepoints[index][:, 0],
        z=real_gamma1_downsampled_spacepoints[index][:, 1],
        mode='markers',
        marker=dict(
            size=3, 
            color='lightgreen',
            opacity=0.8
        ),
        name='Real Gamma 1 Downsampled Spacepoints',
    ))

    fig.add_trace(go.Scatter3d(
        x=real_gamma2_downsampled_spacepoints[index][:, 2],
        y=real_gamma2_downsampled_spacepoints[index][:, 0],
        z=real_gamma2_downsampled_spacepoints[index][:, 1],
        mode='markers',
        marker=dict(
            size=3,
            color='green',
            opacity=0.8
        ),
        name='Real Gamma 2 Downsampled Spacepoints',
    ))

    fig.add_trace(go.Scatter3d(
        x=real_other_particles_downsampled_spacepoints[index][:, 2],
        y=real_other_particles_downsampled_spacepoints[index][:, 0],
        z=real_other_particles_downsampled_spacepoints[index][:, 1],
        mode='markers',
        marker=dict(
            size=3,
            color='brown',
            opacity=0.8
        ),
        name='Real Other Particles Downsampled Spacepoints',
    ))

    fig.add_trace(go.Scatter3d(
        x=real_cosmic_downsampled_spacepoints[index][:, 2],
        y=real_cosmic_downsampled_spacepoints[index][:, 0],
        z=real_cosmic_downsampled_spacepoints[index][:, 1],
        mode='markers',
        marker=dict(
            size=3,
            color='blue',
            opacity=0.8
        ),
        name='Real Cosmic Downsampled Spacepoints',
    ))

    fig.update_layout(
        scene=dict(
            xaxis_title='z',
            yaxis_title='x',
            zaxis_title='y',
            aspectratio=dict(
                x=5,
                y=3,
                z=1
            ),
        ),
        width=2000,
        height=1200,
        autosize=False,
        scene_camera=dict(
            eye=dict(x=-1.5, y=-1.5, z=1.5)
        )
    )

    fig.show(renderer="browser")


def fps_sampling(points, n_samples):
    """
    Perform an optimized Farthest Point Sampling (FPS) using NumPy.

    :param points: numpy array of shape (N, 3) representing the point cloud
    :param n_samples: number of points to sample
    :return: indices of sampled points
    """
    N = points.shape[0]
    if N == 0: return np.empty((0, 3))
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

    if len(points) == 0 or n_samples == 0: return np.empty((0, 3))

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



def energy_weighted_density_sampling(points, energies, n_samples=1000):

    if len(points) == 0 or n_samples == 0: return np.empty((0, 3))

    if n_samples > len(points):
        n_samples = len(points)

    # Sample directly proportional to energy values
    # Normalize energies to use as probabilities
    probs = energies / energies.sum()
    
    # Sample based on energy values
    sampled_indices = np.random.choice(len(points), n_samples, replace=False, p=probs)
    
    sampled_points = points[sampled_indices]
    
    return sampled_points
