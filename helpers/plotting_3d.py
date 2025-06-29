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


def plot_event(index, 
               Tcluster_spacepoints=None, Trec_spacepoints=None, TrueEDep_spacepoints=None, true_gamma1_EDep_spacepoints=None, 
               true_gamma2_EDep_spacepoints=None, other_particles_EDep_spacepoints=None,
               downsampled_Tcluster_spacepoints=None, downsampled_Trec_spacepoints=None, downsampled_TrueEDep_spacepoints=None, 
               downsampled_true_gamma1_EDep_spacepoints=None, downsampled_true_gamma2_EDep_spacepoints=None, downsampled_other_particles_EDep_spacepoints=None,
               real_nu_reco_nu_downsampled_spacepoints=None, real_nu_reco_cosmic_downsampled_spacepoints=None, real_cosmic_reco_nu_downsampled_spacepoints=None, real_cosmic_reco_cosmic_downsampled_spacepoints=None, 
               real_gamma1_downsampled_spacepoints=None, real_gamma2_downsampled_spacepoints=None, real_other_particles_downsampled_spacepoints=None, real_cosmic_downsampled_spacepoints=None,
               reco_nu_vtx=None, true_nu_vtx=None,
               true_gamma1_pairconvert_vtx=None, true_gamma2_pairconvert_vtx=None):

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

    if reco_nu_vtx is not None:
        fig.add_trace(go.Scatter3d(
            x=[reco_nu_vtx[index][2]],
            y=[reco_nu_vtx[index][0]],
            z=[reco_nu_vtx[index][1]],
            mode='markers',
            marker=dict(size=10, color='purple', opacity=1),
            name='Reco Neutrino Vertex',
            visible='legendonly'
        ))

    if true_nu_vtx is not None:
        fig.add_trace(go.Scatter3d(
            x=[true_nu_vtx[index][2]],
            y=[true_nu_vtx[index][0]],
            z=[true_nu_vtx[index][1]],
            mode='markers',
            marker=dict(size=10, color='green', opacity=1),
            name='True Neutrino Vertex',
            visible='legendonly'
        ))

    if true_gamma1_pairconvert_vtx[index] is not None:
        fig.add_trace(go.Scatter3d(
            x=[true_gamma1_pairconvert_vtx[index][2]],
            y=[true_gamma1_pairconvert_vtx[index][0]],
            z=[true_gamma1_pairconvert_vtx[index][1]],
            mode='markers',
            marker=dict(size=10, color='lightgreen', opacity=1),
            name='True Gamma 1 Pair Conversion Vertex',
        ))

    if true_gamma2_pairconvert_vtx[index] is not None:
        fig.add_trace(go.Scatter3d(
            x=[true_gamma2_pairconvert_vtx[index][2]],
            y=[true_gamma2_pairconvert_vtx[index][0]],
            z=[true_gamma2_pairconvert_vtx[index][1]],
            mode='markers',
            marker=dict(size=10, color='green', opacity=1),
            name='True Gamma 2 Pair Conversion Vertex',
        ))


    if Tcluster_spacepoints is not None:
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

    if downsampled_Tcluster_spacepoints is not None:
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

    if Trec_spacepoints is not None:
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

    if downsampled_Trec_spacepoints is not None:
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

    if TrueEDep_spacepoints is not None:
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

    if true_gamma1_EDep_spacepoints is not None:
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

    if true_gamma2_EDep_spacepoints is not None:
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

    if other_particles_EDep_spacepoints is not None:
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

    if downsampled_TrueEDep_spacepoints is not None:
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

    if real_nu_reco_nu_downsampled_spacepoints is not None:
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

    if real_nu_reco_cosmic_downsampled_spacepoints is not None:
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

    if real_cosmic_reco_nu_downsampled_spacepoints is not None:
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

    if real_cosmic_reco_cosmic_downsampled_spacepoints is not None:
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

    if real_gamma1_downsampled_spacepoints is not None:
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

    if real_gamma2_downsampled_spacepoints is not None:
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

    if real_other_particles_downsampled_spacepoints is not None:
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

    if real_cosmic_downsampled_spacepoints is not None:
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

