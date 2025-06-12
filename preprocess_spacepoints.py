import sys
import uproot
import numpy as np
from tqdm import tqdm
import pickle
import argparse

from plot_3d_helpers import fps_clustering_downsample, get_min_dists, energy_weighted_density_sampling

def get_basic_info(f):
    # loads the basic information from the root file

    rse = f["wcpselection"]["T_eval"].arrays(["run", "subrun", "event"], library="np")
    true_nu_vtx = f["wcpselection"]["T_eval"].arrays(["truth_vtxX", "truth_vtxY", "truth_vtxZ"], library="np")
    true_nu_vtx = np.stack([true_nu_vtx["truth_vtxX"], true_nu_vtx["truth_vtxY"], true_nu_vtx["truth_vtxZ"]], axis=-1)
    reco_nu_vtx = f["wcpselection"]["T_PFeval"].arrays(["reco_nuvtxX", "reco_nuvtxY", "reco_nuvtxZ"], library="np")
    reco_nu_vtx = np.stack([reco_nu_vtx["reco_nuvtxX"], reco_nu_vtx["reco_nuvtxY"], reco_nu_vtx["reco_nuvtxZ"]], axis=-1)

    return rse, true_nu_vtx, reco_nu_vtx


def get_geant_points(f, num_interpolated_points=5, num_events=None):

    print("getting Geant4 points from Wire-Cell PF tree")

    wc_geant_dic = f["wcpselection"]["T_PFeval"].arrays(["truth_id", "truth_mother", "truth_pdg", "truth_startMomentum", "truth_startXYZT", "truth_endXYZT"], library="np")

    true_gamma_1_geant_points = []
    true_gamma_2_geant_points = []
    other_particles_geant_points = []

    if num_events is None:
        num_events = len(wc_geant_dic["truth_id"])

    for event_i in tqdm(range(num_events)):

        curr_true_gamma_1_geant_points = []
        curr_true_gamma_2_geant_points = []
        curr_other_particles_geant_points = []

        # finding the primary pi0
        primary_pi0_id = -1
        for i in range(len(wc_geant_dic["truth_id"][event_i])):
            if wc_geant_dic["truth_pdg"][event_i][i] == 111 and wc_geant_dic["truth_mother"][event_i][i] == 0:
                primary_pi0_id = wc_geant_dic["truth_id"][event_i][i]
                break
        if primary_pi0_id == -1: # no primary pi0 found, skip
            true_gamma_1_geant_points.append(np.array([]))
            true_gamma_2_geant_points.append(np.array([]))
            other_particles_geant_points.append(np.array([]))
            continue

        # finding the daughters of the primary pi0
        daughters_of_primary_pi0_ids = []
        daughters_of_primary_pi0_pdgs = []
        for i in range(len(wc_geant_dic["truth_id"][event_i])):
            if wc_geant_dic["truth_mother"][event_i][i] == primary_pi0_id:
                daughters_of_primary_pi0_ids.append(wc_geant_dic["truth_id"][event_i][i])
                daughters_of_primary_pi0_pdgs.append(wc_geant_dic["truth_pdg"][event_i][i])
        if not (len(daughters_of_primary_pi0_pdgs) == 2 and daughters_of_primary_pi0_pdgs[0] == 22 and daughters_of_primary_pi0_pdgs[1] == 22):
            true_gamma_1_geant_points.append(np.array([]))
            true_gamma_2_geant_points.append(np.array([]))
            other_particles_geant_points.append(np.array([]))
            continue

        # finding all descendants of the first daughter of the primary pi0
        first_daughter_and_descendants_ids = [daughters_of_primary_pi0_ids[0]]
        first_daughter_descendants_pdgs = []
        first_daughter_descendants_energies = []
        first_daughter_descendants_start_xs = []
        first_daughter_descendants_start_ys = []
        first_daughter_descendants_start_zs = []
        first_daughter_descendants_end_xs = []
        first_daughter_descendants_end_ys = []
        first_daughter_descendants_end_zs = []
        num_added = 1
        while num_added != 0:
            num_added = 0
            for i in range(len(wc_geant_dic["truth_id"][event_i])):
                if wc_geant_dic["truth_mother"][event_i][i] in first_daughter_and_descendants_ids and wc_geant_dic["truth_id"][event_i][i] not in first_daughter_and_descendants_ids:
                    first_daughter_and_descendants_ids.append(wc_geant_dic["truth_id"][event_i][i])
                    first_daughter_descendants_pdgs.append(wc_geant_dic["truth_pdg"][event_i][i])
                    first_daughter_descendants_energies.append(wc_geant_dic["truth_startMomentum"][event_i][i][3])
                    first_daughter_descendants_start_xs.append(wc_geant_dic["truth_startXYZT"][event_i][i][0])
                    first_daughter_descendants_start_ys.append(wc_geant_dic["truth_startXYZT"][event_i][i][1])
                    first_daughter_descendants_start_zs.append(wc_geant_dic["truth_startXYZT"][event_i][i][2])
                    first_daughter_descendants_end_xs.append(wc_geant_dic["truth_endXYZT"][event_i][i][0])
                    first_daughter_descendants_end_ys.append(wc_geant_dic["truth_endXYZT"][event_i][i][1])
                    first_daughter_descendants_end_zs.append(wc_geant_dic["truth_endXYZT"][event_i][i][2])
                    num_added += 1

        # finding the descendants of the second daughter of the primary pi0
        second_daughter_and_descendants_ids = [daughters_of_primary_pi0_ids[1]]
        second_daughter_descendants_pdgs = []
        second_daughter_descendants_energies = []
        second_daughter_descendants_start_xs = []
        second_daughter_descendants_start_ys = []
        second_daughter_descendants_start_zs = []
        second_daughter_descendants_end_xs = []
        second_daughter_descendants_end_ys = []
        second_daughter_descendants_end_zs = []
        num_added = 1
        while num_added != 0:
            num_added = 0
            for i in range(len(wc_geant_dic["truth_id"][event_i])):
                if wc_geant_dic["truth_mother"][event_i][i] in second_daughter_and_descendants_ids and wc_geant_dic["truth_id"][event_i][i] not in second_daughter_and_descendants_ids:
                    second_daughter_and_descendants_ids.append(wc_geant_dic["truth_id"][event_i][i])
                    second_daughter_descendants_pdgs.append(wc_geant_dic["truth_pdg"][event_i][i])
                    second_daughter_descendants_energies.append(wc_geant_dic["truth_startMomentum"][event_i][i][3])
                    second_daughter_descendants_start_xs.append(wc_geant_dic["truth_startXYZT"][event_i][i][0])
                    second_daughter_descendants_start_ys.append(wc_geant_dic["truth_startXYZT"][event_i][i][1])
                    second_daughter_descendants_start_zs.append(wc_geant_dic["truth_startXYZT"][event_i][i][2])
                    second_daughter_descendants_end_xs.append(wc_geant_dic["truth_endXYZT"][event_i][i][0])
                    second_daughter_descendants_end_ys.append(wc_geant_dic["truth_endXYZT"][event_i][i][1])
                    second_daughter_descendants_end_zs.append(wc_geant_dic["truth_endXYZT"][event_i][i][2])
                    num_added += 1

        other_particles_pdgs = []
        other_particles_energies = []
        other_particles_start_xs = []
        other_particles_start_ys = []
        other_particles_start_zs = []
        other_particles_end_xs = []
        other_particles_end_ys = []
        other_particles_end_zs = []
        for i in range(len(wc_geant_dic["truth_id"][event_i])):
            if wc_geant_dic["truth_id"][event_i][i] not in first_daughter_and_descendants_ids and wc_geant_dic["truth_id"][event_i][i] not in second_daughter_and_descendants_ids:
                other_particles_pdgs.append(wc_geant_dic["truth_pdg"][event_i][i])
                other_particles_energies.append(wc_geant_dic["truth_startMomentum"][event_i][i][3])
                other_particles_start_xs.append(wc_geant_dic["truth_startXYZT"][event_i][i][0])
                other_particles_start_ys.append(wc_geant_dic["truth_startXYZT"][event_i][i][1])
                other_particles_start_zs.append(wc_geant_dic["truth_startXYZT"][event_i][i][2])
                other_particles_end_xs.append(wc_geant_dic["truth_endXYZT"][event_i][i][0])
                other_particles_end_ys.append(wc_geant_dic["truth_endXYZT"][event_i][i][1])
                other_particles_end_zs.append(wc_geant_dic["truth_endXYZT"][event_i][i][2])

        # interpolate between the start and end points
        for particle_i in range(len(first_daughter_descendants_start_xs)):
            if first_daughter_descendants_pdgs[particle_i] == 22 or first_daughter_descendants_pdgs[particle_i] == 2112:
                # neutral particle, no edep, don't include in the geant 3D points for purpose of categorizing the edeps
                continue
            for i in range(num_interpolated_points + 1):
                curr_true_gamma_1_geant_points.append(np.array([
                    first_daughter_descendants_start_xs[particle_i] + (first_daughter_descendants_end_xs[particle_i] - first_daughter_descendants_start_xs[particle_i]) * (i + 1) / (num_interpolated_points + 1), 
                    first_daughter_descendants_start_ys[particle_i] + (first_daughter_descendants_end_ys[particle_i] - first_daughter_descendants_start_ys[particle_i]) * (i + 1) / (num_interpolated_points + 1), 
                    first_daughter_descendants_start_zs[particle_i] + (first_daughter_descendants_end_zs[particle_i] - first_daughter_descendants_start_zs[particle_i]) * (i + 1) / (num_interpolated_points + 1), 
                ]))
        if len(curr_true_gamma_1_geant_points) == 0:
            true_gamma_1_geant_points.append(np.array([]))
        else:
            curr_true_gamma_1_geant_points = np.stack(curr_true_gamma_1_geant_points, axis=0)
            true_gamma_1_geant_points.append(curr_true_gamma_1_geant_points)

        for particle_i in range(len(second_daughter_descendants_start_xs)):
            if second_daughter_descendants_pdgs[particle_i] == 22 or second_daughter_descendants_pdgs[particle_i] == 2112:
                # neutral particle, no edep, don't include in the geant 3D points for purpose of categorizing the edeps
                continue
            for i in range(num_interpolated_points + 1):
                curr_true_gamma_2_geant_points.append(np.array([
                    second_daughter_descendants_start_xs[particle_i] + (second_daughter_descendants_end_xs[particle_i] - second_daughter_descendants_start_xs[particle_i]) * (i + 1) / (num_interpolated_points + 1), 
                    second_daughter_descendants_start_ys[particle_i] + (second_daughter_descendants_end_ys[particle_i] - second_daughter_descendants_start_ys[particle_i]) * (i + 1) / (num_interpolated_points + 1), 
                    second_daughter_descendants_start_zs[particle_i] + (second_daughter_descendants_end_zs[particle_i] - second_daughter_descendants_start_zs[particle_i]) * (i + 1) / (num_interpolated_points + 1), 
                ]))
        if len(curr_true_gamma_2_geant_points) == 0:
            true_gamma_2_geant_points.append(np.array([]))
        else:
            curr_true_gamma_2_geant_points = np.stack(curr_true_gamma_2_geant_points, axis=0)
            true_gamma_2_geant_points.append(curr_true_gamma_2_geant_points)

        for particle_i in range(len(other_particles_start_xs)):
            if other_particles_pdgs[particle_i] == 22 or other_particles_pdgs[particle_i] == 2112:
                # neutral particle, no edep, don't include in the geant 3D points for purpose of categorizing the edeps
                continue
            for i in range(num_interpolated_points + 1):
                curr_other_particles_geant_points.append(np.array([
                    other_particles_start_xs[particle_i] + (other_particles_end_xs[particle_i] - other_particles_start_xs[particle_i]) * (i + 1) / (num_interpolated_points + 1), 
                    other_particles_start_ys[particle_i] + (other_particles_end_ys[particle_i] - other_particles_start_ys[particle_i]) * (i + 1) / (num_interpolated_points + 1), 
                    other_particles_start_zs[particle_i] + (other_particles_end_zs[particle_i] - other_particles_start_zs[particle_i]) * (i + 1) / (num_interpolated_points + 1), 
                ]))
        if len(curr_other_particles_geant_points) == 0:
            other_particles_geant_points.append(np.array([]))
        else:
            curr_other_particles_geant_points = np.stack(curr_other_particles_geant_points, axis=0)
            other_particles_geant_points.append(curr_other_particles_geant_points)

    return true_gamma_1_geant_points, true_gamma_2_geant_points, other_particles_geant_points
        
def load_all_spacepoints(f, num_events):

    print("loading spacepoints from root file")
    # re-formatting the spacepoints before downsampling

    spacepoints = f["wcpselection"]["T_spacepoints"].arrays(["Tcluster_spacepoints_x", 
                                                                    "Tcluster_spacepoints_y", 
                                                                    "Tcluster_spacepoints_z", 
                                                                    #"Tcluster_spacepoints_q",
                                                                    "Trec_spacepoints_x", 
                                                                    "Trec_spacepoints_y", 
                                                                    "Trec_spacepoints_z", 
                                                                    #"Trec_spacepoints_q",
                                                                    "TrueEDep_spacepoints_startx",
                                                                    "TrueEDep_spacepoints_starty",
                                                                    "TrueEDep_spacepoints_startz",
                                                                    "TrueEDep_spacepoints_endx",
                                                                    "TrueEDep_spacepoints_endy",
                                                                    "TrueEDep_spacepoints_endz",
                                                                    "TrueEDep_spacepoints_edep",
                                                                    ], 
                                                                    entry_start=0, entry_stop=num_events, library="np")

    Tcluster_spacepoints = []
    Trec_spacepoints = []
    TrueEDep_spacepoints = []
    TrueEDep_spacepoints_edep = []
    for event_i in tqdm(range(num_events)):

        # reconstructed spacepoints 
        Tcluster_spacepoints.append(np.stack([spacepoints["Tcluster_spacepoints_x"][event_i],
                                            spacepoints["Tcluster_spacepoints_y"][event_i],
                                            spacepoints["Tcluster_spacepoints_z"][event_i]], axis=-1))
        Trec_spacepoints.append(np.stack([spacepoints["Trec_spacepoints_x"][event_i],
                                        spacepoints["Trec_spacepoints_y"][event_i],
                                        spacepoints["Trec_spacepoints_z"][event_i]], axis=-1))
        
        # true edep spacepoints
        starts = np.stack([spacepoints["TrueEDep_spacepoints_startx"][event_i],
                        spacepoints["TrueEDep_spacepoints_starty"][event_i],
                        spacepoints["TrueEDep_spacepoints_startz"][event_i]], axis=-1)
        midpoints = np.stack([(spacepoints["TrueEDep_spacepoints_startx"][event_i] + spacepoints["TrueEDep_spacepoints_endx"][event_i])/2,
                            (spacepoints["TrueEDep_spacepoints_starty"][event_i] + spacepoints["TrueEDep_spacepoints_endy"][event_i])/2,
                            (spacepoints["TrueEDep_spacepoints_startz"][event_i] + spacepoints["TrueEDep_spacepoints_endz"][event_i])/2], axis=-1)
        ends = np.stack([spacepoints["TrueEDep_spacepoints_endx"][event_i],
                        spacepoints["TrueEDep_spacepoints_endy"][event_i],
                        spacepoints["TrueEDep_spacepoints_endz"][event_i]], axis=-1)
        TrueEDep_spacepoints.append(np.concatenate([starts, midpoints, ends], axis=0))
        # assuming a third of the energy at the start, midpoint, and end
        TrueEDep_spacepoints_edep.append(np.concatenate([spacepoints["TrueEDep_spacepoints_edep"][event_i]/3,
                                                        spacepoints["TrueEDep_spacepoints_edep"][event_i]/3,
                                                        spacepoints["TrueEDep_spacepoints_edep"][event_i]/3], axis=0))
        
    del spacepoints
    del f

    return Tcluster_spacepoints, Trec_spacepoints, TrueEDep_spacepoints, TrueEDep_spacepoints_edep

def categorize_true_EDeps(TrueEDep_spacepoints, TrueEDep_spacepoints_edep, true_gamma_1_geant_points, true_gamma_2_geant_points, other_particles_geant_points, num_events):
    
    print("categorizing true energy deposition points")

    # Split TrueEDep_spacepoints into gamma1, gamma2 and other particles
    true_gamma1_EDep_spacepoints = []
    true_gamma1_EDep_spacepoints_edep = []
    true_gamma2_EDep_spacepoints = []
    true_gamma2_EDep_spacepoints_edep = []
    other_particles_EDep_spacepoints = []
    other_particles_EDep_spacepoints_edep = []

    for event_i in tqdm(range(num_events)):

        # Get distances from each TrueEDep point to nearest gamma1 and gamma2 points
        dists_to_gamma1 = get_min_dists(TrueEDep_spacepoints[event_i], true_gamma_1_geant_points[event_i])
        dists_to_gamma2 = get_min_dists(TrueEDep_spacepoints[event_i], true_gamma_2_geant_points[event_i])
        dists_to_other = get_min_dists(TrueEDep_spacepoints[event_i], other_particles_geant_points[event_i])
        
        # Classify each point based on closest gamma
        gamma1_mask = (dists_to_gamma1 < dists_to_gamma2) & (dists_to_gamma1 < dists_to_other)
        gamma2_mask = (dists_to_gamma2 < dists_to_gamma1) & (dists_to_gamma2 < dists_to_other)
        other_mask = (dists_to_other < dists_to_gamma1) & (dists_to_other < dists_to_gamma2)

        gamma1_mask = gamma1_mask.flatten()
        gamma2_mask = gamma2_mask.flatten()
        other_mask = other_mask.flatten()

        # Split points and energies according to masks
        true_gamma1_EDep_spacepoints.append(TrueEDep_spacepoints[event_i][gamma1_mask.reshape(-1)])
        true_gamma1_EDep_spacepoints_edep.append(TrueEDep_spacepoints_edep[event_i][gamma1_mask])

        true_gamma2_EDep_spacepoints.append(TrueEDep_spacepoints[event_i][gamma2_mask])
        true_gamma2_EDep_spacepoints_edep.append(TrueEDep_spacepoints_edep[event_i][gamma2_mask])

        other_particles_EDep_spacepoints.append(TrueEDep_spacepoints[event_i][other_mask])    
        other_particles_EDep_spacepoints_edep.append(TrueEDep_spacepoints_edep[event_i][other_mask])

    return true_gamma1_EDep_spacepoints, true_gamma1_EDep_spacepoints_edep, true_gamma2_EDep_spacepoints, true_gamma2_EDep_spacepoints_edep, other_particles_EDep_spacepoints, other_particles_EDep_spacepoints_edep

def downsample_spacepoints(Tcluster_spacepoints, Trec_spacepoints, 
                           TrueEDep_spacepoints, TrueEDep_spacepoints_edep, 
                           true_gamma1_EDep_spacepoints, true_gamma1_EDep_spacepoints_edep, 
                           true_gamma2_EDep_spacepoints, true_gamma2_EDep_spacepoints_edep, 
                           other_particles_EDep_spacepoints, other_particles_EDep_spacepoints_edep, 
                           num_events, reco_nu_vtx,
                           close_to_reco_nu_vtx_threshold=200,
                           recalculate_downsampling=True):

        
    print("downsampling spacepoints")

    downsampled_Tcluster_spacepoints = {}
    downsampled_Trec_spacepoints = {}
    downsampled_TrueEDep_spacepoints = {}
    downsampled_true_gamma1_EDep_spacepoints = {}
    downsampled_true_gamma2_EDep_spacepoints = {}
    downsampled_other_particles_EDep_spacepoints = {}
    for event_i in tqdm(range(num_events)):

        nearby_reco_nu_vtx_indices = np.where(np.sqrt((Tcluster_spacepoints[event_i][:, 0] - reco_nu_vtx[event_i][0])**2
                                                    + (Tcluster_spacepoints[event_i][:, 1] - reco_nu_vtx[event_i][1])**2
                                                    + (Tcluster_spacepoints[event_i][:, 2] - reco_nu_vtx[event_i][2])**2) < close_to_reco_nu_vtx_threshold)[0]
        Tcluster_spacepoints_near_reco_nu_vtx = Tcluster_spacepoints[event_i][nearby_reco_nu_vtx_indices, :]
        downsampled_Tcluster_spacepoints[event_i] = fps_clustering_downsample(Tcluster_spacepoints_near_reco_nu_vtx, 500)

        downsampled_Trec_spacepoints[event_i] = fps_clustering_downsample(Trec_spacepoints[event_i], 200)

        downsampled_TrueEDep_spacepoints[event_i] = energy_weighted_density_sampling(TrueEDep_spacepoints[event_i], TrueEDep_spacepoints_edep[event_i], 500)

        downsampled_true_gamma1_EDep_spacepoints[event_i] = energy_weighted_density_sampling(true_gamma1_EDep_spacepoints[event_i], true_gamma1_EDep_spacepoints_edep[event_i], 500)
        downsampled_true_gamma2_EDep_spacepoints[event_i] = energy_weighted_density_sampling(true_gamma2_EDep_spacepoints[event_i], true_gamma2_EDep_spacepoints_edep[event_i], 500)
        downsampled_other_particles_EDep_spacepoints[event_i] = energy_weighted_density_sampling(other_particles_EDep_spacepoints[event_i], other_particles_EDep_spacepoints_edep[event_i], 500)

    return (downsampled_Tcluster_spacepoints, downsampled_Trec_spacepoints, downsampled_TrueEDep_spacepoints, 
            downsampled_true_gamma1_EDep_spacepoints, downsampled_true_gamma2_EDep_spacepoints, downsampled_other_particles_EDep_spacepoints)

def categorize_downsampled_reco_spacepoints(downsampled_Tcluster_spacepoints, downsampled_Trec_spacepoints, downsampled_TrueEDep_spacepoints, 
                                            downsampled_true_gamma1_EDep_spacepoints, downsampled_true_gamma2_EDep_spacepoints, downsampled_other_particles_EDep_spacepoints, 
                                            num_events):
    
    print("categorizing downsampled reco spacepoints")
    
    real_nu_reco_nu_downsampled_spacepoints = []
    real_nu_reco_cosmic_downsampled_spacepoints = []
    real_cosmic_reco_nu_downsampled_spacepoints = []
    real_cosmic_reco_cosmic_downsampled_spacepoints = []

    real_gamma1_downsampled_spacepoints = []
    real_gamma2_downsampled_spacepoints = []
    real_other_particles_downsampled_spacepoints = []
    real_cosmic_downsampled_spacepoints = []

    close_to_true_nu_spacepoint_threshold = 5
    close_to_reco_nu_spacepoint_threshold = 5
    close_to_true_particcle_spacepoint_threshold = 5

    for event_i in tqdm(range(num_events)):

        if len(downsampled_Tcluster_spacepoints[event_i]) == 0:
            real_nu_reco_cosmic_downsampled_spacepoints.append(np.empty((0, 3)))
            real_nu_reco_nu_downsampled_spacepoints.append(np.empty((0, 3)))
            real_cosmic_reco_nu_downsampled_spacepoints.append(np.empty((0, 3)))
            real_cosmic_reco_cosmic_downsampled_spacepoints.append(np.empty((0, 3)))
            real_gamma1_downsampled_spacepoints.append(np.empty((0, 3)))
            real_gamma2_downsampled_spacepoints.append(np.empty((0, 3)))
            real_other_particles_downsampled_spacepoints.append(np.empty((0, 3)))
            real_cosmic_downsampled_spacepoints.append(np.empty((0, 3)))
            continue

        # for T_cluster spacepoints, noting distances to true nu and reco nu spacepoints, and which are close to the reco nu vtx
        min_truth_dists = get_min_dists(downsampled_Tcluster_spacepoints[event_i][:, :3], downsampled_TrueEDep_spacepoints[event_i][:, :3])
        min_reco_nu_dists = get_min_dists(downsampled_Tcluster_spacepoints[event_i][:, :3], downsampled_Trec_spacepoints[event_i][:, :3])
        min_true_gamma1_dists = get_min_dists(downsampled_Tcluster_spacepoints[event_i][:, :3], downsampled_true_gamma1_EDep_spacepoints[event_i][:, :3])
        min_true_gamma2_dists = get_min_dists(downsampled_Tcluster_spacepoints[event_i][:, :3], downsampled_true_gamma2_EDep_spacepoints[event_i][:, :3])
        min_other_particles_dists = get_min_dists(downsampled_Tcluster_spacepoints[event_i][:, :3], downsampled_other_particles_EDep_spacepoints[event_i][:, :3])

        # assign features to spacepoints here
        close_to_truth_indices = np.where(min_truth_dists < close_to_true_nu_spacepoint_threshold)[0]
        far_from_truth_indices = np.where(min_truth_dists >= close_to_true_nu_spacepoint_threshold)[0]
        close_to_reco_nu_indices = np.where(min_reco_nu_dists < close_to_reco_nu_spacepoint_threshold)[0]
        far_from_reco_nu_indices = np.where(min_reco_nu_dists >= close_to_reco_nu_spacepoint_threshold)[0]
        close_to_true_gamma1_indices = np.where(min_true_gamma1_dists < close_to_true_particcle_spacepoint_threshold)[0]
        close_to_true_gamma2_indices = np.where(min_true_gamma2_dists < close_to_true_particcle_spacepoint_threshold)[0]
        close_to_other_particles_indices = np.where(min_other_particles_dists < close_to_true_particcle_spacepoint_threshold)[0]

        # categorize spacepoints here
        real_nu_reco_nu_indices = np.intersect1d(close_to_reco_nu_indices, close_to_truth_indices)
        real_nu_reco_cosmic_indices = np.intersect1d(far_from_reco_nu_indices, close_to_truth_indices)
        real_cosmic_reco_nu_indices = np.intersect1d(close_to_reco_nu_indices, far_from_truth_indices)
        real_cosmic_reco_cosmic_indices = np.intersect1d(far_from_reco_nu_indices, far_from_truth_indices)

        real_nu_reco_cosmic_downsampled_spacepoints.append(downsampled_Tcluster_spacepoints[event_i][real_nu_reco_cosmic_indices, :])
        real_nu_reco_nu_downsampled_spacepoints.append(downsampled_Tcluster_spacepoints[event_i][real_nu_reco_nu_indices, :])
        real_cosmic_reco_nu_downsampled_spacepoints.append(downsampled_Tcluster_spacepoints[event_i][real_cosmic_reco_nu_indices, :])
        real_cosmic_reco_cosmic_downsampled_spacepoints.append(downsampled_Tcluster_spacepoints[event_i][real_cosmic_reco_cosmic_indices, :])

        real_gamma1_downsampled_spacepoints.append(downsampled_Tcluster_spacepoints[event_i][close_to_true_gamma1_indices, :])
        real_gamma2_downsampled_spacepoints.append(downsampled_Tcluster_spacepoints[event_i][close_to_true_gamma2_indices, :])
        real_other_particles_downsampled_spacepoints.append(downsampled_Tcluster_spacepoints[event_i][close_to_other_particles_indices, :])
        real_cosmic_downsampled_spacepoints.append(downsampled_Tcluster_spacepoints[event_i][far_from_truth_indices, :])

    return (real_nu_reco_nu_downsampled_spacepoints, real_nu_reco_cosmic_downsampled_spacepoints, real_cosmic_reco_nu_downsampled_spacepoints, real_cosmic_reco_cosmic_downsampled_spacepoints, 
            real_gamma1_downsampled_spacepoints, real_gamma2_downsampled_spacepoints, real_other_particles_downsampled_spacepoints, real_cosmic_downsampled_spacepoints)

def plot_event(index, 
               Tcluster_spacepoints, Trec_spacepoints, TrueEDep_spacepoints, true_gamma1_EDep_spacepoints, 
               true_gamma2_EDep_spacepoints, other_particles_EDep_spacepoints,
               downsampled_Tcluster_spacepoints, downsampled_Trec_spacepoints, downsampled_TrueEDep_spacepoints, 
               downsampled_true_gamma1_EDep_spacepoints, downsampled_true_gamma2_EDep_spacepoints, downsampled_other_particles_EDep_spacepoints,
               real_nu_reco_nu_downsampled_spacepoints, real_nu_reco_cosmic_downsampled_spacepoints, real_cosmic_reco_nu_downsampled_spacepoints, real_cosmic_reco_cosmic_downsampled_spacepoints, 
               real_gamma1_downsampled_spacepoints, real_gamma2_downsampled_spacepoints, real_other_particles_downsampled_spacepoints, real_cosmic_downsampled_spacepoints,
               reco_nu_vtx, true_nu_vtx,
               include_non_downsampled_points=True):

    from plot_3d_helpers import expanded_detector_boundary_points, detector_boundary_points
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

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


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Pre-processing root file to extract spacepoint information.")
    parser.add_argument('-f', '--file', type=str, required=True, help='Path to root file to pre-process.')
    parser.add_argument('-n', '--num_events', type=str, help='Number of events to process (default is entire file).')
    parser.add_argument('-p', '--plot_index', type=str, help='Index of an event to plot in 3D.')
    parser.add_argument('-ns', '--no_save', action='store_true', help='Do not save the downsampled spacepoints to a pickle file.')
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)
    args = parser.parse_args()

    if args.file is None:
        raise ValueError("File path is required.")
    root_filename = args.file

    print(f"loading file: '{root_filename}'")
    f = uproot.open(root_filename)

    rse, true_nu_vtx, reco_nu_vtx = get_basic_info(f)

    if args.num_events is None:
        num_events = len(f["wcpselection"]["T_eval"].arrays(["run"], library="np"))
    else:
        num_events = int(args.num_events)

    true_gamma_1_geant_points, true_gamma_2_geant_points, other_particles_geant_points = get_geant_points(f, num_events=num_events)

    Tcluster_spacepoints, Trec_spacepoints, TrueEDep_spacepoints, TrueEDep_spacepoints_edep = load_all_spacepoints(f, num_events)

    categorized_outputs = categorize_true_EDeps(TrueEDep_spacepoints, TrueEDep_spacepoints_edep, true_gamma_1_geant_points, true_gamma_2_geant_points, other_particles_geant_points, num_events)
    true_gamma1_EDep_spacepoints = categorized_outputs[0]
    true_gamma1_EDep_spacepoints_edep = categorized_outputs[1]
    true_gamma2_EDep_spacepoints = categorized_outputs[2]
    true_gamma2_EDep_spacepoints_edep = categorized_outputs[3]
    other_particles_EDep_spacepoints = categorized_outputs[4]
    other_particles_EDep_spacepoints_edep = categorized_outputs[5]

    downsampled_spacepoint_outputs = downsample_spacepoints(Tcluster_spacepoints, Trec_spacepoints, 
                                                     TrueEDep_spacepoints, TrueEDep_spacepoints_edep, 
                                                     true_gamma1_EDep_spacepoints, true_gamma1_EDep_spacepoints_edep, 
                                                     true_gamma2_EDep_spacepoints, true_gamma2_EDep_spacepoints_edep, 
                                                     other_particles_EDep_spacepoints, other_particles_EDep_spacepoints_edep, 
                                                     num_events, reco_nu_vtx)

    downsampled_Tcluster_spacepoints = downsampled_spacepoint_outputs[0]
    downsampled_Trec_spacepoints = downsampled_spacepoint_outputs[1]
    downsampled_TrueEDep_spacepoints = downsampled_spacepoint_outputs[2]
    downsampled_true_gamma1_EDep_spacepoints = downsampled_spacepoint_outputs[3]
    downsampled_true_gamma2_EDep_spacepoints = downsampled_spacepoint_outputs[4]
    downsampled_other_particles_EDep_spacepoints = downsampled_spacepoint_outputs[5]

    categorized_downsampled_reco_spacepoints_outputs = categorize_downsampled_reco_spacepoints(downsampled_Tcluster_spacepoints, downsampled_Trec_spacepoints, downsampled_TrueEDep_spacepoints, 
                                                                                                downsampled_true_gamma1_EDep_spacepoints, downsampled_true_gamma2_EDep_spacepoints, downsampled_other_particles_EDep_spacepoints, 
                                                                                                num_events)

    real_nu_reco_nu_downsampled_spacepoints = categorized_downsampled_reco_spacepoints_outputs[0]
    real_nu_reco_cosmic_downsampled_spacepoints = categorized_downsampled_reco_spacepoints_outputs[1]
    real_cosmic_reco_nu_downsampled_spacepoints = categorized_downsampled_reco_spacepoints_outputs[2]
    real_cosmic_reco_cosmic_downsampled_spacepoints = categorized_downsampled_reco_spacepoints_outputs[3]
    real_gamma1_downsampled_spacepoints = categorized_downsampled_reco_spacepoints_outputs[4]
    real_gamma2_downsampled_spacepoints = categorized_downsampled_reco_spacepoints_outputs[5]
    real_other_particles_downsampled_spacepoints = categorized_downsampled_reco_spacepoints_outputs[6]
    real_cosmic_downsampled_spacepoints = categorized_downsampled_reco_spacepoints_outputs[7]

    if args.plot_index is not None:
        plot_event(int(args.plot_index), 
                   Tcluster_spacepoints, Trec_spacepoints, TrueEDep_spacepoints, true_gamma1_EDep_spacepoints, true_gamma2_EDep_spacepoints, other_particles_EDep_spacepoints,
                   downsampled_Tcluster_spacepoints, downsampled_Trec_spacepoints, downsampled_TrueEDep_spacepoints, 
                   downsampled_true_gamma1_EDep_spacepoints, downsampled_true_gamma2_EDep_spacepoints, downsampled_other_particles_EDep_spacepoints,
                   real_nu_reco_nu_downsampled_spacepoints, real_nu_reco_cosmic_downsampled_spacepoints, real_cosmic_reco_nu_downsampled_spacepoints, real_cosmic_reco_cosmic_downsampled_spacepoints, 
                   real_gamma1_downsampled_spacepoints, real_gamma2_downsampled_spacepoints, real_other_particles_downsampled_spacepoints, real_cosmic_downsampled_spacepoints,
                   reco_nu_vtx, true_nu_vtx)
        
    if not args.no_save:
        print("saving downsampled spacepoints to pickle file")
        with open("downsampled_spacepoints.pkl", "wb") as f:
            pickle.dump((downsampled_Tcluster_spacepoints, downsampled_Trec_spacepoints, downsampled_TrueEDep_spacepoints, 
                         downsampled_true_gamma1_EDep_spacepoints, downsampled_true_gamma2_EDep_spacepoints, downsampled_other_particles_EDep_spacepoints), f)

        print("finished saving")
    