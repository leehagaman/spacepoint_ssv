"""
Preprocessing script for spacepoint data with multiprocessing support.

This script processes ROOT files containing spacepoint data and extracts downsampled
spacepoints for machine learning applications. The chunk processing loop has been
optimized with multiprocessing to handle large datasets more efficiently.

Usage:
    python preprocess_spacepoints.py -f <root_file> [-n <num_events>] [-p <num_processes>] [-s <seed>] [-o <output_file>]

Multiprocessing:
    - Use -p to specify the number of processes (default: auto-detect)
    - Set -p 1 to force sequential processing
    - The script will automatically fall back to sequential processing if multiprocessing fails
    - I found that -p 4 works well, I think the limiting factor is reading the ROOT file, so a large number of processes just makes the first part of each process very slow
"""
import sys
import uproot
import numpy as np
from tqdm import tqdm
import pickle
import argparse
import pandas as pd
import multiprocessing as mp

from helpers.spacepoint_sampling import fps_clustering_downsample, get_min_dists, energy_weighted_density_sampling

def get_vtx_and_true_gamma_info(f, num_events, deleted_gamma_indices, rng=None):

    # loads non-spacepoint information from the root file, including RSE, true nu vtx, reco nu vtx, and true gamma info

    rse = f["wcpselection"]["T_eval"].arrays(["run", "subrun", "event"], library="np", entry_start=0, entry_stop=num_events)
    true_nu_vtx = f["wcpselection"]["T_eval"].arrays(["truth_vtxX", "truth_vtxY", "truth_vtxZ"], library="np", entry_start=0, entry_stop=num_events)
    true_nu_vtx = np.stack([true_nu_vtx["truth_vtxX"], true_nu_vtx["truth_vtxY"], true_nu_vtx["truth_vtxZ"]], axis=-1)
    reco_nu_vtx = f["wcpselection"]["T_PFeval"].arrays(["reco_nuvtxX", "reco_nuvtxY", "reco_nuvtxZ"], library="np", entry_start=0, entry_stop=num_events)
    reco_nu_vtx = np.stack([reco_nu_vtx["reco_nuvtxX"], reco_nu_vtx["reco_nuvtxY"], reco_nu_vtx["reco_nuvtxZ"]], axis=-1)

    true_gamma_info_df = pd.DataFrame({
        "run": rse["run"],
        "subrun": rse["subrun"],
        "event": rse["event"],
        "true_nu_vtx_x": true_nu_vtx[:, 0],
        "true_nu_vtx_y": true_nu_vtx[:, 1],
        "true_nu_vtx_z": true_nu_vtx[:, 2],
        "reco_nu_vtx_x": reco_nu_vtx[:, 0],
        "reco_nu_vtx_y": reco_nu_vtx[:, 1],
        "reco_nu_vtx_z": reco_nu_vtx[:, 2],
    })

    # these variables will be used to define signal vs background
    # only includes gammas from a pi0 (primary or non-primary)
    true_num_gamma = []
    true_gamma_energies = []
    true_gamma_pairconversion_xs = []
    true_gamma_pairconversion_ys = []
    true_gamma_pairconversion_zs = []
    true_num_gamma_pairconvert = []
    true_num_gamma_pairconvert_in_FV = []
    true_num_gamma_pairconvert_in_FV_20_MeV = []

    wc_geant_dic = f["wcpselection"]["T_PFeval"].arrays(["truth_id", "truth_mother", "truth_pdg", "truth_startMomentum", "truth_startXYZT", "truth_endXYZT"], library="np")

    deleted_gamma_pf_indices = []

    for event_i in range(num_events):

        if deleted_gamma_indices is None:
            delete_gamma_here = False
        else:
            delete_gamma_here = event_i in deleted_gamma_indices

        if not delete_gamma_here:
            deleted_gamma_pf_indices.append(None)

        num_particles = len(wc_geant_dic["truth_id"][event_i])
                
        curr_true_num_gamma = 0
        curr_true_gamma_energies = []
        curr_true_gamma_pairconversion_xs = []
        curr_true_gamma_pairconversion_ys = []
        curr_true_gamma_pairconversion_zs = []
        curr_true_num_gamma_pairconvert = 0
        curr_true_num_gamma_pairconvert_in_FV = 0
        curr_true_num_gamma_pairconvert_in_FV_20_MeV = 0

        pi0_ids = []
        for i in range(num_particles):
            if wc_geant_dic["truth_pdg"][event_i][i] == 111:
                pi0_ids.append(wc_geant_dic["truth_id"][event_i][i])

        primary_or_pi0_gamma_ids = []
        if rng is not None:
            particle_indices = rng.permutation(num_particles)
        else:
            particle_indices = np.arange(num_particles)
        for i in particle_indices: # randomize the order here, so that when we delete one primary/pi0 photon, it's a random one
            if wc_geant_dic["truth_mother"][event_i][i] in pi0_ids or wc_geant_dic["truth_mother"][event_i][i] == 0: # this is a daughter of a pi0 or a primary particle
                if wc_geant_dic["truth_pdg"][event_i][i] == 22: # this is a photon from a pi0 or a primary photon (most likely from an eta or Delta radiative)

                    if delete_gamma_here:
                        deleted_gamma_pf_indices.append(i) # note that we deleted this photon
                        delete_gamma_here = False # don't delete another photon in this event
                        continue

                    curr_true_num_gamma += 1
                    curr_true_gamma_energies.append(wc_geant_dic["truth_startMomentum"][event_i][i][3])
                    primary_or_pi0_gamma_ids.append(wc_geant_dic["truth_id"][event_i][i])

        # looking for the first point where the photon transfers more than half its energy to daughter charged particles
        # should be 100% for pair production, but compton scatters can also effectively cause the start of a shower
        # daughter particles could disappear from the geant tree even if it pair converts, that type of photon won't be included here

        # looking for pair conversion points, allowing for the possibility of Compton scattering
        for i in range(num_particles):
            if wc_geant_dic["truth_id"][event_i][i] in primary_or_pi0_gamma_ids: # pi0/primary -> gamma, this won't include the manually deleted photon

                original_gamma_energy = wc_geant_dic["truth_startMomentum"][event_i][i][3]
                cumulative_deposited_energy = 0

                visited_ids = set()
                max_iterations = 1000
                iteration_count = 0
                
                while True:
                    curr_id = wc_geant_dic["truth_id"][event_i][i]
                    
                    if curr_id in visited_ids or iteration_count >= max_iterations:
                        print(f"Breaking potential infinite loop at event {event_i}, particle {i}, iteration {iteration_count}")
                        break
                    visited_ids.add(curr_id)
                    iteration_count += 1
                    
                    descendants_ids = []
                    descendants_indices = []
                    descendants_pdgs = []
                    for j in range(num_particles):
                        if wc_geant_dic["truth_mother"][event_i][j] == curr_id: # pi0/primary -> gamma -> this particle
                            descendants_ids.append(wc_geant_dic["truth_id"][event_i][j])
                            descendants_indices.append(j)
                            descendants_pdgs.append(wc_geant_dic["truth_pdg"][event_i][j])

                    for descendant_i in range(len(descendants_indices)):
                        if abs(descendants_pdgs[descendant_i]) == 11: # electron/positron daughter
                            cumulative_deposited_energy += wc_geant_dic["truth_startMomentum"][event_i][descendants_indices[descendant_i]][3]

                    if cumulative_deposited_energy > original_gamma_energy / 2: # it has deposited enough energy to effectively count as a pair conversion
                        break

                    if 22 in descendants_pdgs: # found a compton scatter, hasn't deposited enough energy yet, loop to consider that next photon
                        curr_id = descendants_ids[descendants_pdgs.index(22)]
                        #print("doing a compton scatter")
                    else: # no compton scatter, we're done, it's either a pair conversion or photoelectric absorption or a Geant tree deletion
                        break

                if cumulative_deposited_energy < original_gamma_energy / 2: # weird event, didn't deposit enough energy to count as a pair conversion
                    #print(f"weird event, no daughter photon, but also deposited less than half the energy: {cumulative_deposited_energy} / {original_gamma_energy}")
                    pass
                else:

                    curr_true_gamma_pairconversion_xs.append(wc_geant_dic["truth_startXYZT"][event_i][descendants_indices[0]][0])
                    curr_true_gamma_pairconversion_ys.append(wc_geant_dic["truth_startXYZT"][event_i][descendants_indices[0]][1])
                    curr_true_gamma_pairconversion_zs.append(wc_geant_dic["truth_startXYZT"][event_i][descendants_indices[0]][2])
                    curr_true_num_gamma_pairconvert += 1

                    if -1 < curr_true_gamma_pairconversion_xs[-1] <= 254.3 and -115.0 < curr_true_gamma_pairconversion_ys[-1] <= 117.0 and 0.6 < curr_true_gamma_pairconversion_zs[-1] <= 1036.4:
                        curr_true_num_gamma_pairconvert_in_FV += 1

                        if original_gamma_energy > 0.02:
                            curr_true_num_gamma_pairconvert_in_FV_20_MeV += 1


        true_num_gamma.append(curr_true_num_gamma)
        true_gamma_energies.append(curr_true_gamma_energies)
        true_gamma_pairconversion_xs.append(curr_true_gamma_pairconversion_xs)
        true_gamma_pairconversion_ys.append(curr_true_gamma_pairconversion_ys)
        true_gamma_pairconversion_zs.append(curr_true_gamma_pairconversion_zs)
        true_num_gamma_pairconvert.append(curr_true_num_gamma_pairconvert)
        true_num_gamma_pairconvert_in_FV.append(curr_true_num_gamma_pairconvert_in_FV)
        true_num_gamma_pairconvert_in_FV_20_MeV.append(curr_true_num_gamma_pairconvert_in_FV_20_MeV)

    true_gamma_info_df["true_num_gamma"] = true_num_gamma
    true_gamma_info_df["true_gamma_energies"] = true_gamma_energies
    true_gamma_info_df["true_gamma_pairconversion_xs"] = true_gamma_pairconversion_xs
    true_gamma_info_df["true_gamma_pairconversion_ys"] = true_gamma_pairconversion_ys
    true_gamma_info_df["true_gamma_pairconversion_zs"] = true_gamma_pairconversion_zs
    true_gamma_info_df["true_num_gamma_pairconvert"] = true_num_gamma_pairconvert
    true_gamma_info_df["true_num_gamma_pairconvert_in_FV"] = true_num_gamma_pairconvert_in_FV
    true_gamma_info_df["true_num_gamma_pairconvert_in_FV_20_MeV"] = true_num_gamma_pairconvert_in_FV_20_MeV
    true_gamma_info_df["true_one_pairconvert_in_FV_20_MeV"] = true_num_gamma_pairconvert_in_FV_20_MeV == 1

    manually_deleted_gamma = np.zeros(num_events)
    if deleted_gamma_indices is not None:
        manually_deleted_gamma[deleted_gamma_indices] = 1
    true_gamma_info_df["manually_deleted_gamma"] = manually_deleted_gamma

    # returning the vertex information separately, since that will be used for downsampling
    return true_nu_vtx, reco_nu_vtx, true_gamma_info_df, deleted_gamma_pf_indices


def get_geant_points(f, num_events=None, num_interpolated_points=5, deleted_gamma_indices=None, deleted_gamma_pf_indices=None):

    deleted_photon_types = [] # no deleted photon by default, but this variable will be set to 1 or 2 if gamma1 or gamma2 was manually deleted

    true_gamma_1_geant_points = []
    true_gamma_2_geant_points = []
    other_particles_geant_points = []

    wc_geant_dic = f["wcpselection"]["T_PFeval"].arrays(["truth_id", "truth_mother", "truth_pdg", "truth_startMomentum", "truth_startXYZT", "truth_endXYZT"], library="np")

    if num_events is None:
        num_events = len(wc_geant_dic["truth_id"])

    for event_i in range(num_events):

        curr_deleted_photon_type = 0

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
            deleted_photon_types.append(0)
            continue

        if deleted_gamma_indices is None:
            delete_gamma_here = False
        else:
            delete_gamma_here = event_i in deleted_gamma_indices

        # finding the daughters of the primary pi0
        daughters_of_primary_pi0_ids = []
        daughters_of_primary_pi0_pdgs = []

        for i in range(len(wc_geant_dic["truth_id"][event_i])):

            if wc_geant_dic["truth_mother"][event_i][i] == primary_pi0_id:
                daughters_of_primary_pi0_ids.append(wc_geant_dic["truth_id"][event_i][i])
                daughters_of_primary_pi0_pdgs.append(wc_geant_dic["truth_pdg"][event_i][i])

                if delete_gamma_here and deleted_gamma_pf_indices[event_i] == i: # this is the manually deleted photon
                    curr_deleted_photon_type = len(daughters_of_primary_pi0_pdgs) # sets this as either 1 or 2, for gamma1 or gamma2

        if not (len(daughters_of_primary_pi0_pdgs) == 2 and daughters_of_primary_pi0_pdgs[0] == 22 and daughters_of_primary_pi0_pdgs[1] == 22):
            # either rare decay, or one photon was lost from the Geant tree, skip this event
            true_gamma_1_geant_points.append(np.array([]))
            true_gamma_2_geant_points.append(np.array([]))
            other_particles_geant_points.append(np.array([]))
            deleted_photon_types.append(0)
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
                    first_daughter_descendants_start_xs[particle_i]
                      + (first_daughter_descendants_end_xs[particle_i] - first_daughter_descendants_start_xs[particle_i]) * (i + 1) / (num_interpolated_points + 1), 
                    first_daughter_descendants_start_ys[particle_i]
                      + (first_daughter_descendants_end_ys[particle_i] - first_daughter_descendants_start_ys[particle_i]) * (i + 1) / (num_interpolated_points + 1), 
                    first_daughter_descendants_start_zs[particle_i]
                      + (first_daughter_descendants_end_zs[particle_i] - first_daughter_descendants_start_zs[particle_i]) * (i + 1) / (num_interpolated_points + 1), 
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
                    second_daughter_descendants_start_xs[particle_i]
                      + (second_daughter_descendants_end_xs[particle_i] - second_daughter_descendants_start_xs[particle_i]) * (i + 1) / (num_interpolated_points + 1), 
                    second_daughter_descendants_start_ys[particle_i]
                      + (second_daughter_descendants_end_ys[particle_i] - second_daughter_descendants_start_ys[particle_i]) * (i + 1) / (num_interpolated_points + 1), 
                    second_daughter_descendants_start_zs[particle_i]
                      + (second_daughter_descendants_end_zs[particle_i] - second_daughter_descendants_start_zs[particle_i]) * (i + 1) / (num_interpolated_points + 1), 
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

        deleted_photon_types.append(curr_deleted_photon_type)

    return true_gamma_1_geant_points, true_gamma_2_geant_points, other_particles_geant_points, deleted_photon_types
        
def load_spacepoints_chunk(f, start_idx, end_idx):

    # re-formatting the spacepoints before downsampling

    chunk_num_events = end_idx - start_idx

    spacepoints = f["wcpselection"]["T_spacepoints"].arrays(["Tcluster_spacepoints_x", 
                                                                    "Tcluster_spacepoints_y", 
                                                                    "Tcluster_spacepoints_z", 
                                                                    "Tcluster_spacepoints_q",
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
                                                                    entry_start=start_idx, entry_stop=end_idx, library="np")
    
    Tcluster_spacepoints_with_charge = []
    Trec_spacepoints = []
    TrueEDep_spacepoints = []
    TrueEDep_spacepoints_edep = []
    for event_i in range(chunk_num_events):

        # reconstructed spacepoints 
        Tcluster_spacepoints_with_charge.append(np.stack([spacepoints["Tcluster_spacepoints_x"][event_i],
                                            spacepoints["Tcluster_spacepoints_y"][event_i],
                                            spacepoints["Tcluster_spacepoints_z"][event_i],
                                            spacepoints["Tcluster_spacepoints_q"][event_i]], axis=-1))
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

    return Tcluster_spacepoints_with_charge, Trec_spacepoints, TrueEDep_spacepoints, TrueEDep_spacepoints_edep

def categorize_true_EDeps(TrueEDep_spacepoints, TrueEDep_spacepoints_edep, true_gamma_1_geant_points, true_gamma_2_geant_points, other_particles_geant_points, num_events):
    # we don't delete a photon here, we do that as part of the downsampling step later

    # Split TrueEDep_spacepoints into gamma1, gamma2 and other particles
    true_gamma1_EDep_spacepoints = []
    true_gamma1_EDep_spacepoints_edep = []
    true_gamma2_EDep_spacepoints = []
    true_gamma2_EDep_spacepoints_edep = []
    other_particles_EDep_spacepoints = []
    other_particles_EDep_spacepoints_edep = []

    for event_i in range(num_events):

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

    return (true_gamma1_EDep_spacepoints, true_gamma1_EDep_spacepoints_edep, 
            true_gamma2_EDep_spacepoints, true_gamma2_EDep_spacepoints_edep, 
            other_particles_EDep_spacepoints, other_particles_EDep_spacepoints_edep)


def delete_one_gamma_from_spacepoints(spacepoints_maybe_with_charge, downsampled_deleted_gamma_EDep_spacepoints, distance_threshold=5, num_events=None):

    spacepoints_with_deleted_gamma_maybe_with_charge = []

    if num_events is None:
        num_events = len(spacepoints_maybe_with_charge)

    for event_i in range(num_events):

        if len(downsampled_deleted_gamma_EDep_spacepoints[event_i]) == 0: # no true photon spacepoints to delete, keeping everything as it was
            spacepoints_with_deleted_gamma_maybe_with_charge.append(spacepoints_maybe_with_charge[event_i])
            continue

        if len(spacepoints_maybe_with_charge[event_i]) == 0: # no Tcluster spacepoints, so nothing to delete
            spacepoints_with_deleted_gamma_maybe_with_charge.append(spacepoints_maybe_with_charge[event_i])
            continue

        if spacepoints_maybe_with_charge[event_i].shape[1] == 3: # just 3D points
            spacepoints_with_no_charge = spacepoints_maybe_with_charge[event_i]
        elif spacepoints_maybe_with_charge[event_i].shape[1] == 4: # 3D points with charge
            spacepoints_with_no_charge = spacepoints_maybe_with_charge[event_i][:, :3]
        else:
            raise ValueError(f"Points must be of shape (3,) or (4,), got {spacepoints_maybe_with_charge[event_i].shape}")

        min_dists = get_min_dists(spacepoints_with_no_charge, downsampled_deleted_gamma_EDep_spacepoints[event_i])
        
        deleted_gamma_indices = np.where(min_dists < distance_threshold)[0]
        spacepoints_with_deleted_gamma_maybe_with_charge.append(np.delete(spacepoints_maybe_with_charge[event_i], deleted_gamma_indices, axis=0))

    return spacepoints_with_deleted_gamma_maybe_with_charge


def downsample_spacepoints(spacepoints, reco_nu_vtx, rng=None, close_to_reco_nu_vtx_threshold=200, how="fps", num_events=None, spacepoints_edep=None, downsample_ratio=0.05, max_num_spacepoints=3000):

    downsampled_spacepoints = []
    if num_events is None:
        num_events = len(spacepoints)

    for event_i in range(num_events):
        nearby_reco_nu_vtx_indices = np.where(np.sqrt((spacepoints[event_i][:, 0] - reco_nu_vtx[event_i][0])**2
                                                    + (spacepoints[event_i][:, 1] - reco_nu_vtx[event_i][1])**2
                                                    + (spacepoints[event_i][:, 2] - reco_nu_vtx[event_i][2])**2) < close_to_reco_nu_vtx_threshold)[0]
        nearby_spacepoints = spacepoints[event_i][nearby_reco_nu_vtx_indices, :]
        if spacepoints_edep is not None:
            nearby_spacepoints_edep = spacepoints_edep[event_i][nearby_reco_nu_vtx_indices]
        else:
            nearby_spacepoints_edep = None

        num_spacepoints = len(nearby_spacepoints)

        # choosing the number of spacepoints to downsample to
        if num_spacepoints <= max_num_spacepoints:
            target_num_spacepoints = num_spacepoints
        else:
            target_num_spacepoints = int(downsample_ratio * num_spacepoints)
        if target_num_spacepoints > max_num_spacepoints:
            target_num_spacepoints = max_num_spacepoints

        if how == "fps":
            downsampled_spacepoints.append(fps_clustering_downsample(nearby_spacepoints, target_num_spacepoints, rng))
        elif how == "energy_weighted_density":
            downsampled_spacepoints.append(energy_weighted_density_sampling(nearby_spacepoints, nearby_spacepoints_edep, target_num_spacepoints, rng))

    return downsampled_spacepoints


def categorize_downsampled_reco_spacepoints(downsampled_Tcluster_spacepoints_maybe_with_charge, downsampled_Trec_spacepoints, downsampled_TrueEDep_spacepoints, 
                                            downsampled_true_gamma1_EDep_spacepoints, downsampled_true_gamma2_EDep_spacepoints, downsampled_other_particles_EDep_spacepoints,
                                            distance_threshold = 5, num_events=None):
    
    if num_events is None:
        num_events = len(downsampled_Tcluster_spacepoints_maybe_with_charge)
        
    real_nu_reco_nu_downsampled_spacepoints = []
    real_nu_reco_cosmic_downsampled_spacepoints = []
    real_cosmic_reco_nu_downsampled_spacepoints = []
    real_cosmic_reco_cosmic_downsampled_spacepoints = []

    real_gamma1_downsampled_spacepoints = []
    real_gamma2_downsampled_spacepoints = []
    real_other_particles_downsampled_spacepoints = []
    real_cosmic_downsampled_spacepoints = []
    

    for event_i in range(num_events):

        if len(downsampled_Tcluster_spacepoints_maybe_with_charge[event_i]) == 0:
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
        min_truth_dists = get_min_dists(downsampled_Tcluster_spacepoints_maybe_with_charge[event_i][:, :3], downsampled_TrueEDep_spacepoints[event_i][:, :3])
        min_reco_nu_dists = get_min_dists(downsampled_Tcluster_spacepoints_maybe_with_charge[event_i][:, :3], downsampled_Trec_spacepoints[event_i][:, :3])
        min_true_gamma1_dists = get_min_dists(downsampled_Tcluster_spacepoints_maybe_with_charge[event_i][:, :3], downsampled_true_gamma1_EDep_spacepoints[event_i][:, :3])
        min_true_gamma2_dists = get_min_dists(downsampled_Tcluster_spacepoints_maybe_with_charge[event_i][:, :3], downsampled_true_gamma2_EDep_spacepoints[event_i][:, :3])
        min_other_particles_dists = get_min_dists(downsampled_Tcluster_spacepoints_maybe_with_charge[event_i][:, :3], downsampled_other_particles_EDep_spacepoints[event_i][:, :3])

        # Create mutually exclusive categories using priority-based assignment
        # Priority order: gamma1 > gamma2 > other_particles > cosmic
        # Each spacepoint is assigned to the category with the smallest distance (if within threshold)
        
        num_spacepoints = len(downsampled_Tcluster_spacepoints_maybe_with_charge[event_i])
        category_assignments = np.full(num_spacepoints, -1, dtype=int)  # -1 = unassigned
        
        # Ensure distance arrays are properly shaped
        min_true_gamma1_dists_flat = min_true_gamma1_dists.flatten() if min_true_gamma1_dists.size > 0 else np.full(num_spacepoints, np.inf)
        min_true_gamma2_dists_flat = min_true_gamma2_dists.flatten() if min_true_gamma2_dists.size > 0 else np.full(num_spacepoints, np.inf)
        min_other_particles_dists_flat = min_other_particles_dists.flatten() if min_other_particles_dists.size > 0 else np.full(num_spacepoints, np.inf)
        
        # Assign gamma1 (highest priority)
        gamma1_mask = (min_true_gamma1_dists_flat < distance_threshold)
        category_assignments[gamma1_mask] = 0  # gamma1
        
        # Assign gamma2 (second priority) - only if not already assigned to gamma1
        gamma2_mask = (min_true_gamma2_dists_flat < distance_threshold)
        gamma2_mask = gamma2_mask & (category_assignments == -1)  # only unassigned spacepoints
        category_assignments[gamma2_mask] = 1  # gamma2
        
        # Assign other particles (third priority) - only if not already assigned
        other_mask = (min_other_particles_dists_flat < distance_threshold)
        other_mask = other_mask & (category_assignments == -1)  # only unassigned spacepoints
        category_assignments[other_mask] = 2  # other_particles
        
        # Assign cosmic (lowest priority) - all remaining spacepoints
        cosmic_mask = (category_assignments == -1)  # all unassigned spacepoints
        category_assignments[cosmic_mask] = 3  # cosmic
        
        # Extract indices for each category
        gamma1_indices = np.where(category_assignments == 0)[0]
        gamma2_indices = np.where(category_assignments == 1)[0]
        other_particles_indices = np.where(category_assignments == 2)[0]
        cosmic_indices = np.where(category_assignments == 3)[0]
        
        # For the nu/cosmic categorization (keeping original logic for these)
        close_to_truth_indices = np.where(min_truth_dists < distance_threshold)[0]
        far_from_truth_indices = np.where(min_truth_dists >= distance_threshold)[0]
        close_to_reco_nu_indices = np.where(min_reco_nu_dists < distance_threshold)[0]
        far_from_reco_nu_indices = np.where(min_reco_nu_dists >= distance_threshold)[0]

        # categorize spacepoints here
        real_nu_reco_nu_indices = np.intersect1d(close_to_reco_nu_indices, close_to_truth_indices)
        real_nu_reco_cosmic_indices = np.intersect1d(far_from_reco_nu_indices, close_to_truth_indices)
        real_cosmic_reco_nu_indices = np.intersect1d(close_to_reco_nu_indices, far_from_truth_indices)
        real_cosmic_reco_cosmic_indices = np.intersect1d(far_from_reco_nu_indices, far_from_truth_indices)

        real_nu_reco_cosmic_downsampled_spacepoints.append(downsampled_Tcluster_spacepoints_maybe_with_charge[event_i][real_nu_reco_cosmic_indices, :])
        real_nu_reco_nu_downsampled_spacepoints.append(downsampled_Tcluster_spacepoints_maybe_with_charge[event_i][real_nu_reco_nu_indices, :])
        real_cosmic_reco_nu_downsampled_spacepoints.append(downsampled_Tcluster_spacepoints_maybe_with_charge[event_i][real_cosmic_reco_nu_indices, :])
        real_cosmic_reco_cosmic_downsampled_spacepoints.append(downsampled_Tcluster_spacepoints_maybe_with_charge[event_i][real_cosmic_reco_cosmic_indices, :])

        real_gamma1_downsampled_spacepoints.append(downsampled_Tcluster_spacepoints_maybe_with_charge[event_i][gamma1_indices, :])
        real_gamma2_downsampled_spacepoints.append(downsampled_Tcluster_spacepoints_maybe_with_charge[event_i][gamma2_indices, :])
        real_other_particles_downsampled_spacepoints.append(downsampled_Tcluster_spacepoints_maybe_with_charge[event_i][other_particles_indices, :])
        real_cosmic_downsampled_spacepoints.append(downsampled_Tcluster_spacepoints_maybe_with_charge[event_i][cosmic_indices, :])

    return (downsampled_Tcluster_spacepoints_maybe_with_charge, 
            real_nu_reco_nu_downsampled_spacepoints, real_nu_reco_cosmic_downsampled_spacepoints, 
            real_cosmic_reco_nu_downsampled_spacepoints, real_cosmic_reco_cosmic_downsampled_spacepoints, 
            real_gamma1_downsampled_spacepoints, real_gamma2_downsampled_spacepoints, 
            real_other_particles_downsampled_spacepoints, real_cosmic_downsampled_spacepoints)

def process_chunk(chunk_data):
    """
    Process a single chunk of events for multiprocessing.
    
    Args:
        chunk_data: Tuple containing (chunk_i, start_idx, end_idx, num_events, root_filename, true_gamma_1_geant_points, 
                    true_gamma_2_geant_points, other_particles_geant_points, deleted_photon_types, 
                    reco_nu_vtx, seed)
    
    Returns:
        Tuple of processed chunk results
    """
    (chunk_i, start_idx, end_idx, root_filename, true_gamma_1_geant_points, 
     true_gamma_2_geant_points, other_particles_geant_points, deleted_photon_types, 
     reco_nu_vtx, seed) = chunk_data
    
    # Reopen the file in this process
    f = uproot.open(root_filename)
    
    # Create a new random generator for this process
    rng = np.random.default_rng(seed + chunk_i)  # Different seed for each chunk
    
    chunk_num_events = end_idx - start_idx

    # Use the already-sliced geant points data
    chunk_true_gamma_1_geant_points = true_gamma_1_geant_points
    chunk_true_gamma_2_geant_points = true_gamma_2_geant_points
    chunk_other_particles_geant_points = other_particles_geant_points
    chunk_deleted_photon_types = deleted_photon_types
    chunk_reco_nu_vtx = reco_nu_vtx

    chunk_Tcluster_spacepoints_with_charge, chunk_Trec_spacepoints, chunk_TrueEDep_spacepoints, chunk_TrueEDep_spacepoints_edep = load_spacepoints_chunk(f, start_idx, end_idx)

    chunk_categorized_outputs = categorize_true_EDeps(chunk_TrueEDep_spacepoints, chunk_TrueEDep_spacepoints_edep, chunk_true_gamma_1_geant_points, chunk_true_gamma_2_geant_points, chunk_other_particles_geant_points, chunk_num_events)
    chunk_true_gamma1_EDep_spacepoints = chunk_categorized_outputs[0]
    chunk_true_gamma1_EDep_spacepoints_edep = chunk_categorized_outputs[1]
    chunk_true_gamma2_EDep_spacepoints = chunk_categorized_outputs[2]
    chunk_true_gamma2_EDep_spacepoints_edep = chunk_categorized_outputs[3]
    chunk_other_particles_EDep_spacepoints = chunk_categorized_outputs[4]
    chunk_other_particles_EDep_spacepoints_edep = chunk_categorized_outputs[5]

    chunk_downsampled_true_gamma1_EDep_spacepoints = downsample_spacepoints(chunk_true_gamma1_EDep_spacepoints, chunk_reco_nu_vtx, rng, how="energy_weighted_density", 
                                                                    num_events=chunk_num_events, spacepoints_edep=chunk_true_gamma1_EDep_spacepoints_edep)
    chunk_downsampled_true_gamma2_EDep_spacepoints = downsample_spacepoints(chunk_true_gamma2_EDep_spacepoints, chunk_reco_nu_vtx, rng, how="energy_weighted_density", 
                                                                    num_events=chunk_num_events, spacepoints_edep=chunk_true_gamma2_EDep_spacepoints_edep)
    chunk_downsampled_other_particles_EDep_spacepoints = downsample_spacepoints(chunk_other_particles_EDep_spacepoints, chunk_reco_nu_vtx, rng, how="energy_weighted_density", 
                                                                        num_events=chunk_num_events, spacepoints_edep=chunk_other_particles_EDep_spacepoints_edep)
    chunk_downsampled_TrueEDep_spacepoints = downsample_spacepoints(chunk_TrueEDep_spacepoints, chunk_reco_nu_vtx, rng, how="energy_weighted_density", 
                                                                    num_events=chunk_num_events, spacepoints_edep=chunk_TrueEDep_spacepoints_edep)

    chunk_downsampled_deleted_gamma_EDep_spacepoints = []
    chunk_downsampled_remaining_true_gamma1_EDep_spacepoints = []
    chunk_downsampled_remaining_true_gamma2_EDep_spacepoints = []
    for event_i in range(chunk_num_events):
        if chunk_deleted_photon_types[event_i] == 1:
            chunk_downsampled_deleted_gamma_EDep_spacepoints.append(chunk_downsampled_true_gamma1_EDep_spacepoints[event_i])
            chunk_downsampled_remaining_true_gamma1_EDep_spacepoints.append(np.empty((0, 3)))
            chunk_downsampled_remaining_true_gamma2_EDep_spacepoints.append(chunk_downsampled_true_gamma2_EDep_spacepoints[event_i])
        elif chunk_deleted_photon_types[event_i] == 2:
            chunk_downsampled_deleted_gamma_EDep_spacepoints.append(chunk_downsampled_true_gamma2_EDep_spacepoints[event_i])
            chunk_downsampled_remaining_true_gamma1_EDep_spacepoints.append(chunk_downsampled_true_gamma1_EDep_spacepoints[event_i])
            chunk_downsampled_remaining_true_gamma2_EDep_spacepoints.append(np.empty((0, 3)))
        else:
            chunk_downsampled_deleted_gamma_EDep_spacepoints.append(np.empty((0, 3)))
            chunk_downsampled_remaining_true_gamma1_EDep_spacepoints.append(chunk_downsampled_true_gamma1_EDep_spacepoints[event_i])
            chunk_downsampled_remaining_true_gamma2_EDep_spacepoints.append(chunk_downsampled_true_gamma2_EDep_spacepoints[event_i])

    # deleting the photon from reco spacepoints here, in order to avoid any potential influence of the gamma deletion on the downsampling process
    chunk_Tcluster_spacepoints_with_deleted_gamma_with_charge = delete_one_gamma_from_spacepoints(chunk_Tcluster_spacepoints_with_charge, chunk_downsampled_deleted_gamma_EDep_spacepoints, num_events=chunk_num_events)
    chunk_Trec_spacepoints_with_deleted_gamma = delete_one_gamma_from_spacepoints(chunk_Trec_spacepoints, chunk_downsampled_deleted_gamma_EDep_spacepoints, num_events=chunk_num_events)
    
    chunk_downsampled_Tcluster_spacepoints_with_deleted_gamma_with_charge = downsample_spacepoints(chunk_Tcluster_spacepoints_with_deleted_gamma_with_charge, chunk_reco_nu_vtx, rng, how="fps", num_events=chunk_num_events)
    chunk_downsampled_Trec_spacepoints_with_deleted_gamma = downsample_spacepoints(chunk_Trec_spacepoints_with_deleted_gamma, chunk_reco_nu_vtx, rng, how="fps", num_events=chunk_num_events)

    chunk_categorized_downsampled_reco_spacepoints_outputs = categorize_downsampled_reco_spacepoints(chunk_downsampled_Tcluster_spacepoints_with_deleted_gamma_with_charge, chunk_downsampled_Trec_spacepoints_with_deleted_gamma, 
                                                                                                chunk_downsampled_TrueEDep_spacepoints, chunk_downsampled_remaining_true_gamma1_EDep_spacepoints, 
                                                                                                chunk_downsampled_remaining_true_gamma2_EDep_spacepoints, chunk_downsampled_other_particles_EDep_spacepoints, 
                                                                                                num_events=chunk_num_events)

    return chunk_categorized_downsampled_reco_spacepoints_outputs


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Pre-processing root file to extract spacepoint information.")
    parser.add_argument('-f', '--file', type=str, help='Path to root file to pre-process.', default="input_files/bdt_convert_superunified_bnb_ncpi0_full_spacepoints.root")
    parser.add_argument('-n', '--num_events', type=int, help='Number of events to process (default is entire file).')
    parser.add_argument('-ns', '--no_save', action='store_true', help='Do not save the downsampled spacepoints to a pickle file.')
    parser.add_argument('-fd', '--fraction_of_events_with_deleted_photons', type=float, help='Fraction of events with one photon deleted.', default=0.5)
    parser.add_argument('-s', '--seed', type=int, help='Random seed for reproducibility (default: 42).', default=42)
    parser.add_argument('-o', '--out_file', type=str, help='Output pickle file for the downsampled spacepoints.', default="downsampled_spacepoints.pkl")
    parser.add_argument('-p', '--num_processes', type=int, help='Number of processes for multiprocessing.', default=4)
    parser.add_argument('-c', '--chunk_size', type=int, help='Number of events to process in each chunk (default: 100).', default=100)
    args = parser.parse_args()

    # Create a random generator with the specified seed for reproducibility
    rng = np.random.default_rng(args.seed)
    print(f"Using random seed: {args.seed}")
    
    if args.file is None:
        raise ValueError("File path is required.")
    root_filename = args.file

    print(f"loading file: '{root_filename}'")
    f = uproot.open(root_filename)

    if args.num_events is None:
        args.num_events = len(f["wcpselection"]["T_eval"].arrays(["run"], library="np")["run"])
        print(f"Number of events not specified, using all {args.num_events} events in file")

    if args.fraction_of_events_with_deleted_photons == 0:
        deleted_gamma_indices = None
    else:
        deleted_gamma_indices = rng.choice([0, 1], size=args.num_events, p=[1 - args.fraction_of_events_with_deleted_photons, args.fraction_of_events_with_deleted_photons])
        deleted_gamma_indices = np.where(deleted_gamma_indices == 1)[0]
        print(f"deleted one gamma in {len(deleted_gamma_indices)} / {args.num_events} events")

    print("getting true neutrino and gamma info")
    
    true_nu_vtx, reco_nu_vtx, true_gamma_info_df, deleted_gamma_pf_indices = get_vtx_and_true_gamma_info(f, args.num_events, deleted_gamma_indices, rng)

    #print(true_gamma_info_df.head())

    print("getting true Geant4 spatial information to categorize spacepoints")

    true_gamma_1_geant_points, true_gamma_2_geant_points, other_particles_geant_points, deleted_photon_types = get_geant_points(f, num_events=args.num_events, num_interpolated_points=5, 
                                                                                                                                deleted_gamma_indices=deleted_gamma_indices, 
                                                                                                                                deleted_gamma_pf_indices=deleted_gamma_pf_indices)
    
    all_downsampled_Tcluster_spacepoints_with_charge = []
    all_real_nu_reco_nu_downsampled_spacepoints = []
    all_real_nu_reco_cosmic_downsampled_spacepoints = []
    all_real_cosmic_reco_nu_downsampled_spacepoints = []
    all_real_cosmic_reco_cosmic_downsampled_spacepoints = []
    all_real_gamma1_downsampled_spacepoints = []
    all_real_gamma2_downsampled_spacepoints = []
    all_real_other_particles_downsampled_spacepoints = []
    all_real_cosmic_downsampled_spacepoints = []

    chunk_size = args.chunk_size
    num_chunks = (args.num_events + chunk_size - 1) // chunk_size  # Ceiling division to include remainder
    
    # Prepare chunk data for multiprocessing
    chunk_data_list = []
    for chunk_i in range(num_chunks):
        start_idx = chunk_i * chunk_size
        end_idx = min(start_idx + chunk_size, args.num_events)
        
        # Slice the geant points arrays for this chunk
        chunk_true_gamma_1_geant_points = true_gamma_1_geant_points[start_idx:end_idx]
        chunk_true_gamma_2_geant_points = true_gamma_2_geant_points[start_idx:end_idx]
        chunk_other_particles_geant_points = other_particles_geant_points[start_idx:end_idx]
        chunk_deleted_photon_types = deleted_photon_types[start_idx:end_idx]
        chunk_reco_nu_vtx = reco_nu_vtx[start_idx:end_idx]
        
        chunk_data = (chunk_i, start_idx, end_idx, root_filename, 
                      chunk_true_gamma_1_geant_points, chunk_true_gamma_2_geant_points, 
                      chunk_other_particles_geant_points, chunk_deleted_photon_types, 
                      chunk_reco_nu_vtx, args.seed)
        chunk_data_list.append(chunk_data)
    
    # Use multiprocessing to process chunks in parallel
    if args.num_processes is None:
        num_processes = min(mp.cpu_count(), num_chunks)  # Don't use more processes than chunks
    else:
        num_processes = min(args.num_processes, num_chunks)  # Don't use more processes than chunks
    
    # If only one process is requested or only one chunk, use sequential processing
    if num_processes <= 1:
        print(f"Processing {num_chunks} chunks sequentially")
        chunk_results = []
        for chunk_data in tqdm(chunk_data_list, desc="Processing spacepoint data in chunks"):
            chunk_results.append(process_chunk(chunk_data))
    else:
        print(f"Processing {num_chunks} chunks using {num_processes} processes")
        try:
            with mp.Pool(processes=num_processes) as pool:
                chunk_results = list(tqdm(pool.imap(process_chunk, chunk_data_list), total=num_chunks, desc="Processing spacepoint data in chunks", smoothing=0))
        except Exception as e:
            print(f"Multiprocessing failed with error: {e}")
            print("Falling back to sequential processing")
            chunk_results = []
            for chunk_data in tqdm(chunk_data_list, desc="Processing spacepoint data in chunks"):
                chunk_results.append(process_chunk(chunk_data))
    
    # Combine results from all chunks
    for chunk_i, chunk_categorized_downsampled_reco_spacepoints_outputs in enumerate(chunk_results):
        chunk_downsampled_Tcluster_spacepoints_with_charge = chunk_categorized_downsampled_reco_spacepoints_outputs[0]
        chunk_real_nu_reco_nu_downsampled_spacepoints = chunk_categorized_downsampled_reco_spacepoints_outputs[1]
        chunk_real_nu_reco_cosmic_downsampled_spacepoints = chunk_categorized_downsampled_reco_spacepoints_outputs[2]
        chunk_real_cosmic_reco_nu_downsampled_spacepoints = chunk_categorized_downsampled_reco_spacepoints_outputs[3]
        chunk_real_cosmic_reco_cosmic_downsampled_spacepoints = chunk_categorized_downsampled_reco_spacepoints_outputs[4]
        chunk_real_gamma1_downsampled_spacepoints = chunk_categorized_downsampled_reco_spacepoints_outputs[5]
        chunk_real_gamma2_downsampled_spacepoints = chunk_categorized_downsampled_reco_spacepoints_outputs[6]
        chunk_real_other_particles_downsampled_spacepoints = chunk_categorized_downsampled_reco_spacepoints_outputs[7]
        chunk_real_cosmic_downsampled_spacepoints = chunk_categorized_downsampled_reco_spacepoints_outputs[8]

        all_downsampled_Tcluster_spacepoints_with_charge.extend(chunk_downsampled_Tcluster_spacepoints_with_charge)
        all_real_nu_reco_nu_downsampled_spacepoints.extend(chunk_real_nu_reco_nu_downsampled_spacepoints)
        all_real_nu_reco_cosmic_downsampled_spacepoints.extend(chunk_real_nu_reco_cosmic_downsampled_spacepoints)
        all_real_cosmic_reco_nu_downsampled_spacepoints.extend(chunk_real_cosmic_reco_nu_downsampled_spacepoints)
        all_real_cosmic_reco_cosmic_downsampled_spacepoints.extend(chunk_real_cosmic_reco_cosmic_downsampled_spacepoints)
        all_real_gamma1_downsampled_spacepoints.extend(chunk_real_gamma1_downsampled_spacepoints)
        all_real_gamma2_downsampled_spacepoints.extend(chunk_real_gamma2_downsampled_spacepoints)
        all_real_other_particles_downsampled_spacepoints.extend(chunk_real_other_particles_downsampled_spacepoints)
        all_real_cosmic_downsampled_spacepoints.extend(chunk_real_cosmic_downsampled_spacepoints)
        
    if not args.no_save:
        print("saving downsampled spacepoints to pickle file")
        with open("intermediate_files/" + args.out_file, "wb") as f:
            pickle.dump((true_gamma_info_df, 
                         all_downsampled_Tcluster_spacepoints_with_charge,
                         all_real_nu_reco_nu_downsampled_spacepoints, all_real_nu_reco_cosmic_downsampled_spacepoints,
                         all_real_cosmic_reco_nu_downsampled_spacepoints, all_real_cosmic_reco_cosmic_downsampled_spacepoints,
                         all_real_gamma1_downsampled_spacepoints, all_real_gamma2_downsampled_spacepoints, 
                         all_real_other_particles_downsampled_spacepoints, all_real_cosmic_downsampled_spacepoints), f)

        print("finished saving")
    