from tqdm import tqdm
import pickle
import argparse
import pandas as pd
import sys
import numpy as np


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train spacepoint SSV neural network.")
    parser.add_argument('-f', '--file', type=str, required=False, help='Path to root file to pre-process.', default='intermediate_files/downsampled_spacepoints.pkl')
    parser.add_argument('-n', '--num_events', type=int, required=False, help='Number of training events to use.')
    parser.add_argument('-tf', '--train_fraction', type=float, required=False, help='Fraction of training events to use.', default=0.5)
    args = parser.parse_args()

    print("loading spacepoints from pickle file")
    with open(args.file, "rb") as f:
        outputs = pickle.load(f)
    true_gamma_info_df = outputs[0]
    #real_nu_reco_nu_downsampled_spacepoints = outputs[1]
    #real_nu_reco_cosmic_downsampled_spacepoints = outputs[2]
    #real_cosmic_reco_nu_downsampled_spacepoints = outputs[3]
    #real_cosmic_reco_cosmic_downsampled_spacepoints = outputs[4]
    real_gamma1_downsampled_spacepoints = outputs[5]
    real_gamma2_downsampled_spacepoints = outputs[6]
    real_other_particles_downsampled_spacepoints = outputs[7]
    real_cosmic_downsampled_spacepoints = outputs[8]

    print("splitting into training and testing sets")
    num_events = args.num_events
    if num_events is None:
        num_events = true_gamma_info_df.shape[0]
    num_training_events = int(num_events * args.train_fraction)
    num_testing_events = num_events - num_training_events

    indices = np.arange(num_events)
    rng = np.random.default_rng(seed=42)
    rng.shuffle(indices)
    train_indices = indices[:num_training_events]
    test_indices = indices[num_training_events:]

    train_true_gamma_info = true_gamma_info_df.iloc[train_indices]
    test_true_gamma_info = true_gamma_info_df.iloc[test_indices]
    train_gamma1 = [real_gamma1_downsampled_spacepoints[i] for i in train_indices]
    train_gamma2 = [real_gamma2_downsampled_spacepoints[i] for i in train_indices]
    train_other_particles = [real_other_particles_downsampled_spacepoints[i] for i in train_indices]
    train_cosmic = [real_cosmic_downsampled_spacepoints[i] for i in train_indices]
    test_gamma1 = [real_gamma1_downsampled_spacepoints[i] for i in test_indices]
    test_gamma2 = [real_gamma2_downsampled_spacepoints[i] for i in test_indices]
    test_other_particles = [real_other_particles_downsampled_spacepoints[i] for i in test_indices]
    test_cosmic = [real_cosmic_downsampled_spacepoints[i] for i in test_indices]

    train_RSEs = train_true_gamma_info[["run", "subrun", "event"]].values

    print("saving training RSEs to text file")
    np.savetxt('training_files/train_RSEs.txt', train_RSEs, fmt='%d', delimiter=' ', header='run subrun event')

