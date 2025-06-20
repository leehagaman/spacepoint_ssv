import torch
from torch.utils.data import Dataset, DataLoader
import pickle
import numpy as np
import pandas as pd


class SpacepointDataset(Dataset):
    
    def __init__(self, 
                 pickle_file,
                 num_events = None,
                 random_seed = 42):
        """
        Initialize the dataset.
        
        Args:
            pickle_file: Path to the pickle file containing spacepoint data
            num_events: Number of events to use (None for all events)
            random_seed: Random seed for reproducible shuffling
        """
        self.pickle_file = pickle_file
        self.num_events = num_events
        self.random_seed = random_seed
        
        # Load and preprocess data
        self._load_and_preprocess_data()
        
    def _load_and_preprocess_data(self):
        """Load data from pickle file."""
        print(f"Loading spacepoints from pickle file: {self.pickle_file}")
        with open(self.pickle_file, "rb") as f:
            outputs = pickle.load(f)
        
        self.true_gamma_info_df = outputs[0]

        # Don't need these for now, these contain the original WC guesses 
        # self.real_nu_reco_nu_downsampled_spacepoints = outputs[1]
        # self.real_nu_reco_cosmic_downsampled_spacepoints = outputs[2]
        # self.real_cosmic_reco_nu_downsampled_spacepoints = outputs[3]
        # self.real_cosmic_reco_cosmic_downsampled_spacepoints = outputs[4]

        # These are what we're using for training
        self.real_gamma1_downsampled_spacepoints = outputs[5]
        self.real_gamma2_downsampled_spacepoints = outputs[6]
        self.real_other_particles_downsampled_spacepoints = outputs[7]
        self.real_cosmic_downsampled_spacepoints = outputs[8]

        # Filter out events with no spacepoints
        total_num_spacepoints_per_event = (np.array([len(spacepoints) for spacepoints in self.real_gamma1_downsampled_spacepoints])
                                         + np.array([len(spacepoints) for spacepoints in self.real_gamma2_downsampled_spacepoints])
                                         + np.array([len(spacepoints) for spacepoints in self.real_other_particles_downsampled_spacepoints])
                                         + np.array([len(spacepoints) for spacepoints in self.real_cosmic_downsampled_spacepoints]))
        for num_spacepoints_in_event in total_num_spacepoints_per_event:
            assert num_spacepoints_in_event == 0 or num_spacepoints_in_event == 500, f"Event with {num_spacepoints_in_event} spacepoints, expected 0 or 500"

        event_indices_with_spacepoints = np.where(total_num_spacepoints_per_event > 0)[0]
        self.true_gamma_info_df = self.true_gamma_info_df.iloc[event_indices_with_spacepoints].reset_index(drop=True)
        self.real_gamma1_downsampled_spacepoints = [self.real_gamma1_downsampled_spacepoints[i] for i in event_indices_with_spacepoints]
        self.real_gamma2_downsampled_spacepoints = [self.real_gamma2_downsampled_spacepoints[i] for i in event_indices_with_spacepoints]
        self.real_other_particles_downsampled_spacepoints = [self.real_other_particles_downsampled_spacepoints[i] for i in event_indices_with_spacepoints]
        self.real_cosmic_downsampled_spacepoints = [self.real_cosmic_downsampled_spacepoints[i] for i in event_indices_with_spacepoints]

        if self.num_events is None:
            self.num_events = self.true_gamma_info_df.shape[0]
        print(f"Loading {self.num_events} events")
        self.true_gamma_info_df = self.true_gamma_info_df.iloc[:self.num_events]
        self.real_gamma1_downsampled_spacepoints = self.real_gamma1_downsampled_spacepoints[:self.num_events]
        self.real_gamma2_downsampled_spacepoints = self.real_gamma2_downsampled_spacepoints[:self.num_events]
        self.real_other_particles_downsampled_spacepoints = self.real_other_particles_downsampled_spacepoints[:self.num_events]
        self.real_cosmic_downsampled_spacepoints = self.real_cosmic_downsampled_spacepoints[:self.num_events]
        
        print("Shuffling event ordering")
        # shuffle the event ordering
        shuffled_indices = self.true_gamma_info_df.sample(frac=1, random_state=self.random_seed).index
        self.true_gamma_info_df = self.true_gamma_info_df.iloc[shuffled_indices].reset_index(drop=True)
        self.real_gamma1_downsampled_spacepoints = [self.real_gamma1_downsampled_spacepoints[i] for i in shuffled_indices]
        self.real_gamma2_downsampled_spacepoints = [self.real_gamma2_downsampled_spacepoints[i] for i in shuffled_indices]
        self.real_other_particles_downsampled_spacepoints = [self.real_other_particles_downsampled_spacepoints[i] for i in shuffled_indices]
        self.real_cosmic_downsampled_spacepoints = [self.real_cosmic_downsampled_spacepoints[i] for i in shuffled_indices]

        self.global_y = torch.tensor(self.true_gamma_info_df["true_num_gamma_one_pairconvert_in_FV_20_MeV"].values, dtype=torch.long)
        
        print("Creating spacepoint tensors")
        
        # Create tensors for each event in the format expected by PointNet2: (B, C, N)
        # where B = batch size (events), C = channels (3 for xyz), N = points per event (500)
        self.x = []  # List to store spacepoint tensors for each event
        self.y = []  # List to store label tensors for each event
        
        for event_idx in range(self.num_events):
            event_spacepoints = []
            event_true_labels = []
            
            if len(self.real_gamma1_downsampled_spacepoints[event_idx]) > 0:
                event_spacepoints.append(self.real_gamma1_downsampled_spacepoints[event_idx])
                event_true_labels.extend([0] * len(self.real_gamma1_downsampled_spacepoints[event_idx]))
            if len(self.real_gamma2_downsampled_spacepoints[event_idx]) > 0:
                event_spacepoints.append(self.real_gamma2_downsampled_spacepoints[event_idx])
                event_true_labels.extend([1] * len(self.real_gamma2_downsampled_spacepoints[event_idx]))
            if len(self.real_other_particles_downsampled_spacepoints[event_idx]) > 0:
                event_spacepoints.append(self.real_other_particles_downsampled_spacepoints[event_idx])
                event_true_labels.extend([2] * len(self.real_other_particles_downsampled_spacepoints[event_idx]))
            if len(self.real_cosmic_downsampled_spacepoints[event_idx]) > 0:
                event_spacepoints.append(self.real_cosmic_downsampled_spacepoints[event_idx])
                event_true_labels.extend([3] * len(self.real_cosmic_downsampled_spacepoints[event_idx]))
            
            # Combine all spacepoints for this event
            if event_spacepoints:
                event_spacepoints = np.vstack(event_spacepoints)  # Shape: (500, 3)
                # Convert to tensor and transpose to (3, 500) format
                event_tensor = torch.tensor(event_spacepoints, dtype=torch.float32).transpose(0, 1)  # Shape: (3, 500)
                self.x.append(event_tensor)
                self.y.append(torch.tensor(event_true_labels, dtype=torch.long))
        
        print(f"Created dataset with {len(self.x)} events, each with 500 spacepoints")
        
    def __len__(self):
        """Return the number of events in the dataset."""
        return len(self.x)
    
    def __getitem__(self, idx):
        """Get a specific event from the dataset.
        
        Returns:
            tuple: (spacepoint_coordinates, labels, global_label)
            spacepoint_coordinates: tensor of shape (3, 500) for xyz coordinates
            labels: tensor of shape (500,) for point labels
            global_label: tensor of shape () for event-level label
        """
        return self.x[idx], self.y[idx], self.global_y[idx]


def create_dataloaders(pickle_file,
                      batch_size,
                      num_events,
                      train_fraction,
                      num_workers,
                      random_seed,
                      out_dir,
                      no_save):
    """
    Create train and test dataloaders.
    
    Args:
        pickle_file: Path to the pickle file containing spacepoint data
        batch_size: Batch size for the dataloaders (number of events per batch)
        num_events: Number of events to use (None for all events)
        train_fraction: Fraction of data to use for training
        num_workers: Number of worker processes for data loading
        random_seed: Random seed for reproducible splits
        
    Returns:
        Tuple of (train_dataloader, test_dataloader)
    """
    
    # Create a single dataset with all data
    full_dataset = SpacepointDataset(
        pickle_file=pickle_file,
        num_events=num_events,
        random_seed=random_seed
    )
    
    # Get unique events and create event-based splits
    num_training_events = int(full_dataset.num_events * train_fraction)
    
    # Shuffle events
    generator = torch.Generator().manual_seed(random_seed)
    event_perm = torch.randperm(full_dataset.num_events, generator=generator)
    
    train_event_indices = event_perm[:num_training_events]
    test_event_indices = event_perm[num_training_events:]
    
    # Create train and test datasets using Subset
    from torch.utils.data import Subset
    train_dataset = Subset(full_dataset, train_event_indices)
    test_dataset = Subset(full_dataset, test_event_indices)

    if not no_save:
        train_rses = full_dataset.true_gamma_info_df.iloc[train_event_indices][["run", "subrun", "event"]].values
        np.savetxt(f'{out_dir}/train_RSEs.txt', train_rses, fmt='%d', delimiter=' ', header='run subrun event')
        print(f"Saved training RSEs to {out_dir}/train_RSEs.txt")
        test_rses = full_dataset.true_gamma_info_df.iloc[test_event_indices][["run", "subrun", "event"]].values
        np.savetxt(f'{out_dir}/test_RSEs.txt', test_rses, fmt='%d', delimiter=' ', header='run subrun event')
        print(f"Saved test RSEs to {out_dir}/test_RSEs.txt")
    
    # Determine if pin_memory should be used (only beneficial for CUDA)
    pin_memory = torch.cuda.is_available()
    
    # Create dataloaders with batch indexing
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    return train_dataloader, test_dataloader

