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
        self._load_data()
        self._preprocess_data()
        
    def _load_data(self):
        """Load data from pickle file."""
        print(f"Loading spacepoints from pickle file: {self.pickle_file}")
        with open(self.pickle_file, "rb") as f:
            outputs = pickle.load(f)
        
        self.true_gamma_info_df = outputs[0]
        # Uncomment these if needed for other particle types
        # self.real_nu_reco_nu_downsampled_spacepoints = outputs[1]
        # self.real_nu_reco_cosmic_downsampled_spacepoints = outputs[2]
        # self.real_cosmic_reco_nu_downsampled_spacepoints = outputs[3]
        # self.real_cosmic_reco_cosmic_downsampled_spacepoints = outputs[4]
        self.real_gamma1_downsampled_spacepoints = outputs[5]
        self.real_gamma2_downsampled_spacepoints = outputs[6]
        self.real_other_particles_downsampled_spacepoints = outputs[7]
        self.real_cosmic_downsampled_spacepoints = outputs[8]
        
        print("Shuffling event ordering")
        # shuffle the event ordering, the same for true_gamma_info_df and the spacepoint data
        # First, get the shuffled indices
        shuffled_indices = self.true_gamma_info_df.sample(frac=1, random_state=self.random_seed).index
        # Then shuffle the dataframe
        self.true_gamma_info_df = self.true_gamma_info_df.iloc[shuffled_indices].reset_index(drop=True)
        # Shuffle the spacepoint data using the same indices
        self.real_gamma1_downsampled_spacepoints = [self.real_gamma1_downsampled_spacepoints[i] for i in shuffled_indices]
        self.real_gamma2_downsampled_spacepoints = [self.real_gamma2_downsampled_spacepoints[i] for i in shuffled_indices]
        self.real_other_particles_downsampled_spacepoints = [self.real_other_particles_downsampled_spacepoints[i] for i in shuffled_indices]
        self.real_cosmic_downsampled_spacepoints = [self.real_cosmic_downsampled_spacepoints[i] for i in shuffled_indices]
        
    def _preprocess_data(self):
        """Preprocess and concatenate spacepoint data."""
        print("Creating spacepoint tensors")
        
        # Convert lists of numpy arrays to single numpy arrays first, then to tensors
        # This avoids the slow tensor creation warning and handles variable lengths
        
        # Convert each list of arrays to a single numpy array
        gamma1_data = np.vstack(self.real_gamma1_downsampled_spacepoints) if self.real_gamma1_downsampled_spacepoints else np.empty((0, 3))
        gamma2_data = np.vstack(self.real_gamma2_downsampled_spacepoints) if self.real_gamma2_downsampled_spacepoints else np.empty((0, 3))
        other_particles_data = np.vstack(self.real_other_particles_downsampled_spacepoints) if self.real_other_particles_downsampled_spacepoints else np.empty((0, 3))
        cosmic_data = np.vstack(self.real_cosmic_downsampled_spacepoints) if self.real_cosmic_downsampled_spacepoints else np.empty((0, 3))
        
        # Convert to tensors
        x = torch.tensor(gamma1_data, dtype=torch.float32)
        x = torch.cat([x, torch.tensor(gamma2_data, dtype=torch.float32)], dim=0)
        x = torch.cat([x, torch.tensor(other_particles_data, dtype=torch.float32)], dim=0)
        x = torch.cat([x, torch.tensor(cosmic_data, dtype=torch.float32)], dim=0)
        
        # Create labels for each category
        y = torch.zeros(gamma1_data.shape[0], dtype=torch.long)
        y = torch.cat([y, torch.ones(gamma2_data.shape[0], dtype=torch.long)], dim=0)
        y = torch.cat([y, torch.ones(other_particles_data.shape[0], dtype=torch.long) * 2], dim=0)
        y = torch.cat([y, torch.ones(cosmic_data.shape[0], dtype=torch.long) * 3], dim=0)
        
        # Create original indices to track the relationship with RSE data
        # Each spacepoint corresponds to an event in the original dataframe
        gamma1_indices = np.repeat(np.arange(len(self.real_gamma1_downsampled_spacepoints)), 
                                 [len(arr) for arr in self.real_gamma1_downsampled_spacepoints])
        gamma2_indices = np.repeat(np.arange(len(self.real_gamma2_downsampled_spacepoints)), 
                                 [len(arr) for arr in self.real_gamma2_downsampled_spacepoints])
        other_indices = np.repeat(np.arange(len(self.real_other_particles_downsampled_spacepoints)), 
                                [len(arr) for arr in self.real_other_particles_downsampled_spacepoints])
        cosmic_indices = np.repeat(np.arange(len(self.real_cosmic_downsampled_spacepoints)), 
                                 [len(arr) for arr in self.real_cosmic_downsampled_spacepoints])
        
        # Concatenate all indices
        original_indices = np.concatenate([gamma1_indices, gamma2_indices, other_indices, cosmic_indices])
        
        # Shuffle the data using the random seed
        generator = torch.Generator().manual_seed(self.random_seed)
        perm = torch.randperm(x.shape[0], generator=generator)
        self.x = x[perm]
        self.y = y[perm]
        self.original_indices = original_indices[perm.numpy()]
        
        # Limit number of events if specified
        if self.num_events is None:
            self.num_events = self.true_gamma_info_df.shape[0]
        self.x = self.x[:self.num_events]
        self.y = self.y[:self.num_events]
        self.original_indices = self.original_indices[:self.num_events]
            
    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.x)
    
    def __getitem__(self, idx):
        """Get a single sample from the dataset."""
        return self.x[idx], self.y[idx]
    
    def save_train_rses(self, out_dir):
        """Save RSE (run, subrun, event) information for training events."""
        train_rses = self.true_gamma_info_df[["run", "subrun", "event"]].values
        np.savetxt(f'{out_dir}/train_RSEs.txt', train_rses, fmt='%d', delimiter=' ', header='run subrun event')
        print(f"Saved training RSEs to {out_dir}/train_RSEs.txt")


def create_dataloaders(pickle_file,
                      batch_size = 32,
                      num_events = None,
                      train_fraction = 0.5,
                      num_workers = 0,
                      shuffle = True,
                      random_seed = 42):
    """
    Create train and test dataloaders.
    
    Args:
        pickle_file: Path to the pickle file containing spacepoint data
        batch_size: Batch size for the dataloaders
        num_events: Number of events to use (None for all events)
        train_fraction: Fraction of data to use for training
        num_workers: Number of worker processes for data loading
        shuffle: Whether to shuffle the training data
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
    
    # Create train/test splits from the full dataset
    num_training_events = int(len(full_dataset) * train_fraction)
    
    # Shuffle events
    generator = torch.Generator().manual_seed(random_seed)
    perm = torch.randperm(len(full_dataset), generator=generator)
    
    train_indices = perm[:num_training_events]
    test_indices = perm[num_training_events:]
    
    # Create train and test datasets using Subset
    from torch.utils.data import Subset
    train_dataset = Subset(full_dataset, train_indices)
    test_dataset = Subset(full_dataset, test_indices)
    
    # Determine if pin_memory should be used (only beneficial for CUDA)
    pin_memory = torch.cuda.is_available()
    
    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
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


def save_train_rses_from_dataloader(train_dataloader, out_dir):
    """Save training RSEs from a train dataloader that uses Subset."""
    from torch.utils.data import Subset
    
    if isinstance(train_dataloader.dataset, Subset):
        # Get the underlying full dataset
        full_dataset = train_dataloader.dataset.dataset
        # Get the indices used by the Subset
        subset_indices = train_dataloader.dataset.indices
        # Get the original dataframe indices that correspond to the subset
        shuffled_original_indices = full_dataset.original_indices[subset_indices]
        train_rses = full_dataset.true_gamma_info_df.iloc[shuffled_original_indices][["run", "subrun", "event"]].values
        np.savetxt(f'{out_dir}/train_RSEs.txt', train_rses, fmt='%d', delimiter=' ', header='run subrun event')
        print(f"Saved training RSEs to {out_dir}/train_RSEs.txt")
    else:
        # Fallback to the old method
        train_dataloader.dataset.save_train_rses(out_dir)
