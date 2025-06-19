import torch
from torch.utils.data import Dataset, DataLoader, Sampler
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
        # Uncomment these if needed for other particle types
        # self.real_nu_reco_nu_downsampled_spacepoints = outputs[1]
        # self.real_nu_reco_cosmic_downsampled_spacepoints = outputs[2]
        # self.real_cosmic_reco_nu_downsampled_spacepoints = outputs[3]
        # self.real_cosmic_reco_cosmic_downsampled_spacepoints = outputs[4]
        self.real_gamma1_downsampled_spacepoints = outputs[5]
        self.real_gamma2_downsampled_spacepoints = outputs[6]
        self.real_other_particles_downsampled_spacepoints = outputs[7]
        self.real_cosmic_downsampled_spacepoints = outputs[8]

        # filter out events with no spacepoints
        total_num_spacepoints_per_event = (np.array([len(spacepoints) for spacepoints in self.real_gamma1_downsampled_spacepoints])
                                         + np.array([len(spacepoints) for spacepoints in self.real_gamma2_downsampled_spacepoints])
                                         + np.array([len(spacepoints) for spacepoints in self.real_other_particles_downsampled_spacepoints])
                                         + np.array([len(spacepoints) for spacepoints in self.real_cosmic_downsampled_spacepoints]))
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
        
        print("Creating spacepoint tensors")
        
        # Initialize lists to store all spacepoints and their metadata
        spacepoints = []
        true_labels = []
        event_indices = []
        # Get information to combine all spacepoints in all events into a single pytorch tensor
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
                event_spacepoints = np.vstack(event_spacepoints)
                spacepoints.append(event_spacepoints)
                true_labels.extend(event_true_labels)
                event_indices.extend([event_idx] * len(event_spacepoints))
        
        # Convert to tensors
        if spacepoints:
            self.x = torch.tensor(np.vstack(spacepoints), dtype=torch.float32)
            self.y = torch.tensor(true_labels, dtype=torch.long)
            self.event_indices = torch.tensor(event_indices, dtype=torch.long)
        else:
            # Handle case with no spacepoints
            self.x = torch.empty((0, 3), dtype=torch.float32)
            self.y = torch.empty((0,), dtype=torch.long)
            self.event_indices = torch.empty((0,), dtype=torch.long)
        
        print(f"Created dataset with {len(self.x)} spacepoints from {len(self.true_gamma_info_df)} events")
        
    def __len__(self):
        """Return the number of spacepoints in the dataset."""
        return len(self.x)
    
    def __getitem__(self, idx):
        """Get a specific spacepoint from the dataset.
        
        Returns:
            tuple: (spacepoint_coordinates, label, event_index)
        """
        return self.x[idx], self.y[idx], self.event_indices[idx]
    
    def get_spacepoints_by_event(self, event_idx):
        """Get all spacepoints and labels for a specific event."""
        mask = self.event_indices == event_idx
        return self.x[mask], self.y[mask]


class EventBasedBatchSampler(Sampler):
    """
    Custom batch sampler that ensures each batch contains only complete events.
    """
    
    def __init__(self, dataset, batch_size, shuffle=True):
        """
        Initialize the event-based batch sampler.
        
        Args:
            dataset: The dataset containing spacepoints with event indices
            batch_size: Target batch size (approximate, will be adjusted to ensure complete events)
            shuffle: Whether to shuffle batches between epochs
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        # Group spacepoint indices by event
        self.event_to_indices = {}
        for idx in range(len(dataset)):
            _, _, event_idx = dataset[idx]
            if event_idx.item() not in self.event_to_indices:
                self.event_to_indices[event_idx.item()] = []
            self.event_to_indices[event_idx.item()].append(idx)
        
        # Create batches that contain complete events
        self.batches = self._create_batches()
    
    def _create_batches(self):
        """Create batches ensuring each batch contains only complete events."""
        batches = []
        current_batch = []
        current_batch_size = 0
        
        event_list = list(self.event_to_indices.keys())
        
        # Shuffle events for better batch diversity
        if self.shuffle:
            np.random.shuffle(event_list)
    
        for event_idx in event_list:
            event_indices = self.event_to_indices[event_idx]
            event_size = len(event_indices)
            
            # If adding this event would exceed batch size, start a new batch
            if current_batch_size + event_size > self.batch_size and current_batch:
                batches.append(current_batch)
                current_batch = []
                current_batch_size = 0
            
            # Add this event to the current batch
            current_batch.extend(event_indices)
            current_batch_size += event_size
        
        # Add the last batch if it's not empty
        if current_batch:
            batches.append(current_batch)
        
        return batches
    
    def __iter__(self):
        """Return an iterator over the batches."""
        if self.shuffle:
            # Create a copy of batches and shuffle them
            shuffled_batches = self.batches.copy()
            np.random.shuffle(shuffled_batches)
            return iter(shuffled_batches)
        else:
            return iter(self.batches)
    
    def __len__(self):
        """Return the number of batches."""
        return len(self.batches)


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
        batch_size: Batch size for the dataloaders
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
    
    # Convert event indices back to spacepoint indices
    train_spacepoint_indices = []
    test_spacepoint_indices = []
    
    for spacepoint_idx in range(len(full_dataset)):
        x, y, event_idx = full_dataset[spacepoint_idx]
        if event_idx in train_event_indices:
            train_spacepoint_indices.append(spacepoint_idx)
        else:
            test_spacepoint_indices.append(spacepoint_idx)
    
    train_indices = torch.tensor(train_spacepoint_indices)
    test_indices = torch.tensor(test_spacepoint_indices)
    
    # Create train and test datasets using Subset
    from torch.utils.data import Subset
    train_dataset = Subset(full_dataset, train_indices)
    test_dataset = Subset(full_dataset, test_indices)

    if not no_save:
        train_rses = full_dataset.true_gamma_info_df.iloc[train_event_indices][["run", "subrun", "event"]].values
        np.savetxt(f'{out_dir}/train_RSEs.txt', train_rses, fmt='%d', delimiter=' ', header='run subrun event')
        print(f"Saved training RSEs to {out_dir}/train_RSEs.txt")
        test_rses = full_dataset.true_gamma_info_df.iloc[test_event_indices][["run", "subrun", "event"]].values
        np.savetxt(f'{out_dir}/test_RSEs.txt', test_rses, fmt='%d', delimiter=' ', header='run subrun event')
        print(f"Saved test RSEs to {out_dir}/test_RSEs.txt")
    
    # Determine if pin_memory should be used (only beneficial for CUDA)
    pin_memory = torch.cuda.is_available()
    
    # Create custom batch samplers that ensure complete events
    train_batch_sampler = EventBasedBatchSampler(train_dataset, batch_size, shuffle=True)
    test_batch_sampler = EventBasedBatchSampler(test_dataset, batch_size, shuffle=False)
    
    # Log batch information
    train_batch_sizes = [len(batch) for batch in train_batch_sampler.batches]
    test_batch_sizes = [len(batch) for batch in test_batch_sampler.batches]
    
    # Create dataloaders with custom batch samplers
    train_dataloader = DataLoader(
        train_dataset,
        batch_sampler=train_batch_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    test_dataloader = DataLoader(
        test_dataset,
        batch_sampler=test_batch_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    return train_dataloader, test_dataloader

