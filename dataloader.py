import torch
from torch.utils.data import Dataset, DataLoader, Sampler
import pickle
import numpy as np
import pandas as pd


class SpacepointDataset(Dataset):
    
    def __init__(self, 
                 pickle_file,
                 num_events,
                 random_seed,
                 spacepoints_type,
                 rng):
        """
        Initialize the dataset.
        
        Args:
            pickle_file: Path to the pickle file containing spacepoint data
            num_events: Number of events to use (None for all events)
            random_seed: Random seed for reproducible shuffling
            spacepoints_type: Type of training to perform
                'all_points': Use all points
                'only_photons': Use only photons
                'only_neutrinos': Use only neutrinos
        """

        self.pickle_file = pickle_file
        self.num_events = num_events
        self.random_seed = random_seed
        self.spacepoints_type = spacepoints_type
        self.rng = rng
        
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

        # 0 spacepoints if failed WC generic selection, also potentially could have fewer than 500 spacepoints for a very low-charge image
        total_num_spacepoints_per_event = (np.array([len(spacepoints) for spacepoints in self.real_gamma1_downsampled_spacepoints])
                                         + np.array([len(spacepoints) for spacepoints in self.real_gamma2_downsampled_spacepoints])
                                         + np.array([len(spacepoints) for spacepoints in self.real_other_particles_downsampled_spacepoints])
                                         + np.array([len(spacepoints) for spacepoints in self.real_cosmic_downsampled_spacepoints]))
        only_photons_num_spacepoints_per_event = (np.array([len(spacepoints) for spacepoints in self.real_gamma1_downsampled_spacepoints])
                                                  + np.array([len(spacepoints) for spacepoints in self.real_gamma2_downsampled_spacepoints]))
        only_neutrinos_num_spacepoints_per_event = (np.array([len(spacepoints) for spacepoints in self.real_cosmic_downsampled_spacepoints]))

        if self.spacepoints_type == "all_points":
            enough_spacepoints_mask = total_num_spacepoints_per_event > 0
        elif self.spacepoints_type == "only_photons":
            enough_spacepoints_mask = only_photons_num_spacepoints_per_event > 0
        elif self.spacepoints_type == "only_neutrinos":
            enough_spacepoints_mask = only_neutrinos_num_spacepoints_per_event > 0
        else:
            raise ValueError(f"Invalid training type: {self.spacepoints_type}")

        # Require one or two true primary/pi0 gammas pair converting in the final volume
        sig_mask = self.true_gamma_info_df["true_num_gamma_pairconvert_in_FV"].values == 1
        bkg_mask = self.true_gamma_info_df["true_num_gamma_pairconvert_in_FV"].values == 2

        process_mask = enough_spacepoints_mask & (sig_mask | bkg_mask)
        process_indices = np.where(process_mask)[0]

        self.true_gamma_info_df = self.true_gamma_info_df.iloc[process_indices].reset_index(drop=True)
        self.real_gamma1_downsampled_spacepoints = [self.real_gamma1_downsampled_spacepoints[i] for i in process_indices]
        self.real_gamma2_downsampled_spacepoints = [self.real_gamma2_downsampled_spacepoints[i] for i in process_indices]
        self.real_other_particles_downsampled_spacepoints = [self.real_other_particles_downsampled_spacepoints[i] for i in process_indices]
        self.real_cosmic_downsampled_spacepoints = [self.real_cosmic_downsampled_spacepoints[i] for i in process_indices]

        if self.num_events is None:
            self.num_events = self.true_gamma_info_df.shape[0]

        if self.num_events < len(self.true_gamma_info_df):
            print(f"Restricting to first {self.num_events} events")
            self.true_gamma_info_df = self.true_gamma_info_df.iloc[:self.num_events]
            self.real_gamma1_downsampled_spacepoints = self.real_gamma1_downsampled_spacepoints[:self.num_events]
            self.real_gamma2_downsampled_spacepoints = self.real_gamma2_downsampled_spacepoints[:self.num_events]
            self.real_other_particles_downsampled_spacepoints = self.real_other_particles_downsampled_spacepoints[:self.num_events]
            self.real_cosmic_downsampled_spacepoints = self.real_cosmic_downsampled_spacepoints[:self.num_events]

        if self.num_events > len(self.true_gamma_info_df):
            raise ValueError(f"num_events ({self.num_events}) is greater than the number of qualifying events in the dataset ({len(self.true_gamma_info_df)})")

        if self.spacepoints_type == "all_points":
            pass
        elif self.spacepoints_type == "only_photons":
            self.real_other_particles_downsampled_spacepoints = [np.empty((0, 3)) for i in range(len(self.real_other_particles_downsampled_spacepoints))]
            self.real_cosmic_downsampled_spacepoints = [np.empty((0, 3)) for i in range(len(self.real_cosmic_downsampled_spacepoints))]
        elif self.spacepoints_type == "only_neutrinos":
            self.real_other_particles_downsampled_spacepoints = [np.empty((0, 3)) for i in range(len(self.real_other_particles_downsampled_spacepoints))]
        else:
            raise ValueError(f"Invalid training type: {self.spacepoints_type}")
        
        print("Shuffling event ordering")
        shuffled_indices = torch.randperm(self.num_events, generator=self.rng)
        print("max of shuffled indices", shuffled_indices.max())
        print("len of true_gamma_info_df", len(self.true_gamma_info_df))
        self.true_gamma_info_df = self.true_gamma_info_df.iloc[shuffled_indices].reset_index(drop=True)
        self.real_gamma1_downsampled_spacepoints = [self.real_gamma1_downsampled_spacepoints[i] for i in shuffled_indices]
        self.real_gamma2_downsampled_spacepoints = [self.real_gamma2_downsampled_spacepoints[i] for i in shuffled_indices]
        self.real_other_particles_downsampled_spacepoints = [self.real_other_particles_downsampled_spacepoints[i] for i in shuffled_indices]
        self.real_cosmic_downsampled_spacepoints = [self.real_cosmic_downsampled_spacepoints[i] for i in shuffled_indices]

        self.event_y = torch.tensor(self.true_gamma_info_df["true_num_gamma_pairconvert_in_FV"].values == 1, dtype=torch.long)
        
        # Extract true pair conversion coordinates
        print("Extracting true pair conversion coordinates")
        self.pair_conversion_coords = []
        for event_idx in range(self.num_events):
            # Get the pair conversion coordinates for this event
            xs = self.true_gamma_info_df.iloc[event_idx]["true_gamma_pairconversion_xs"]
            ys = self.true_gamma_info_df.iloc[event_idx]["true_gamma_pairconversion_ys"]
            zs = self.true_gamma_info_df.iloc[event_idx]["true_gamma_pairconversion_zs"]
            
            # Combine into a list of [x, y, z] coordinates
            event_pair_coords = []
            for i in range(len(xs)):
                event_pair_coords.append([xs[i], ys[i], zs[i]])
            
            self.pair_conversion_coords.append(event_pair_coords)
        
        print("Creating spacepoint tensors")
        
        # Create tensors for each event in the format expected by PointNet2: (B, C, N)
        # where B = batch size (events), C = channels (3 for xyz), N = points per event (500)
        self.x = []  # List to store spacepoint tensors for each event
        self.y = []  # List to store label tensors for each event
        
        for event_idx in range(self.num_events):
            event_spacepoints = []
            event_true_labels = []
            
            if len(self.real_gamma1_downsampled_spacepoints[event_idx]) > 0:
                event_spacepoints.extend(self.real_gamma1_downsampled_spacepoints[event_idx])
                event_true_labels.extend([0] * len(self.real_gamma1_downsampled_spacepoints[event_idx]))
            if len(self.real_gamma2_downsampled_spacepoints[event_idx]) > 0:
                event_spacepoints.extend(self.real_gamma2_downsampled_spacepoints[event_idx])
                event_true_labels.extend([1] * len(self.real_gamma2_downsampled_spacepoints[event_idx]))
            if len(self.real_other_particles_downsampled_spacepoints[event_idx]) > 0:
                event_spacepoints.extend(self.real_other_particles_downsampled_spacepoints[event_idx])
                event_true_labels.extend([2] * len(self.real_other_particles_downsampled_spacepoints[event_idx]))
            if len(self.real_cosmic_downsampled_spacepoints[event_idx]) > 0:
                event_spacepoints.extend(self.real_cosmic_downsampled_spacepoints[event_idx])
                event_true_labels.extend([3] * len(self.real_cosmic_downsampled_spacepoints[event_idx]))

            assert len(event_spacepoints) == len(event_true_labels), f"Number of spacepoints ({len(event_spacepoints)}) and labels ({len(event_true_labels)}) do not match for event {event_idx}"

            if len(event_spacepoints) == 0:
                raise ValueError(f"Event {event_idx} has no spacepoints")
            
            # Combine all spacepoints for this event
            event_spacepoints = np.vstack(event_spacepoints)  # Shape: (N, 3) where N can vary
            event_true_labels = np.array(event_true_labels)

            # Convert to tensor and transpose to (3, N) format
            event_tensor = torch.tensor(event_spacepoints, dtype=torch.float32).transpose(0, 1)  # Shape: (3, N)
            self.x.append(event_tensor)
            self.y.append(torch.tensor(event_true_labels, dtype=torch.long))
        
        print(f"Created dataset with {len(self.x)} events")
        
    def __len__(self):
        """Return the number of events in the dataset."""
        return len(self.x)
    
    def __getitem__(self, idx):
        """Get a specific event from the dataset.
        
        Returns:
            tuple: (spacepoint_coordinates, labels, event_label, pair_conversion_coords)
            spacepoint_coordinates: tensor of shape (3, N) for xyz coordinates where N can vary
            labels: tensor of shape (N,) for point labels where N can vary
            event_label: tensor of shape () for event-level label
            pair_conversion_coords: list of [x, y, z] coordinates for true pair conversion points
        """
        return self.x[idx], self.y[idx], self.event_y[idx], self.pair_conversion_coords[idx]


class PointBasedBatchSampler(Sampler):
    """
    Custom batch sampler that creates batches based on maximum number of points.
    This allows for more efficient batching when events have variable numbers of points.
    """
    
    def __init__(self, dataset, max_points_per_batch, shuffle=True):
        """
        Initialize the batch sampler.
        
        Args:
            dataset: The dataset to sample from
            max_points_per_batch: Maximum number of points per batch
            shuffle: Whether to shuffle the events
        """
        self.dataset = dataset
        self.max_points_per_batch = max_points_per_batch
        self.shuffle = shuffle
        
        # Get the number of points for each event
        self.event_sizes = []
        for i in range(len(dataset)):
            _, labels, _, _ = dataset[i]
            self.event_sizes.append(len(labels))
        
        self.event_sizes = np.array(self.event_sizes)
        self.event_indices = np.arange(len(dataset))
        
        # Create batches
        self.batches = self._create_batches()
    
    def _create_batches(self):
        """Create batches based on maximum points per batch."""
        batches = []
        current_batch = []
        current_batch_points = 0
        
        # Shuffle events if requested
        if self.shuffle:
            np.random.shuffle(self.event_indices)
        
        for event_idx in self.event_indices:
            event_points = self.event_sizes[event_idx]
            
            # If adding this event would exceed the max points, start a new batch
            if current_batch_points + event_points > self.max_points_per_batch and current_batch:
                batches.append(current_batch)
                current_batch = [event_idx]
                current_batch_points = event_points
            else:
                current_batch.append(event_idx)
                current_batch_points += event_points
        
        # Add the last batch if it's not empty
        if current_batch:
            batches.append(current_batch)
        
        # Log batch statistics
        batch_sizes = [len(batch) for batch in batches]
        batch_point_counts = [sum(self.event_sizes[batch]) for batch in batches]
        print(f"Created {len(batches)} batches with max {self.max_points_per_batch} points per batch")
        print(f"Average batch size: {np.mean(batch_sizes):.1f} events, {np.mean(batch_point_counts):.1f} points")
        print(f"Batch size range: {min(batch_sizes)}-{max(batch_sizes)} events, {min(batch_point_counts)}-{max(batch_point_counts)} points")
        
        return batches
    
    def __iter__(self):
        """Return an iterator over batches."""
        if self.shuffle:
            # Recreate batches with new shuffling
            self.batches = self._create_batches()
        return iter(self.batches)
    
    def __len__(self):
        """Return the number of batches."""
        return len(self.batches)


def custom_collate_fn(batch):
    """
    Custom collate function that concatenates all points from all events in a batch.
    This is the most efficient approach for variable-sized point clouds.
    
    Args:
        batch: List of tuples (spacepoint_coordinates, labels, event_label, pair_conversion_coords)
    
    Returns:
        Tuple of (batch_coords, batch_labels, batch_event_labels, batch_indices, batch_pair_conversion_coords)
        batch_coords: tensor of shape (total_points, 3) - all points concatenated
        batch_labels: tensor of shape (total_points,) - all labels concatenated
        batch_event_labels: tensor of shape (batch_size,) - event-level labels
        batch_indices: tensor of shape (total_points,) - batch index for each point
        batch_pair_conversion_coords: list of lists of [x, y, z] coordinates for each event in batch
    """
    spacepoints, labels, event_labels, pair_conversion_coords = zip(*batch)
    
    # Concatenate all points and labels
    all_coords = []
    all_labels = []
    all_batch_indices = []
    
    for i, (sp, lab) in enumerate(zip(spacepoints, labels)):
        num_points = sp.shape[1]
        all_coords.append(sp.transpose(0, 1))  # (3, N) -> (N, 3)
        all_labels.append(lab)
        all_batch_indices.append(torch.full((num_points,), i, dtype=torch.long))
    
    # Concatenate everything
    batch_coords = torch.cat(all_coords, dim=0)  # (total_points, 3)
    batch_labels = torch.cat(all_labels, dim=0)  # (total_points,)
    batch_indices = torch.cat(all_batch_indices, dim=0)  # (total_points,)
    batch_event_labels = torch.stack(event_labels, dim=0)  # (batch_size,)
    batch_pair_conversion_coords = list(pair_conversion_coords)  # list of lists
    
    return batch_coords, batch_labels, batch_event_labels, batch_indices, batch_pair_conversion_coords


def create_dataloaders(pickle_file,
                      batch_size,
                      num_events,
                      train_fraction,
                      num_workers,
                      random_seed,
                      out_dir,
                      no_save,
                      spacepoints_type,
                      rng):
    """
    Create train and test dataloaders.
    
    Args:
        pickle_file: Path to the pickle file containing spacepoint data
        batch_size: Maximum number of points per batch (not number of events)
        num_events: Number of events to use (None for all events)
        train_fraction: Fraction of data to use for training
        num_workers: Number of worker processes for data loading
        random_seed: Random seed for reproducible splits
        spacepoints_type: Type of training to perform
            'all_points': Use all points
            'only_photons': Use only photons
            'only_neutrinos': Use only neutrinos
        
    Returns:
        Tuple of (train_dataloader, test_dataloader, num_train_events, num_test_events)
    """
    
    # Create a single dataset with all data
    full_dataset = SpacepointDataset(
        pickle_file=pickle_file,
        num_events=num_events,
        random_seed=random_seed,
        spacepoints_type=spacepoints_type,
        rng=rng
    )
    
    # Get the actual number of events after filtering
    actual_num_events = len(full_dataset)
    
    # Get unique events and create event-based splits
    num_train_events = int(actual_num_events * train_fraction)
    num_test_events = actual_num_events - num_train_events
    
    # Shuffle events
    generator = torch.Generator().manual_seed(random_seed)
    event_perm = torch.randperm(actual_num_events, generator=generator)
    
    train_event_indices = event_perm[:num_train_events]
    test_event_indices = event_perm[num_train_events:]
    
    # Create train and test datasets using Subset
    from torch.utils.data import Subset
    train_dataset = Subset(full_dataset, train_event_indices)
    test_dataset = Subset(full_dataset, test_event_indices)

    if not no_save:
        # Use the actual indices from the filtered dataset
        train_rses = full_dataset.true_gamma_info_df.iloc[train_event_indices][["run", "subrun", "event"]].values
        np.savetxt(f'{out_dir}/train_RSEs.txt', train_rses, fmt='%d', delimiter=' ', header='run subrun event')
        print(f"Saved training RSEs to {out_dir}/train_RSEs.txt")
        test_rses = full_dataset.true_gamma_info_df.iloc[test_event_indices][["run", "subrun", "event"]].values
        np.savetxt(f'{out_dir}/test_RSEs.txt', test_rses, fmt='%d', delimiter=' ', header='run subrun event')
        print(f"Saved test RSEs to {out_dir}/test_RSEs.txt")
    
    # Determine if pin_memory should be used (only beneficial for CUDA)
    pin_memory = torch.cuda.is_available()
    
    # Create point-based batch samplers
    print(f"Creating point-based batch samplers with max {batch_size} points per batch")
    train_batch_sampler = PointBasedBatchSampler(
        train_dataset, 
        max_points_per_batch=batch_size, 
        shuffle=True
    )
    
    test_batch_sampler = PointBasedBatchSampler(
        test_dataset, 
        max_points_per_batch=batch_size, 
        shuffle=False
    )
    
    # Create dataloaders with point-based batching
    train_dataloader = DataLoader(
        train_dataset,
        batch_sampler=train_batch_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=custom_collate_fn
    )
    
    test_dataloader = DataLoader(
        test_dataset,
        batch_sampler=test_batch_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=custom_collate_fn
    )
    
    return train_dataloader, test_dataloader, num_train_events, num_test_events

