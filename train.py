from tqdm import tqdm
import pickle
import argparse
import pandas as pd
import sys
import numpy as np
import torch 
import os
from datetime import datetime
from dataloader import create_dataloaders, save_train_rses_from_dataloader

# TODO: support multi-GPU training
# torchrun --standalone --nproc_per_node=4 train.py
# reference: https://github.com/karpathy/nanoGPT/blob/master/train.py

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train spacepoint SSV neural network.")
    parser.add_argument('-f', '--file', type=str, required=False, help='Path to root file to pre-process.', default='intermediate_files/downsampled_spacepoints.pkl')
    parser.add_argument('-o', '--outdir', type=str, required=False, help='Path to directory to save logs and checkpoints.')
    parser.add_argument('-n', '--num_events', type=int, required=False, help='Number of training events to use.')
    parser.add_argument('-tf', '--train_fraction', type=float, required=False, help='Fraction of training events to use.', default=0.5)
    parser.add_argument('-b', '--batch_size', type=int, required=False, help='Batch size for training.', default=32)
    parser.add_argument('-w', '--num_workers', type=int, required=False, help='Number of worker processes for data loading.', default=0)
    parser.add_argument('-ns', '--no_save', action='store_true', required=False, help='Do not save checkpoints.')
    parser.add_argument('-wb', '--wandb', action='store_true', required=False, help='Use wandb to track training.')
    args = parser.parse_args()

    out_dir = args.outdir
    if out_dir is None:
        curr_datetime = datetime.now().strftime("%Y_%m_%d-%H:%M:%S")
        out_dir = f"training_files/{curr_datetime}"

    if not args.no_save:
        os.makedirs(out_dir, exist_ok=True)

    print("Pytorch version: ", torch.__version__)
    if torch.backends.mps.is_available():
        device = torch.device("mps") # Metal Performance Shaders on M-series Macs
        print("Using Mac GPU")
    elif torch.cuda.is_available():
        device = torch.device("cuda:0") # CUDA on NVIDIA GPUs, default to first GPU
        print(f"Using CUDA GPU: {torch.cuda.get_device_name(0)}")
        # TODO: multi-GPU training:
        #for i in range(torch.cuda.device_count()):
        #    print(f"Using CUDA GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        device = torch.device("cpu") # CPU on other machines
        print("Using CPU")

    # Create dataloaders
    print("Creating dataloaders")
    train_dataloader, test_dataloader = create_dataloaders(
        pickle_file=args.file,
        batch_size=args.batch_size,
        num_events=args.num_events,
        train_fraction=args.train_fraction,
        num_workers=args.num_workers,
        shuffle=True,
        random_seed=42
    )
    
    print(f"Training batches: {len(train_dataloader)}")
    print(f"Testing batches: {len(test_dataloader)}")
    
    # Get a sample batch to determine input dimensions
    sample_batch_x, sample_batch_y = next(iter(train_dataloader))
    print(f"Sample batch shape: {sample_batch_x.shape}")
    print(f"Sample labels shape: {sample_batch_y.shape}")
    print(f"Number of classes: {len(torch.unique(sample_batch_y))}")

    # Save training RSEs
    if not args.no_save:
        print("Saving training RSEs to text file")
        save_train_rses_from_dataloader(train_dataloader, out_dir)

    # TODO: Add your neural network model definition here
    # model = YourModel(input_dim=sample_batch_x.shape[-1], num_classes=len(torch.unique(sample_batch_y)))
    # model = model.to(device)
    
    # TODO: Add your training loop here
    # Example training loop structure:
    """
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(num_epochs):
        model.train()
        for batch_x, batch_y in train_dataloader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y.long())
            loss.backward()
            optimizer.step()
    """
