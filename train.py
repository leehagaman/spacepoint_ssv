from tqdm import tqdm
import pickle
import argparse
import pandas as pd
import sys
import numpy as np
import torch 
import os
from datetime import datetime
from dataloader import create_dataloaders
from model import SpacepointClassifier

# TODO: support multi-GPU training
# torchrun --standalone --nproc_per_node=4 train.py
# reference: https://github.com/karpathy/nanoGPT/blob/master/train.py

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train spacepoint SSV neural network.")
    parser.add_argument('-f', '--file', type=str, required=False, help='Path to root file to pre-process.', default='intermediate_files/downsampled_spacepoints.pkl')
    parser.add_argument('-o', '--outdir', type=str, required=False, help='Path to directory to save logs and checkpoints.', default=f"training_files/{datetime.now().strftime("%Y_%m_%d-%H:%M:%S")}")
    parser.add_argument('-n', '--num_events', type=int, required=False, help='Number of training events to use.')
    parser.add_argument('-tf', '--train_fraction', type=float, required=False, help='Fraction of training events to use.', default=0.5)
    parser.add_argument('-b', '--batch_size', type=int, required=False, help='''Batch size for training. 
                        Note that this refers to the target number of spacepoints in a batch, not the number of events.
                        If this number isn't an exact multiple of the number of spacepoints per event, the batch size will be adjusted to ensure that every batch contains only complete events.
                        ''', default=2048)
    parser.add_argument('-e', '--num_epochs', type=int, required=False, help='Number of epochs to train for.', default=20)
    parser.add_argument('-w', '--num_workers', type=int, required=False, help='Number of worker processes for data loading.', default=0)
    parser.add_argument('-ns', '--no_save', action='store_true', required=False, help='Do not save checkpoints.')
    parser.add_argument('-wb', '--wandb', action='store_true', required=False, help='Use wandb to track training.')
    args = parser.parse_args()

    if not args.no_save:
        os.makedirs(args.outdir, exist_ok=True)

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
        random_seed=42,
        out_dir=args.outdir,
        no_save=args.no_save
    )
    
    print(f"Training batches: {len(train_dataloader)}")
    print(f"Testing batches: {len(test_dataloader)}")
    
    # Get a sample batch to determine input dimensions
    sample_batch_x, sample_batch_y, sample_batch_event_indices = next(iter(train_dataloader))
    
    # Create model
    input_dim = 3 # x, y, z
    num_classes = 4 # gamma1, gamma2, other, cosmic
    
    print(f"Input dimension: {input_dim}")
    print(f"Number of classes: {num_classes}")
    
    model = SpacepointClassifier(
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_dim=128,
        num_layers=3,
        dropout=0.1
    )
    model = model.to(device)
    
    # Training setup
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # Training parameters
    best_test_loss = float('inf')
    
    # Training loop
    print(f"Starting training for {args.num_epochs} epochs...")
    
    for epoch in range(args.num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        train_pbar = tqdm(train_dataloader, desc=f'Epoch {epoch+1}/{args.num_epochs} [Train]')
        for batch_x, batch_y, batch_event_indices in train_pbar:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            batch_event_indices = batch_event_indices.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(batch_x, batch_event_indices)
            loss = criterion(outputs, batch_y)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Statistics
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += batch_y.size(0)
            train_correct += (predicted == batch_y).sum().item()
            
            # Update progress bar
            train_pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100 * train_correct / train_total:.2f}%'
            })
        
        avg_train_loss = train_loss / len(train_dataloader)
        train_accuracy = 100 * train_correct / train_total
        
        # Validation phase
        model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            test_pbar = tqdm(test_dataloader, desc=f'Epoch {epoch+1}/{args.num_epochs} [Test] ')
            for batch_x, batch_y, batch_event_indices in test_pbar:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                batch_event_indices = batch_event_indices.to(device)
                
                # Forward pass
                outputs = model(batch_x, batch_event_indices)
                loss = criterion(outputs, batch_y)
                
                # Statistics
                test_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                test_total += batch_y.size(0)
                test_correct += (predicted == batch_y).sum().item()
                
                # Update progress bar
                test_pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{100 * test_correct / test_total:.2f}%'
                })
        
        avg_test_loss = test_loss / len(test_dataloader)
        test_accuracy = 100 * test_correct / test_total
        
        # Learning rate scheduling
        scheduler.step(avg_test_loss)
        
        # Print epoch summary
        """print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'  Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}%')
        print(f'  Test Loss: {avg_test_loss:.4f}, Test Acc: {test_accuracy:.2f}%')
        print(f'  Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')"""
        
        # Save best model
        if avg_test_loss < best_test_loss and not args.no_save:
            best_test_loss = avg_test_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'test_loss': avg_test_loss,
                'train_accuracy': train_accuracy,
                'test_accuracy': test_accuracy,
            }, f'{args.outdir}/best_model.pth')
            print(f'  Saved best model with test loss: {best_test_loss:.4f}')
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0 and not args.no_save:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'test_loss': avg_test_loss,
                'train_accuracy': train_accuracy,
                'test_accuracy': test_accuracy,
            }, f'{args.outdir}/checkpoint_epoch_{epoch+1}.pth')
    
    print("Training completed!")
    if not args.no_save:
        print(f"Best model saved to: {args.outdir}/best_model.pth")
