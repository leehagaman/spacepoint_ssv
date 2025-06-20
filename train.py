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
from models.my_pointnet_model import get_model, get_loss

# TODO: support multi-GPU training
# torchrun --standalone --nproc_per_node=4 train.py
# reference: https://github.com/karpathy/nanoGPT/blob/master/train.py


def train_step(model, train_dataloader, optimizer, criterion, device, epoch, args):

    model.train()
    train_loss = 0.0
    train_correct = 0
    train_global_correct = 0
    train_total = 0
    train_global_total = 0
    
    train_pbar = tqdm(train_dataloader, desc=f'Epoch {epoch+1}/{args.num_epochs} [Train]')
    for batch_x, batch_y, batch_global_y in train_pbar:
        # batch_x has shape (B, 3, 500) where B is batch size (number of events)
        # batch_y has shape (B, 500) where each row contains labels for 500 spacepoints
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        batch_global_y = batch_global_y.to(device)
        
        # Reshape for model input: (B, 3, 500) -> (B*500, 3) for loss calculation
        B, C, N = batch_x.shape
        batch_x_reshaped = batch_x.transpose(1, 2).reshape(B*N, C)  # Shape: (B*500, 3)
        batch_y_reshaped = batch_y.reshape(B*N)  # Shape: (B*500,)
        batch_global_y_reshaped = batch_global_y.reshape(B)  # Shape: (B,)
        
        # Create dummy cls_label, the model expects cls_label to be one-hot encoded with 16 classes
        # originally used for telling the model "chair", "table", etc., but we don't use it
        cls_label = torch.zeros(B, 16, dtype=torch.float32, device=device)
        cls_label[:, 0] = 1.0
        
        optimizer.zero_grad()
        
        # Forward pass
        outputs, global_outputs = model(batch_x, cls_label)  # Extract point segmentation predictions
        # The model returns outputs in shape (B, N, num_classes) which needs to be reshaped
        # Reshape outputs to (B*N, num_classes) for the loss function
        outputs = outputs.reshape(B*N, -1)
        
        # Global outputs are now binary classification logits (B, 1)
        # Global targets are binary labels (B,)
        loss, point_loss, global_loss = criterion(outputs, batch_y_reshaped, global_outputs, batch_global_y_reshaped)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        train_total += batch_y_reshaped.size(0)
        train_correct += (predicted == batch_y_reshaped).sum().item()
        
        # Global accuracy calculation
        global_predicted = (global_outputs.squeeze(1) > 0).long()  # Convert logits to binary predictions
        train_global_correct += (global_predicted == batch_global_y_reshaped).sum().item()
        train_global_total += B
        
        # Update progress bar
        train_pbar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Point_Loss': f'{point_loss.item():.4f}',
            'Global_Loss': f'{global_loss.item():.4f}',
            'Point_Acc': f'{100 * train_correct / train_total:.2f}%',
            'Global_Acc': f'{100 * train_global_correct / train_global_total:.2f}%'
        })
    
    avg_train_loss = train_loss / len(train_dataloader)
    train_accuracy = 100 * train_correct / train_total
    train_global_accuracy = 100 * train_global_correct / train_global_total

    return avg_train_loss, train_accuracy, train_global_accuracy

def test_step(model, test_dataloader, criterion, device, epoch, args, avg_train_loss, train_accuracy, train_global_accuracy, best_test_loss):

    model.eval()
    test_loss = 0.0
    test_correct = 0
    test_correct_global = 0
    test_total = 0
    test_global_total = 0

    test_num_cosmic_guesses = 0
    test_num_gamma1_guesses = 0
    test_num_gamma2_guesses = 0
    test_num_other_particles_guesses = 0

    num_test_events = 0
    
    with torch.no_grad():
        test_pbar = tqdm(test_dataloader, desc=f'Epoch {epoch+1}/{args.num_epochs} [Test] ')
        for batch_x, batch_y, batch_global_y in test_pbar:
            # batch_x has shape (B, 3, 500) where B is batch size (number of events)
            # batch_y has shape (B, 500) where each row contains labels for 500 spacepoints
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            batch_global_y = batch_global_y.to(device)

            B, C, N = batch_x.shape
            num_test_events += B
            
            # Reshape for model input: (B, 3, 500) -> (B*500, 3) for loss calculation
            batch_x_reshaped = batch_x.transpose(1, 2).reshape(B*N, C)  # Shape: (B*500, 3)
            batch_y_reshaped = batch_y.reshape(B*N)  # Shape: (B*500,)
            batch_global_y_reshaped = batch_global_y.reshape(B)  # Shape: (B,)
            
            # Create dummy cls_label, the model expects cls_label to be one-hot encoded with 16 classes
            # originally used for telling the model "chair", "table", etc., but we don't use it
            cls_label = torch.zeros(B, 16, dtype=torch.float32, device=device)
            cls_label[:, 0] = 1.0
            
            # Forward pass
            outputs, global_outputs = model(batch_x, cls_label)  # Extract point segmentation predictions
            # The model returns outputs in shape (B, N, num_classes) which needs to be reshaped
            # Reshape outputs to (B*N, num_classes) for the loss function
            outputs = outputs.reshape(B*N, -1)
                        
            loss, point_loss, global_loss = criterion(outputs, batch_y_reshaped, global_outputs, batch_global_y_reshaped)
            
            # Statistics
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            test_total += batch_y_reshaped.size(0)
            test_correct += (predicted == batch_y_reshaped).sum().item()
            
            # Global accuracy calculation
            global_predicted = (global_outputs.squeeze(1) > 0).long()  # Convert logits to binary predictions
            test_correct_global += (global_predicted == batch_global_y_reshaped).sum().item()
            test_global_total += B

            test_num_gamma1_guesses += predicted.eq(0).sum().item()
            test_num_gamma2_guesses += predicted.eq(1).sum().item()
            test_num_other_particles_guesses += predicted.eq(2).sum().item()
            test_num_cosmic_guesses += predicted.eq(3).sum().item()
            
            # Update progress bar
            test_pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Point_Loss': f'{point_loss.item():.4f}',
                'Global_Loss': f'{global_loss.item():.4f}',
                'Point Acc': f'{100 * test_correct / test_total:.2f}%',
                'Global Acc': f'{100 * test_correct_global / test_global_total:.2f}%'
            })
    
    avg_test_loss = test_loss / len(test_dataloader)
    test_accuracy = 100 * test_correct / test_total
    
    # Learning rate scheduling
    scheduler.step(avg_test_loss)
    
    # Print epoch summary
    print(f'Epoch {epoch+1}/{args.num_epochs}:')
    print(f'  Train Loss: {avg_train_loss:.4f}, Train Point Acc: {train_accuracy:.2f}%, Train Global Acc: {train_global_accuracy:.2f}%')
    print(f'  Test Loss: {avg_test_loss:.4f}, Test Point Acc: {test_accuracy:.2f}%, Test Global Acc: {100 * test_correct_global / test_global_total:.2f}%')
    print(f'  Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')
    print(f'  Average num cosmic guesses per event: {test_num_cosmic_guesses / num_test_events:.2f}')
    print(f'  Average num gamma1 guesses per event: {test_num_gamma1_guesses / num_test_events:.2f}')
    print(f'  Average num gamma2 guesses per event: {test_num_gamma2_guesses / num_test_events:.2f}')
    print(f'  Average num other particles guesses per event: {test_num_other_particles_guesses / num_test_events:.2f}')
    
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
        print(f'  Saved checkpoint to {args.outdir}/checkpoint_epoch_{epoch+1}.pth')


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train spacepoint SSV neural network.")
    parser.add_argument('-f', '--file', type=str, required=False, help='Path to root file to pre-process.', default='intermediate_files/downsampled_spacepoints.pkl')
    parser.add_argument('-o', '--outdir', type=str, required=False, help='Path to directory to save logs and checkpoints.', default=f"training_files/{datetime.now().strftime("%Y_%m_%d-%H:%M:%S")}")
    parser.add_argument('-n', '--num_events', type=int, required=False, help='Number of training events to use.')
    parser.add_argument('-tf', '--train_fraction', type=float, required=False, help='Fraction of training events to use.', default=0.5)
    parser.add_argument('-b', '--batch_size', type=int, required=False, help='Batch size for training.', default=4)
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
    sample_batch_x, sample_batch_y, sample_batch_global_y = next(iter(train_dataloader))
    
    model = get_model()
    model = model.to(device)

    print(f"created model with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Training setup
    criterion = get_loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # Training parameters
    best_test_loss = float('inf')
    
    # Training loop
    print(f"Starting training for {args.num_epochs} epochs...")

    test_step(model, test_dataloader, criterion, device, -1, args, -1, -1, -1, best_test_loss)
    
    for epoch in range(args.num_epochs):
        avg_train_loss, train_accuracy, train_global_accuracy = train_step(model, train_dataloader, optimizer, criterion, device, epoch, args)
        test_step(model, test_dataloader, criterion, device, epoch, args, avg_train_loss, train_accuracy, train_global_accuracy, best_test_loss)
    
    print("Training completed!")
    if not args.no_save:
        print(f"Best model saved to: {args.outdir}/best_model.pth")
