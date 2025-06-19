#!/usr/bin/env python3
"""
Example usage of the SpacepointDataset and DataLoader.

This script demonstrates how to use the custom dataloader with a simple neural network.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from dataloader import create_dataloaders, save_train_rses


class SimpleSpacepointClassifier(nn.Module):
    """Simple neural network for spacepoint classification."""
    
    def __init__(self, input_dim=3, hidden_dim=64, num_classes=4):
        super(SimpleSpacepointClassifier, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, num_classes)
        )
    
    def forward(self, x):
        return self.network(x)


def train_model(model, train_dataloader, test_dataloader, device, num_epochs=10):
    """Train the model."""
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    train_losses = []
    test_losses = []
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_x, batch_y in train_dataloader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y.long())
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += batch_y.size(0)
            train_correct += (predicted == batch_y).sum().item()
        
        # Testing phase
        model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for batch_x, batch_y in test_dataloader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y.long())
                
                test_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                test_total += batch_y.size(0)
                test_correct += (predicted == batch_y).sum().item()
        
        # Calculate average losses and accuracies
        avg_train_loss = train_loss / len(train_dataloader)
        avg_test_loss = test_loss / len(test_dataloader)
        train_accuracy = 100 * train_correct / train_total
        test_accuracy = 100 * test_correct / test_total
        
        train_losses.append(avg_train_loss)
        test_losses.append(avg_test_loss)
        
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}%')
        print(f'Test Loss: {avg_test_loss:.4f}, Test Acc: {test_accuracy:.2f}%')
        print('-' * 50)
    
    return train_losses, test_losses


def main():
    """Main function demonstrating dataloader usage."""
    
    # Set device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Mac GPU")
    elif torch.cuda.is_available():
        device = torch.device("cuda:0")
        print(f"Using CUDA GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    
    # Create dataloaders
    print("Creating dataloaders...")
    train_dataloader, test_dataloader = create_dataloaders(
        pickle_file='intermediate_files/downsampled_spacepoints.pkl',
        batch_size=64,
        num_events=1000,  # Use only 1000 events for this example
        train_fraction=0.8,
        num_workers=0,
        shuffle=True,
        random_seed=42
    )
    
    print(f"Training batches: {len(train_dataloader)}")
    print(f"Testing batches: {len(test_dataloader)}")
    
    # Get sample batch to determine input dimensions
    sample_batch_x, sample_batch_y = next(iter(train_dataloader))
    input_dim = sample_batch_x.shape[-1]
    num_classes = len(torch.unique(sample_batch_y))
    
    print(f"Input dimension: {input_dim}")
    print(f"Number of classes: {num_classes}")
    print(f"Sample batch shape: {sample_batch_x.shape}")
    
    # Create model
    model = SimpleSpacepointClassifier(
        input_dim=input_dim,
        hidden_dim=128,
        num_classes=num_classes
    ).to(device)
    
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Train model
    print("Starting training...")
    train_losses, test_losses = train_model(
        model, train_dataloader, test_dataloader, device, num_epochs=5
    )
    
    print("Training completed!")
    
    # Save training RSEs (optional)
    try:
        save_train_rses(train_dataloader.dataset, "training_files/example_output/")
        print("Training RSEs saved")
    except Exception as e:
        print(f"Could not save RSEs: {e}")


if __name__ == "__main__":
    main() 