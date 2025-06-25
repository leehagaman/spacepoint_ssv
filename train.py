import warnings
# Ignore FutureWarning from timm.models.layers - must be before any timm imports
warnings.filterwarnings("ignore", category=FutureWarning, module="timm.models.layers")

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
from models.my_PointTransformer_model import MultiTaskPointTransformerV3

# Import wandb for experiment tracking
import wandb

# Import additional libraries for metrics and plotting
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def train_step(model, train_dataloader, optimizer, device, epoch, args):

    model.train()
    train_loss = 0.0
    train_correct = 0
    train_global_correct = 0
    train_total = 0
    train_global_total = 0
    
    # Track batch-level metrics for wandb
    batch_losses = []
    batch_point_losses = []
    batch_event_losses = []
    batch_point_accuracies = []
    batch_global_accuracies = []
    
    train_pbar = tqdm(train_dataloader, desc=f'Epoch {epoch+1}/{args.num_epochs} [Train]')
    for batch_idx, (batch_x, batch_y, batch_global_y) in enumerate(train_pbar):
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
        
        optimizer.zero_grad()
        
        # Prepare data for MultiTaskPointTransformerV3
        coord = batch_x_reshaped.contiguous()  # [B*N, 3] - make contiguous
        batch_idx_tensor = torch.arange(B, device=device).repeat_interleave(N)  # [B*N]
        
        data_dict = {
            'coord': coord,
            'feat': coord,  # Use coordinates as initial features
            'grid_size': torch.tensor(0.1, device=device),  # Increased from 0.01 to 0.1
            'batch': batch_idx_tensor
        }
        
        # Forward pass
        predictions = model(data_dict)
        
        # Prepare targets for loss computation
        targets = {
            'point_labels': batch_y_reshaped,
            'event_labels': batch_global_y_reshaped
        }
        
        # Compute loss using model's compute_loss method
        losses = model.compute_loss(predictions, targets)
        total_loss = losses['total_loss']
        
        # Extract individual losses for logging
        point_loss = losses.get('point_loss', torch.tensor(0.0, device=device))
        event_loss = losses.get('event_loss', torch.tensor(0.0, device=device))
        
        # Statistics
        train_loss += total_loss.item()
        
        # Backward pass
        total_loss.backward()
        optimizer.step()
        
        # Statistics for point-wise predictions
        point_features = predictions['point_features']
        point_logits = point_features.feat  # [B*N, num_classes]
        _, predicted = torch.max(point_logits.data, 1)
        train_total += batch_y_reshaped.size(0)
        train_correct += (predicted == batch_y_reshaped).sum().item()
        
        # Statistics for global predictions
        if 'event_logits' in predictions:
            event_logits = predictions['event_logits']  # [B, num_event_classes]
            global_predicted = torch.argmax(event_logits, dim=1)  # [B]
            train_global_correct += (global_predicted == batch_global_y_reshaped).sum().item()
            train_global_total += B
        
        # Calculate batch-level metrics
        batch_point_accuracy = 100 * (predicted == batch_y_reshaped).sum().item() / batch_y_reshaped.size(0)
        batch_global_accuracy = 0
        if 'event_logits' in predictions:
            batch_global_accuracy = 100 * (global_predicted == batch_global_y_reshaped).sum().item() / B
        
        # Store batch metrics for wandb
        batch_losses.append(total_loss.item())
        batch_point_losses.append(point_loss.item())
        batch_event_losses.append(event_loss.item())
        batch_point_accuracies.append(batch_point_accuracy)
        batch_global_accuracies.append(batch_global_accuracy)
        
        # Log batch metrics to wandb if enabled
        if args.wandb:
            # Calculate gradient norm for monitoring
            grad_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    grad_norm += p.grad.data.norm(2).item() ** 2
            grad_norm = grad_norm ** 0.5
            
            wandb.log({
                'train/batch_loss': total_loss.item(),
                'train/batch_point_loss': point_loss.item(),
                'train/batch_event_loss': event_loss.item(),
                'train/batch_point_accuracy': batch_point_accuracy,
                'train/batch_global_accuracy': batch_global_accuracy,
                'train/learning_rate': optimizer.param_groups[0]['lr'],
                'train/gradient_norm': grad_norm,
                'train/batch': batch_idx,
                'train/epoch': epoch
            })
        
        # Update progress bar
        train_pbar.set_postfix({
            'Loss': f'{total_loss.item():.4f}',
            'Point_Loss': f'{point_loss.item():.4f}',
            'Event_Loss': f'{event_loss.item():.4f}',
            'Point_Acc': f'{100 * train_correct / train_total:.2f}%',
            'Global_Acc': f'{100 * train_global_correct / train_global_total:.2f}%' if train_global_total > 0 else 'N/A'
        })
    
    avg_train_loss = train_loss / len(train_dataloader)
    train_accuracy = 100 * train_correct / train_total
    train_global_accuracy = 100 * train_global_correct / train_global_total if train_global_total > 0 else 0

    return avg_train_loss, train_accuracy, train_global_accuracy


def test_step(model, test_dataloader, device, epoch, args, avg_train_loss, train_accuracy, train_global_accuracy, best_test_loss):

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
    
    # Track batch-level metrics for wandb
    batch_losses = []
    batch_point_losses = []
    batch_event_losses = []
    batch_point_accuracies = []
    batch_global_accuracies = []
    
    with torch.no_grad():
        test_pbar = tqdm(test_dataloader, desc=f'Epoch {epoch+1}/{args.num_epochs} [Test] ')
        for batch_idx, (batch_x, batch_y, batch_global_y) in enumerate(test_pbar):
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
            
            # Prepare data for MultiTaskPointTransformerV3
            coord = batch_x_reshaped.contiguous()  # [B*N, 3] - make contiguous
            batch_idx_tensor = torch.arange(B, device=device).repeat_interleave(N)  # [B*N]
            
            data_dict = {
                'coord': coord,
                'feat': coord,  # Use coordinates as initial features
                'grid_size': torch.tensor(0.1, device=device),  # Increased from 0.01 to 0.1
                'batch': batch_idx_tensor
            }
            
            # Forward pass
            predictions = model(data_dict)
            
            # Prepare targets for loss computation
            targets = {
                'point_labels': batch_y_reshaped,
                'event_labels': batch_global_y_reshaped
            }
            
            # Compute loss using model's compute_loss method
            losses = model.compute_loss(predictions, targets)
            total_loss = losses['total_loss']
            
            # Extract individual losses for logging
            point_loss = losses.get('point_loss', torch.tensor(0.0, device=device))
            event_loss = losses.get('event_loss', torch.tensor(0.0, device=device))
            
            # Statistics
            test_loss += total_loss.item()
            
            # Statistics for point-wise predictions
            point_features = predictions['point_features']
            point_logits = point_features.feat  # [B*N, num_classes]
            _, predicted = torch.max(point_logits.data, 1)
            test_total += batch_y_reshaped.size(0)
            test_correct += (predicted == batch_y_reshaped).sum().item()
            
            # Statistics for global predictions
            if 'event_logits' in predictions:
                event_logits = predictions['event_logits']  # [B, num_event_classes]
                global_predicted = torch.argmax(event_logits, dim=1)  # [B]
                test_correct_global += (global_predicted == batch_global_y_reshaped).sum().item()
                test_global_total += B

            test_num_gamma1_guesses += predicted.eq(0).sum().item()
            test_num_gamma2_guesses += predicted.eq(1).sum().item()
            test_num_other_particles_guesses += predicted.eq(2).sum().item()
            test_num_cosmic_guesses += predicted.eq(3).sum().item()
            
            # Calculate batch-level metrics
            batch_point_accuracy = 100 * (predicted == batch_y_reshaped).sum().item() / batch_y_reshaped.size(0)
            batch_global_accuracy = 0
            if 'event_logits' in predictions:
                batch_global_accuracy = 100 * (global_predicted == batch_global_y_reshaped).sum().item() / B
            
            # Store batch metrics for wandb
            batch_losses.append(total_loss.item())
            batch_point_losses.append(point_loss.item())
            batch_event_losses.append(event_loss.item())
            batch_point_accuracies.append(batch_point_accuracy)
            batch_global_accuracies.append(batch_global_accuracy)
            
            # Log batch metrics to wandb if enabled
            if args.wandb:
                wandb.log({
                    'test/batch_loss': total_loss.item(),
                    'test/batch_point_loss': point_loss.item(),
                    'test/batch_event_loss': event_loss.item(),
                    'test/batch_point_accuracy': batch_point_accuracy,
                    'test/batch_global_accuracy': batch_global_accuracy,
                    'test/batch': batch_idx,
                    'test/epoch': epoch
                })
            
            # Update progress bar
            test_pbar.set_postfix({
                'Loss': f'{total_loss.item():.4f}',
                'Point_Loss': f'{point_loss.item():.4f}',
                'Event_Loss': f'{event_loss.item():.4f}',
                'Point Acc': f'{100 * test_correct / test_total:.2f}%',
                'Global Acc': f'{100 * test_correct_global / test_global_total:.2f}%' if test_global_total > 0 else 'N/A'
            })
    
    avg_test_loss = test_loss / len(test_dataloader)
    test_accuracy = 100 * test_correct / test_total
    test_global_accuracy = 100 * test_correct_global / test_global_total if test_global_total > 0 else 0
    
    # Learning rate scheduling
    scheduler.step(avg_test_loss)
    
    # Calculate average guesses per event
    avg_cosmic_guesses = test_num_cosmic_guesses / num_test_events
    avg_gamma1_guesses = test_num_gamma1_guesses / num_test_events
    avg_gamma2_guesses = test_num_gamma2_guesses / num_test_events
    avg_other_particles_guesses = test_num_other_particles_guesses / num_test_events
    
    # Log epoch metrics to wandb if enabled
    if True or args.wandb:
        # Calculate confusion matrix for point classification (only every 5 epochs to save time)
        if True or epoch % 5 == 0 or epoch == args.num_epochs - 1:
            # Get all predictions and true labels for confusion matrix
            all_point_predictions = []
            all_true_point_labels = []
            
            model.eval()


            with torch.no_grad():
                for batch_x, batch_y, batch_global_y in test_dataloader:
                    batch_x = batch_x.to(device)
                    batch_y = batch_y.to(device)
                    
                    B, C, N = batch_x.shape
                    batch_x_reshaped = batch_x.transpose(1, 2).reshape(B*N, C)
                    batch_y_reshaped = batch_y.reshape(B*N)
                    
                    coord = batch_x_reshaped.contiguous()
                    batch_idx_tensor = torch.arange(B, device=device).repeat_interleave(N)
                    
                    data_dict = {
                        'coord': coord,
                        'feat': coord,
                        'grid_size': torch.tensor(0.1, device=device),
                        'batch': batch_idx_tensor
                    }
                    
                    predictions = model(data_dict)
                    point_logits = predictions['point_features'].feat
                    _, predicted = torch.max(point_logits.data, 1)
                    
                    all_point_predictions.extend(predicted.cpu().numpy())
                    all_true_point_labels.extend(batch_y_reshaped.cpu().numpy())
            
            # Create confusion matrix
            cm = confusion_matrix(all_true_point_labels, all_point_predictions)
            class_names = ['Gamma1', 'Gamma2', 'Other', 'Cosmic']
            confusion_fig, confusion_ax = plt.subplots(figsize=(8, 6))
            im = confusion_ax.imshow(cm, interpolation='nearest', cmap='Blues')
            confusion_fig.colorbar(im)
            thresh = cm.max() / 2.
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    confusion_ax.text(j, i, format(cm[i, j], 'd'),
                            ha="center", va="center",
                            color="white" if cm[i, j] > thresh else "black")
            
            confusion_ax.set_title('Point Classification Confusion Matrix')
            confusion_ax.set_ylabel('True Label')
            confusion_ax.set_xlabel('Predicted Label')
            confusion_ax.set_xticks(range(len(class_names)))
            confusion_ax.set_xticklabels(class_names)
            confusion_ax.set_yticks(range(len(class_names)))
            confusion_ax.set_yticklabels(class_names)
            confusion_fig.tight_layout()

            # create an array of 2D projections of the spacepoints
            # eight events, each with the guess on the left and the true on the right,
            # causing 16 images total in the plot array
            eight_events_predictions = []
            eight_events_true_labels = []
            eight_events_spacepoints = []
            eight_events_pred_signal_probs = []
            eight_events_true_signal = []
            with torch.no_grad():
                for batch_idx, (batch_x, batch_y, batch_global_y) in enumerate(test_dataloader):
                    batch_x = batch_x.to(device)
                    batch_y = batch_y.to(device)
                    
                    B, C, N = batch_x.shape
                    # Reshape for model input: (B, 3, 500) -> (B*500, 3) for loss calculation
                    batch_x_reshaped = batch_x.transpose(1, 2).reshape(B*N, C)  # Shape: (B*500, 3)
                    batch_y_reshaped = batch_y.reshape(B*N)  # Shape: (B*500,)
                    
                    coord = batch_x_reshaped.contiguous()  # [B*N, 3] - make contiguous
                    batch_idx_tensor = torch.arange(B, device=device).repeat_interleave(N)  # [B*N]
                    
                    data_dict = {
                        'coord': coord,
                        'feat': coord,  # Use coordinates as initial features
                        'grid_size': torch.tensor(0.1, device=device),  # Increased from 0.01 to 0.1
                        'batch': batch_idx_tensor
                    }
                    
                    predictions = model(data_dict)

                    event_logits = predictions['event_logits']
                    event_probs = torch.softmax(event_logits, dim=1)
                    event_pred_signal_prob = list(event_probs[:, 1].cpu().numpy())

                    event_true_labels = list(batch_global_y.reshape(B).cpu().numpy())

                    point_logits = predictions['point_features'].feat
                    _, predicted = torch.max(point_logits.data, 1)
                    reshaped_predicted = predicted.reshape(B, N)
                    predicted_by_event = [reshaped_predicted[i].cpu().numpy() for i in range(B)]

                    reshaped_true_labels = batch_y_reshaped.reshape(B, N)
                    true_labels_by_event = [reshaped_true_labels[i].cpu().numpy() for i in range(B)]

                    coords = data_dict['coord']
                    reshaped_coords = coords.reshape(B, N, C)
                    coords_by_event = [reshaped_coords[i].cpu().numpy() for i in range(B)]

                    eight_events_spacepoints += coords_by_event
                    eight_events_predictions += predicted_by_event
                    eight_events_true_labels += true_labels_by_event
                    eight_events_pred_signal_probs += event_pred_signal_prob
                    eight_events_true_signal += event_true_labels

                    print(f"appended one batch with {B} events")

                    if len(eight_events_spacepoints) >= 8:
                        print(f"length of eight_events_spacepoints: {len(eight_events_spacepoints)}, breaking")
                        break

            # sort the events by true signal category
            sorted_indices = sorted(range(len(eight_events_true_signal)), key=lambda x: eight_events_true_signal[x])
            eight_events_spacepoints = [eight_events_spacepoints[i] for i in sorted_indices]
            eight_events_predictions = [eight_events_predictions[i] for i in sorted_indices]
            eight_events_true_labels = [eight_events_true_labels[i] for i in sorted_indices]
            eight_events_pred_signal_probs = [eight_events_pred_signal_probs[i] for i in sorted_indices]
            eight_events_true_signal = [eight_events_true_signal[i] for i in sorted_indices]

            print(f"{eight_events_true_signal=}")

            spacepoint_fig, axs = plt.subplots(4, 4, figsize=(10, 8))
            for event_i in range(8):
                pred_col = 2 * (event_i // 4)
                true_col = 2 * (event_i // 4) + 1
                row = event_i % 4

                pred_colors = []
                true_colors = []
                for i in range(len(eight_events_spacepoints[event_i])):
                    pred_label = eight_events_predictions[event_i][i]
                    true_label = eight_events_true_labels[event_i][i]

                    if pred_label == 0:
                        pred_colors.append('green')
                    elif pred_label == 1:
                        pred_colors.append('lightgreen')
                    elif pred_label == 2:
                        pred_colors.append('brown')
                    else:
                        pred_colors.append('blue')

                    if true_label == 0:
                        true_colors.append('green')
                    elif true_label == 1:
                        true_colors.append('lightgreen')
                    elif true_label == 2:
                        true_colors.append('brown')
                    else:
                        true_colors.append('blue')

                axs[row, pred_col].scatter(eight_events_spacepoints[event_i][:, 2], eight_events_spacepoints[event_i][:, 0], s=1, c=pred_colors)
                axs[row, true_col].scatter(eight_events_spacepoints[event_i][:, 2], eight_events_spacepoints[event_i][:, 0], s=1, c=true_colors)

                axs[row, pred_col].set_xticks([])
                axs[row, pred_col].set_yticks([])
                axs[row, true_col].set_xticks([])
                axs[row, true_col].set_yticks([])

                non_true_cosmic_indices = [i for i in range(len(eight_events_spacepoints[event_i])) if eight_events_true_labels[event_i][i] != 3]
                non_true_cosmic_spacepoints = np.array([eight_events_spacepoints[event_i][i] for i in non_true_cosmic_indices])

                min_x = min(non_true_cosmic_spacepoints[:, 2])
                max_x = max(non_true_cosmic_spacepoints[:, 2])
                min_y = min(non_true_cosmic_spacepoints[:, 0])
                max_y = max(non_true_cosmic_spacepoints[:, 0])
                x_width = max_x - min_x
                y_width = max_y - min_y
                extra_scale_factor = 0.5
                axs[row, pred_col].set_xlim(min_x - extra_scale_factor * x_width, max_x + extra_scale_factor * x_width)
                axs[row, pred_col].set_ylim(min_y - extra_scale_factor * y_width, max_y + extra_scale_factor * y_width)
                axs[row, true_col].set_xlim(min_x - extra_scale_factor * x_width, max_x + extra_scale_factor * x_width)
                axs[row, true_col].set_ylim(min_y - extra_scale_factor * y_width, max_y + extra_scale_factor * y_width)

                axs[row, pred_col].text(0.95, 0.05, f"Pred Signal Prob: {eight_events_pred_signal_probs[event_i]:.2f}", transform=axs[row, pred_col].transAxes, ha='right', va='bottom')
                axs[row, true_col].text(0.95, 0.05, f"True Signal: {eight_events_true_signal[event_i]}", transform=axs[row, true_col].transAxes, ha='right', va='bottom')

            spacepoint_fig.tight_layout()
            
            wandb.log({
                'epoch': epoch,
                'train/loss': avg_train_loss,
                'train/point_accuracy': train_accuracy,
                'train/global_accuracy': train_global_accuracy,
                'test/loss': avg_test_loss,
                'test/point_accuracy': test_accuracy,
                'test/global_accuracy': test_global_accuracy,
                'learning_rate': optimizer.param_groups[0]['lr'],
                'test/avg_cosmic_guesses_per_event': avg_cosmic_guesses,
                'test/avg_gamma1_guesses_per_event': avg_gamma1_guesses,
                'test/avg_gamma2_guesses_per_event': avg_gamma2_guesses,
                'test/avg_other_particles_guesses_per_event': avg_other_particles_guesses,
                'test/confusion_matrix': wandb.Image(confusion_fig),
                'test/spacepoint_visualization': wandb.Image(spacepoint_fig),
            })
            
            plt.close(confusion_fig)
            plt.close(spacepoint_fig)
        else:
            # Log epoch metrics without extra plots and calculations
            wandb.log({
                'epoch': epoch,
                'train/loss': avg_train_loss,
                'train/point_accuracy': train_accuracy,
                'train/global_accuracy': train_global_accuracy,
                'test/loss': avg_test_loss,
                'test/point_accuracy': test_accuracy,
                'test/global_accuracy': test_global_accuracy,
                'learning_rate': optimizer.param_groups[0]['lr'],
                'test/avg_cosmic_guesses_per_event': avg_cosmic_guesses,
                'test/avg_gamma1_guesses_per_event': avg_gamma1_guesses,
                'test/avg_gamma2_guesses_per_event': avg_gamma2_guesses,
                'test/avg_other_particles_guesses_per_event': avg_other_particles_guesses,
            })
    
    # Print epoch summary
    print(f'Epoch {epoch+1}/{args.num_epochs}:')
    print(f'  Train Loss: {avg_train_loss:.4f}, Train Point Acc: {train_accuracy:.2f}%, Train Global Acc: {train_global_accuracy:.2f}%')
    print(f'  Test Loss: {avg_test_loss:.4f}, Test Point Acc: {test_accuracy:.2f}%, Test Global Acc: {test_global_accuracy:.2f}%')
    print(f'  Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')
    print(f'  Average num cosmic guesses per event: {avg_cosmic_guesses:.2f}')
    print(f'  Average num gamma1 guesses per event: {avg_gamma1_guesses:.2f}')
    print(f'  Average num gamma2 guesses per event: {avg_gamma2_guesses:.2f}')
    print(f'  Average num other particles guesses per event: {avg_other_particles_guesses:.2f}')
    
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
        
        # Log checkpoint to wandb if enabled
        if args.wandb:
            wandb.save(f'{args.outdir}/checkpoint_epoch_{epoch+1}.pth')
    
    return best_test_loss


def create_parameter_log(args, model, device, train_dataloader, test_dataloader, start_time):
    """
    Create a comprehensive parameter log file with all arguments, system info, and model details.
    """
    log_content = []
    
    # Header
    log_content.append("=" * 80)
    log_content.append("SPACEPOINT SSV TRAINING LOG")
    log_content.append("=" * 80)
    log_content.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_content.append(f"Training started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    log_content.append("")
    
    # System Information
    log_content.append("SYSTEM INFORMATION")
    log_content.append("-" * 40)
    log_content.append(f"PyTorch version: {torch.__version__}")
    log_content.append(f"Device: {device}")
    if torch.cuda.is_available():
        log_content.append(f"CUDA device: {torch.cuda.get_device_name(0)}")
        log_content.append(f"CUDA version: {torch.version.cuda}")
    log_content.append("")
    
    # Command Line Arguments
    log_content.append("COMMAND LINE ARGUMENTS")
    log_content.append("-" * 40)
    for arg, value in vars(args).items():
        log_content.append(f"{arg}: {value}")
    log_content.append("")
    
    # Model Information
    log_content.append("MODEL INFORMATION")
    log_content.append("-" * 40)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log_content.append(f"Model type: MultiTaskPointTransformerV3")
    log_content.append(f"Total parameters: {total_params:,}")
    log_content.append(f"Trainable parameters: {trainable_params:,}")
    log_content.append(f"Model size: {total_params * 4 / 1024 / 1024:.2f} MB (assuming float32)")
    log_content.append("")
    
    # Data Information
    log_content.append("DATA INFORMATION")
    log_content.append("-" * 40)
    log_content.append(f"Training batches: {len(train_dataloader)}")
    log_content.append(f"Testing batches: {len(test_dataloader)}")
    log_content.append(f"Batch size: {args.batch_size}")
    log_content.append(f"Training fraction: {args.train_fraction}")
    if args.num_events:
        log_content.append(f"Number of events: {args.num_events}")
    log_content.append("")
    
    # Training Configuration
    log_content.append("TRAINING CONFIGURATION")
    log_content.append("-" * 40)
    log_content.append(f"Number of epochs: {args.num_epochs}")
    log_content.append(f"Learning rate: 0.001")
    log_content.append(f"Weight decay: 1e-4")
    log_content.append(f"Scheduler factor: 0.5")
    log_content.append(f"Scheduler patience: 5")
    log_content.append(f"Compile mode: {args.compile}")
    log_content.append("")
    
    # Model Architecture Details
    log_content.append("MODEL ARCHITECTURE DETAILS")
    log_content.append("-" * 40)
    log_content.append(f"Number of point classes: 4 (Gamma1, Gamma2, Other, Cosmic)")
    log_content.append(f"Number of event classes: 2 (Signal 1g, Background 2g)")
    log_content.append(f"Input channels: 3 (x, y, z coordinates)")
    log_content.append(f"Grid size: 0.1")
    log_content.append(f"Event loss weight: 1.0")
    log_content.append(f"Event classification enabled: True")
    log_content.append("")
    
    # File Paths
    log_content.append("FILE PATHS")
    log_content.append("-" * 40)
    log_content.append(f"Input file: {args.file}")
    log_content.append(f"Output directory: {args.outdir}")
    log_content.append("")
    
    # Wandb Information (if enabled)
    if args.wandb:
        log_content.append("WANDB CONFIGURATION")
        log_content.append("-" * 40)
        log_content.append(f"Project: {args.wandb_project}")
        log_content.append(f"Entity: {args.wandb_entity}")
        log_content.append(f"Run name: {args.wandb_run_name}")
        log_content.append(f"Run URL: {wandb.run.get_url()}")
        log_content.append("")
    
    # Footer
    log_content.append("=" * 80)
    log_content.append("END OF TRAINING LOG")
    log_content.append("=" * 80)
    
    return "\n".join(log_content)


if __name__ == "__main__":

    # Record start time
    start_time = datetime.now()

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
    parser.add_argument('-c', '--compile', type=str, required=False, help='Compile mode for pytorch model.', default="none")
    parser.add_argument('--wandb_project', type=str, required=False, help='Wandb project name.', default='spacepoint-ssv')
    parser.add_argument('--wandb_entity', type=str, required=False, help='Wandb entity/username.', default=None)
    parser.add_argument('--wandb_run_name', type=str, required=False, help='Wandb run name.', default=None)
    args = parser.parse_args()

    if args.wandb_run_name is None:
        args.wandb_run_name = f"spacepoint-ssv-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    else:
        args.wandb = True

    if not args.no_save:
        os.makedirs(args.outdir, exist_ok=True)

    print("Pytorch version: ", torch.__version__)
    if torch.backends.mps.is_available():
        device = torch.device("mps") # Metal Performance Shaders on M-series Macs
        print("Using Mac GPU")
    elif torch.cuda.is_available():
        device = torch.device("cuda:0") # CUDA on NVIDIA GPUs, default to first GPU
        print(f"Using CUDA GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu") # CPU on other machines
        print("Using CPU")

    # Initialize wandb if enabled
    if args.wandb:
        
        # Initialize wandb
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name,
            config={
                'file': args.file,
                'outdir': args.outdir,
                'num_events': args.num_events,
                'train_fraction': args.train_fraction,
                'batch_size': args.batch_size,
                'num_epochs': args.num_epochs,
                'num_workers': args.num_workers,
                'compile_mode': args.compile,
                'model_type': 'MultiTaskPointTransformerV3',
                'num_point_classes': 4,
                'num_event_classes': 2,
                'learning_rate': 0.001,
                'weight_decay': 1e-4,
                'scheduler_factor': 0.5,
                'scheduler_patience': 5,
                'grid_size': 0.1,
                'event_loss_weight': 1.0,
                'pytorch_version': torch.__version__,
                'device': str(device),
            }
        )
        print(f"Initialized wandb run: {wandb.run.name}")

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
    
    # Determine number of classes from the data
    num_point_classes = 4
    num_event_classes = 2
    
    model = MultiTaskPointTransformerV3(
        num_classes=4,          # true gamma 1, true gamma 2, other particles, cosmic
        num_event_classes=2,    # signal 1g, background 2g
        event_loss_weight=1.0,
        in_channels=3,          # 3 coordinates (x, y, z)
        enable_event_classification=True,  # Disable for now
    )
    model = model.to(device)

    if args.compile == "none":
        print("Not compiling model")
    else:
        print(f"Compiling model for optimized performance with mode {args.compile}...")
        if args.compile not in ["none", "default", "max-autotune", "max-autotune-no-cudagraphs"]:
            raise ValueError(f"Invalid compile mode: {args.compile}")
        model = torch.compile(model, mode=args.compile)
        print("Model compilation completed")
    
    print(f"created model with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Log model info to wandb if enabled
    if args.wandb:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        wandb.config.update({
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
        })
        
        # Log model architecture
        wandb.watch(model, log="all", log_freq=100)
        
        print(f"Logged model parameters to wandb: {total_params} total, {trainable_params} trainable")
    
    # Training setup
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # Training parameters
    best_test_loss = float('inf')

    # Create training log
    parameter_log = create_parameter_log(args, model, device, train_dataloader, test_dataloader, start_time)

    # Save training log to file
    log_file_path = f"{args.outdir}/parameters.txt"
    with open(log_file_path, "w") as f:
        f.write(parameter_log)
    print(f"Parameter log saved to: {log_file_path}")

    # Upload training log to wandb if enabled
    if args.wandb:
        artifact = wandb.Artifact(
            name=f"parameters-{wandb.run.name}",
            type="parameters",
            description="Comprehensive parameters log with arguments, system info, and model details"
        )
        artifact.add_file(log_file_path)
        wandb.log_artifact(artifact)
        print("Training log uploaded to wandb as artifact")
    
    # Training loop
    print(f"Starting training for {args.num_epochs} epochs...")

    print("First, a test step with random initialization")
    best_test_loss = test_step(model, test_dataloader, device, -1, args, -1, -1, -1, best_test_loss)
    
    for epoch in range(args.num_epochs):
        avg_train_loss, train_accuracy, train_global_accuracy = train_step(model, train_dataloader, optimizer, device, epoch, args)
        best_test_loss = test_step(model, test_dataloader, device, epoch, args, avg_train_loss, train_accuracy, train_global_accuracy, best_test_loss)
    
    print("Training completed!")
    if not args.no_save:
        print(f"Best model saved to: {args.outdir}/best_model.pth")
    
    # Finish wandb run
    if args.wandb:
        wandb.finish()
        print("Wandb run completed")

