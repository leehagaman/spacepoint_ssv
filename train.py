import warnings
# Ignore FutureWarning from timm.models.layers - must be before any timm imports
warnings.filterwarnings("ignore", category=FutureWarning, module="timm.models.layers")
# Ignore RuntimeWarning from numpy histogram divide operations
warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in divide")

from tqdm import tqdm
import pickle
import argparse
import pandas as pd
import sys
import numpy as np
import torch 
import os
from datetime import datetime
import math
import json
import wandb

from dataloader import create_dataloaders
from models.my_PointTransformer_model import MultiTaskPointTransformerV3

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def train_step(model, train_dataloader, optimizer, device, epoch, args):

    model.train()

    total_train_loss = 0.0
    total_train_point_loss = 0.0
    total_train_event_loss = 0.0
    total_train_correct_points = 0
    total_train_num_points = 0

    num_train_events = 0
    
    # Track batch-level metrics for wandb
    batch_losses = []
    batch_point_losses = []
    batch_event_losses = []
    batch_point_accuracies = []
    
    train_pbar = tqdm(train_dataloader, desc=f'Epoch {epoch+1}/{args.num_epochs} [Train]')
    for batch_idx, (batch_features, batch_labels, batch_event_y, batch_indices, batch_pair_conversion_coords) in enumerate(train_pbar):
        # batch_coords has shape (total_points, 3) - all points from all events concatenated
        # batch_labels has shape (total_points,) - all labels concatenated
        # batch_event_y has shape (batch_size,) - event-level labels
        # batch_indices has shape (total_points,) - batch index for each point
        # batch_pair_conversion_coords is a list of lists of [x, y, z] coordinates for each event in batch
        batch_features = batch_features.to(device)
        batch_labels = batch_labels.to(device)
        batch_event_y = batch_event_y.to(device)
        batch_indices = batch_indices.to(device)
        
        B = batch_event_y.size(0)  # Number of events in this batch
        
        optimizer.zero_grad()

        if batch_features.shape[1] == 4: # including charge
            batch_coords = batch_features[:, :3]
        else:
            batch_coords = batch_features
        
        coord = batch_coords.contiguous()
        feat = batch_features.contiguous()
        
        data_dict = {
            'coord': coord,
            'feat': feat, # either xyz or xyzq
            'grid_size': torch.tensor(args.model_settings['grid_size'], device=device),
            'batch': batch_indices  # [total_points] - tells model which points belong to which event
        }
        
        # Forward pass
        predictions = model(data_dict)
        
        # Prepare targets for loss computation
        targets = {
            'point_labels': batch_labels,
            'event_labels': batch_event_y,
        }
        
        # Compute loss using model's compute_loss method
        losses = model.compute_loss(predictions, targets, coord)
        total_loss = losses['total_loss']
        
        # Extract individual losses for logging
        point_loss = losses.get('point_loss', torch.tensor(0.0, device=device))
        event_loss = losses.get('event_loss', torch.tensor(0.0, device=device))
        
        # These are total losses, not average per-event losses
        total_train_loss += total_loss.item() * B
        total_train_point_loss += point_loss.item() * B
        total_train_event_loss += event_loss.item() * B
        
        # Backward pass
        total_loss.backward()
        optimizer.step()
        
        # Statistics for point-wise predictions
        point_features = predictions['point_features']
        point_logits = point_features.feat  # [total_points, num_classes]
        _, predicted = torch.max(point_logits.data, 1)
        
        total_train_num_points += batch_labels.size(0)
        total_train_correct_points += (predicted == batch_labels).sum().item()

        num_train_events += B
        
        # Calculate batch-level metrics
        batch_point_accuracy = 100 * (predicted == batch_labels).sum().item() / batch_labels.size(0)
        
        # Store batch metrics for wandb
        batch_losses.append(total_loss.item())
        batch_point_losses.append(point_loss.item())
        batch_event_losses.append(event_loss.item())
        batch_point_accuracies.append(batch_point_accuracy)
        
        # Log batch metrics to wandb if enabled
        if args.wandb:
            # Calculate gradient norm for monitoring
            grad_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    grad_norm += p.grad.data.norm(2).item() ** 2
            grad_norm = grad_norm ** 0.5
            
            wandb.log({
                'train/batch_loss': np.sum(batch_losses) / num_train_events,
                'train/batch_point_loss': np.mean(batch_point_losses),
                'train/batch_event_loss': np.mean(batch_event_losses),
                'train/batch_point_accuracy': np.mean(batch_point_accuracies),
                'train/batch_gradient_norm': grad_norm,
                'train/batch_id': batch_idx,
            })
        
        # Update progress bar
        train_pbar.set_postfix({
            'Loss': f'{total_loss.item():.4f}',
            'Point_Loss': f'{point_loss.item():.4f}',
            'Event_Loss': f'{event_loss.item():.4f}',
        })

    loss = total_train_loss / num_train_events
    point_loss = total_train_point_loss / num_train_events
    event_loss = total_train_event_loss / num_train_events
    point_accuracy = total_train_correct_points / total_train_num_points

    return loss, point_loss, event_loss, point_accuracy


def test_step(model, test_dataloader, device, epoch, args):

    model.eval()
    total_test_loss = 0.0
    total_test_point_loss = 0.0
    total_test_event_loss = 0.0
    total_test_correct_points = 0
    total_test_num_points = 0

    num_test_events = 0
    
    # Track batch-level metrics for wandb
    batch_losses = []
    batch_point_losses = []
    batch_event_losses = []
    batch_point_accuracies = []
    
    # For point-level confusion matrix and point category histogram
    all_point_predictions = []
    all_point_true_labels = []
    all_point_probabilities = []  # Store predicted probabilities for each point

    # for event-level score histogram
    all_event_probs = []
    all_event_true_labels = []
    
    # for spacepoint-level visualization
    num_spacepoint_plot_events = 8
    subset_sig_point_predictions = []
    subset_sig_point_true_labels = []
    subset_sig_spacepoints = []
    subset_sig_event_predictions = []
    subset_sig_event_true_labels = []
    subset_sig_pair_conversion_coords = []
    subset_bkg_point_predictions = []
    subset_bkg_point_true_labels = []
    subset_bkg_spacepoints = []
    subset_bkg_event_predictions = []
    subset_bkg_event_true_labels = []
    subset_bkg_pair_conversion_coords = []

    with torch.no_grad():
        test_pbar = tqdm(test_dataloader, desc=f'Epoch {epoch+1}/{args.num_epochs} [Test] ')
        for batch_idx, (batch_features, batch_labels, batch_event_y, batch_indices, batch_pair_conversion_coords) in enumerate(test_pbar):
            # batch_coords has shape (total_points, 3) - all points from all events concatenated
            # batch_labels has shape (total_points,) - all labels concatenated
            # batch_event_y has shape (batch_size,) - event-level labels
            # batch_indices has shape (total_points,) - batch index for each point
            # batch_pair_conversion_coords is a list of lists of [x, y, z] coordinates for each event in batch
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)
            batch_event_y = batch_event_y.to(device)
            batch_indices = batch_indices.to(device)

            B = batch_event_y.size(0)  # Number of events in this batch
            num_test_events += B

            if batch_features.shape[1] == 4: # including charge
                batch_coords = batch_features[:, :3]
            else:
                batch_coords = batch_features
            
            coord = batch_coords.contiguous()
            feat = batch_features.contiguous()
            
            data_dict = {
                'coord': coord,
                'feat': feat,
                'grid_size': torch.tensor(args.model_settings['grid_size'], device=device),
                'batch': batch_indices  # [total_points] - tells model which points belong to which event
            }
            
            # Forward pass
            predictions = model(data_dict)
            
            # Prepare targets for loss computation
            targets = {
                'point_labels': batch_labels,
                'event_labels': batch_event_y
            }
            
            # Compute loss using model's compute_loss method
            losses = model.compute_loss(predictions, targets, coord)
            total_loss = losses['total_loss']
            
            # Extract individual losses for logging
            point_loss = losses.get('point_loss', torch.tensor(0.0, device=device))
            event_loss = losses.get('event_loss', torch.tensor(0.0, device=device))
            
            # These are total losses, not average per-event losses
            total_test_loss += total_loss.item() * B
            total_test_point_loss += point_loss.item() * B
            total_test_event_loss += event_loss.item() * B
            
            # Statistics for point-wise predictions
            point_features = predictions['point_features']
            point_logits = point_features.feat  # [total_points, num_classes]
            point_probs = torch.softmax(point_logits, dim=1)  # [total_points, num_classes]
            _, predicted = torch.max(point_logits.data, 1)
            
            total_test_num_points += batch_labels.size(0)
            total_test_correct_points += (predicted == batch_labels).sum().item()

            # For visualization, separate predictions and labels by event
            predicted_by_event = []
            true_labels_by_event = []
            coords_by_event = []
            probabilities_by_event = []
            
            for i in range(B):
                event_mask = (batch_indices == i)
                event_pred = predicted[event_mask].cpu().numpy()
                event_true = batch_labels[event_mask].cpu().numpy()
                event_coords = batch_coords[event_mask].cpu().numpy()
                event_probs = point_probs[event_mask].cpu().numpy()
                
                predicted_by_event.append(event_pred)
                true_labels_by_event.append(event_true)
                coords_by_event.append(event_coords)
                probabilities_by_event.append(event_probs)
            
            # Calculate batch-level metrics
            batch_point_accuracy = 100 * (predicted == batch_labels).sum().item() / batch_labels.size(0)
            
            # Store batch metrics for wandb
            batch_losses.append(total_loss.item())
            batch_point_losses.append(point_loss.item())
            batch_event_losses.append(event_loss.item())
            batch_point_accuracies.append(batch_point_accuracy)
            
            # Log batch metrics to wandb if enabled
            if args.wandb:
                wandb.log({
                    'test/batch_loss': total_loss.item(),
                    'test/batch_point_loss': point_loss.item(),
                    'test/batch_event_loss': event_loss.item(),
                    'test/batch_point_accuracy': batch_point_accuracy,
                    'test/batch': batch_idx,
                })
            
            # Save variables to create extra metrics occasionally
            if epoch == -1 or epoch % 5 == 0 or epoch == args.num_epochs - 1:
                all_point_predictions.extend(predicted_by_event)
                all_point_true_labels.extend(true_labels_by_event)
                all_point_probabilities.extend(probabilities_by_event)

                event_probs = torch.softmax(predictions['event_logits'], dim=1)[:, 1].cpu().numpy()
                all_event_probs.extend(event_probs)
                all_event_true_labels.extend(batch_event_y.cpu().numpy())

                for event_i in range(B):
                    if len(subset_sig_point_predictions) < num_spacepoint_plot_events and batch_event_y[event_i] == 1:
                        subset_sig_point_predictions.append(predicted_by_event[event_i])
                        subset_sig_point_true_labels.append(true_labels_by_event[event_i])
                        subset_sig_spacepoints.append(coords_by_event[event_i])
                        subset_sig_event_predictions.append(event_probs[event_i])
                        subset_sig_event_true_labels.append(batch_event_y[event_i])
                        subset_sig_pair_conversion_coords.append(batch_pair_conversion_coords[event_i])
                    elif len(subset_bkg_point_predictions) < num_spacepoint_plot_events and batch_event_y[event_i] == 0:
                        subset_bkg_point_predictions.append(predicted_by_event[event_i])
                        subset_bkg_point_true_labels.append(true_labels_by_event[event_i])
                        subset_bkg_spacepoints.append(coords_by_event[event_i])
                        subset_bkg_event_predictions.append(event_probs[event_i])
                        subset_bkg_event_true_labels.append(batch_event_y[event_i])
                        subset_bkg_pair_conversion_coords.append(batch_pair_conversion_coords[event_i])
            
            test_pbar.set_postfix({
                'Loss': f'{total_loss.item():.4f}',
                'Point_Loss': f'{point_loss.item():.4f}',
                'Event_Loss': f'{event_loss.item():.4f}',
            })
    
    # Create extra test plots
    if epoch == -1 or epoch % 5 == 0 or epoch == args.num_epochs - 1:

        print("Creating extra test plots")
                
        # flatten all_point_true_labels and all_point_predictions, so points from all events are together
        flattened_all_point_true_labels = []
        for all_point_true_labels_event in all_point_true_labels:
            flattened_all_point_true_labels.extend(list(all_point_true_labels_event))
        flattened_all_point_true_labels = np.array(flattened_all_point_true_labels)

        flattened_all_point_predictions = []
        for all_point_predictions_event in all_point_predictions:
            #print(f"{all_point_predictions_event=}")
            flattened_all_point_predictions.extend(list(all_point_predictions_event))
        flattened_all_point_predictions = np.array(flattened_all_point_predictions)
        
        point_cm = confusion_matrix(flattened_all_point_true_labels, flattened_all_point_predictions)
        class_names = ['Gamma1', 'Gamma2', 'Other', 'Cosmic']
        plt.rcParams.update({'font.size': 16})
        point_confusion_fig, point_confusion_ax = plt.subplots(figsize=(8, 6))
        im = point_confusion_ax.imshow(point_cm, interpolation='nearest', cmap='Blues')
        point_confusion_fig.colorbar(im)
        thresh = point_cm.max() / 2.
        for i in range(point_cm.shape[0]):
            for j in range(point_cm.shape[1]):
                point_confusion_ax.text(j, i, format(point_cm[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if point_cm[i, j] > thresh else "black")
        
        point_confusion_ax.set_title('Point Classification Confusion Matrix')
        point_confusion_ax.set_ylabel('True Label')
        point_confusion_ax.set_xlabel('Predicted Label')
        point_confusion_ax.set_xticks(range(len(class_names)))
        point_confusion_ax.set_xticklabels(class_names)
        point_confusion_ax.set_yticks(range(len(class_names)))
        point_confusion_ax.set_yticklabels(class_names)
        point_confusion_fig.tight_layout()
        
        point_confusion_fig.savefig(f'{args.outdir}/plots/point_confusion_fig.jpg', dpi=300)


        # Create event score histogram
        true_signal_probs = np.array(all_event_probs)[np.array(all_event_true_labels) == 1]
        true_background_probs = np.array(all_event_probs)[np.array(all_event_true_labels) == 0]
        bins = np.linspace(0, 1, 51)
        lw = 2

        event_score_fig, event_score_ax = plt.subplots(figsize=(8, 6))
        event_score_ax.hist(true_signal_probs, bins=bins, histtype='step', label='True Signal', linewidth=lw, density=True)
        event_score_ax.hist(true_background_probs, bins=bins, histtype='step', label='True Background', linewidth=lw, density=True)
        event_score_ax.legend()
        event_score_ax.set_xlabel('Output Signal Score')
        event_score_ax.set_ylabel('Relative Number of Events')
        event_score_fig.tight_layout()

        event_score_fig.savefig(f'{args.outdir}/plots/event_score_fig.jpg', dpi=300)

        # create efficiency curve and AUC curve
        all_signal_plus_background_probs = np.concatenate([true_signal_probs, true_background_probs])
        quantiles = np.quantile(all_signal_plus_background_probs, np.linspace(0, 1, 1000))
        total_signal_events = len(true_signal_probs)
        total_background_events = len(true_background_probs)
        signal_efficiencies = []
        background_rejections = []
        eff_times_rej = []
        for cut in quantiles:
            signal_probs_above_cut = true_signal_probs[true_signal_probs > cut]
            background_probs_above_cut = true_background_probs[true_background_probs > cut]
            if total_signal_events == 0:
                signal_efficiency = 0
            else:
                signal_efficiency = len(signal_probs_above_cut) / total_signal_events
            if total_background_events == 0:
                background_rejection = 0
            else:
                background_rejection = 1 - len(background_probs_above_cut) / total_background_events
            signal_efficiencies.append(signal_efficiency)
            background_rejections.append(background_rejection)
            eff_times_rej.append(signal_efficiency * background_rejection)

        efficiency_fig, efficiency_ax = plt.subplots(figsize=(8, 6))
        efficiency_ax.plot(quantiles, signal_efficiencies, label='Signal Efficiency')
        efficiency_ax.plot(quantiles, background_rejections, label='Background Rejection')
        efficiency_ax.plot(quantiles, eff_times_rej, label='Signal Efficiency * Background Rejection')
        efficiency_ax.legend(loc='upper left', fontsize=12)
        efficiency_ax.set_xlabel('Output Signal Score Cut')
        efficiency_ax.set_xlim(0, 1)
        efficiency_ax.set_ylim(0, 1)
        efficiency_fig.tight_layout()

        efficiency_fig.savefig(f'{args.outdir}/plots/efficiency_fig.jpg', dpi=300)

        auc = 0
        for i in range(len(signal_efficiencies) - 1):
            x_width = -(signal_efficiencies[i + 1] - signal_efficiencies[i])
            y_height = (background_rejections[i + 1] + background_rejections[i]) / 2
            auc += x_width * y_height

        auc_fig, auc_ax = plt.subplots(figsize=(8, 6))
        auc_ax.plot(signal_efficiencies, background_rejections)
        auc_ax.set_xlabel('Signal Efficiency')
        auc_ax.set_ylabel('Background Rejection')
        auc_ax.set_title(f"AUC: {auc:.3f}")
        auc_ax.set_xlim(0, 1)
        auc_ax.set_ylim(0, 1)
        auc_fig.tight_layout()

        auc_fig.savefig(f'{args.outdir}/plots/auc_fig.jpg', dpi=300)
        
        
        # Create point category histogram
        true_gamma1_counts_by_event = []
        true_gamma2_counts_by_event = []
        true_other_counts_by_event = []
        true_cosmic_counts_by_event = []
        predicted_gamma1_counts_by_event = []
        predicted_gamma2_counts_by_event = []
        predicted_other_counts_by_event = []
        predicted_cosmic_counts_by_event = []
        for event_i in range(len(all_point_true_labels)):
            true_gamma1_counts_by_event.append(np.sum(all_point_true_labels[event_i] == 0))
            true_gamma2_counts_by_event.append(np.sum(all_point_true_labels[event_i] == 1))
            true_other_counts_by_event.append(np.sum(all_point_true_labels[event_i] == 2))
            true_cosmic_counts_by_event.append(np.sum(all_point_true_labels[event_i] == 3))

            predicted_gamma1_counts_by_event.append(np.sum(all_point_predictions[event_i] == 0))
            predicted_gamma2_counts_by_event.append(np.sum(all_point_predictions[event_i] == 1))
            predicted_other_counts_by_event.append(np.sum(all_point_predictions[event_i] == 2))
            predicted_cosmic_counts_by_event.append(np.sum(all_point_predictions[event_i] == 3))

        bins = np.linspace(0, 500, 26)

        point_category_fig, point_category_ax = plt.subplots(figsize=(8, 6))

        lw = 2

        point_category_ax.hist(predicted_gamma1_counts_by_event, bins=bins, histtype='step', label='Gamma1', color="C0", linewidth=lw, density=True)
        point_category_ax.hist(predicted_gamma2_counts_by_event, bins=bins, histtype='step', label='Gamma2', color="C1", linewidth=lw, density=True)
        point_category_ax.hist(predicted_other_counts_by_event, bins=bins, histtype='step', label='Other', color="C2", linewidth=lw, density=True)
        point_category_ax.hist(predicted_cosmic_counts_by_event, bins=bins, histtype='step', label='Cosmic', color="C3", linewidth=lw, density=True)
        
        point_category_ax.hist(true_gamma1_counts_by_event, bins=bins, histtype='step', color="C0", linestyle="--", linewidth=lw, density=True)
        point_category_ax.hist(true_gamma2_counts_by_event, bins=bins, histtype='step', color="C1", linestyle="--", linewidth=lw, density=True)
        point_category_ax.hist(true_other_counts_by_event, bins=bins, histtype='step', color="C2", linestyle="--", linewidth=lw, density=True)
        point_category_ax.hist(true_cosmic_counts_by_event, bins=bins, histtype='step', color="C3", linestyle="--", linewidth=lw, density=True)

        point_category_ax.plot([], [], c="k", label="Predicted Category", linestyle="-", linewidth=lw)
        point_category_ax.plot([], [], c="k", label="True Category", linestyle="--", linewidth=lw)
        point_category_ax.legend()

        point_category_ax.set_xlabel('Number of Points in Event')
        point_category_ax.set_ylabel('Relative Number of Events')
        point_category_fig.tight_layout()

        point_category_fig.savefig(f'{args.outdir}/plots/point_category_fig.jpg', dpi=300)


        # Create spacepoint-level visualization
        # Create individual plots for each event and save to wandb table
        spacepoint_table_data = []
        plt.rcParams.update({'font.size': 10})
        
        # Define event types and their corresponding data
        event_types = [
            ("Signal", subset_sig_point_predictions, subset_sig_point_true_labels, 
             subset_sig_spacepoints, subset_sig_event_predictions, subset_sig_event_true_labels, 
             subset_sig_pair_conversion_coords),
            ("Background", subset_bkg_point_predictions, subset_bkg_point_true_labels, 
             subset_bkg_spacepoints, subset_bkg_event_predictions, subset_bkg_event_true_labels, 
             subset_bkg_pair_conversion_coords)
        ]
        
        # Process both signal and background events
        for event_type, point_predictions, point_true_labels, spacepoints, event_predictions, event_true_labels, pair_conversion_coords in event_types:
            for event_i in range(min(len(point_predictions), num_spacepoint_plot_events)):
                # Create single figure with two subplots side by side
                fig, (pred_ax, true_ax) = plt.subplots(1, 2, figsize=(12, 6))
                
                # Create predicted plot (left subplot)
                pred_colors = []
                for i in range(len(point_predictions[event_i])):
                    pred_label = point_predictions[event_i][i]
                    if pred_label == 0:
                        pred_colors.append('green')
                    elif pred_label == 1:
                        pred_colors.append('lawngreen')
                    elif pred_label == 2:
                        pred_colors.append('brown')
                    else:
                        pred_colors.append('blue')
                
                s = 0.2
                
                pred_ax.scatter(spacepoints[event_i][:, 2], 
                               spacepoints[event_i][:, 0], 
                               s=s, c=pred_colors)
                
                # Set plot limits and formatting
                non_true_cosmic_indices = [i for i in range(len(spacepoints[event_i])) 
                                         if point_true_labels[event_i][i] != 3]
                non_true_cosmic_spacepoints = np.array([spacepoints[event_i][i] 
                                                      for i in non_true_cosmic_indices])
                
                if len(non_true_cosmic_spacepoints) > 0:
                    all_coords = np.vstack([non_true_cosmic_spacepoints, pair_conversion_coords[event_i]])
                    min_x = min(all_coords[:, 2])
                    max_x = max(all_coords[:, 2])
                    min_y = min(all_coords[:, 0])
                    max_y = max(all_coords[:, 0])
                else:
                    min_x = min(spacepoints[event_i][:, 2])
                    max_x = max(spacepoints[event_i][:, 2])
                    min_y = min(spacepoints[event_i][:, 0])
                    max_y = max(spacepoints[event_i][:, 0])

                if len(pair_conversion_coords[event_i]) > 0:
                    for coord in pair_conversion_coords[event_i]:
                        min_x = min(min_x, coord[2])
                        max_x = max(max_x, coord[2])
                        min_y = min(min_y, coord[0])
                        max_y = max(max_y, coord[0])
                
                x_width = max_x - min_x
                y_width = max_y - min_y
                extra_scale_factor = 0.2
                pred_ax.set_xlim(min_x - extra_scale_factor * x_width, max_x + extra_scale_factor * x_width)
                pred_ax.set_ylim(min_y - extra_scale_factor * y_width, max_y + extra_scale_factor * y_width)
                
                pred_ax.set_title(f'{event_type} Event {event_i} - Predicted')
                pred_ax.set_xlabel('Z')
                pred_ax.set_ylabel('X')
                pred_ax.set_xticks([])
                pred_ax.set_yticks([])

                pred_ax.text(0.95, 0.05, f"Pred Signal Prob: {event_predictions[event_i]:.2f}", 
                            transform=pred_ax.transAxes, ha='right', va='bottom', fontsize=10)
                
                # Create true plot (right subplot)
                true_colors = []
                for i in range(len(point_true_labels[event_i])):
                    true_label = point_true_labels[event_i][i]
                    if true_label == 0:
                        true_colors.append('green')
                    elif true_label == 1:
                        true_colors.append('lawngreen')
                    elif true_label == 2:
                        true_colors.append('brown')
                    else:
                        true_colors.append('blue')
                
                true_ax.scatter(spacepoints[event_i][:, 2], 
                               spacepoints[event_i][:, 0], 
                               s=s, c=true_colors)
                
                # Plot pair conversion points as red stars
                pair_x = [coord[2] for coord in pair_conversion_coords[event_i]]
                pair_y = [coord[0] for coord in pair_conversion_coords[event_i]]
                true_ax.scatter(pair_x, pair_y, 
                               s=50, c='red', marker='*', edgecolors='black', linewidth=0.5)
                
                true_ax.set_xlim(min_x - extra_scale_factor * x_width, max_x + extra_scale_factor * x_width)
                true_ax.set_ylim(min_y - extra_scale_factor * y_width, max_y + extra_scale_factor * y_width)
                
                true_ax.set_title(f'{event_type} Event {event_i} - True')
                true_ax.set_xlabel('Z')
                true_ax.set_ylabel('X')
                true_ax.set_xticks([])
                true_ax.set_yticks([])
                
                true_ax.text(0.95, 0.05, f"True Signal: {event_true_labels[event_i]}", 
                            transform=true_ax.transAxes, ha='right', va='bottom', fontsize=10)
                
                if len(non_true_cosmic_spacepoints) == 0:
                    true_ax.text(0.5, 0.5, "No reco spacepoints\ncorresponding to \ntrue neutrino EDeps", 
                                transform=true_ax.transAxes, ha='center', va='center', fontsize=8)

                fig.tight_layout()
                
                # Add to table data
                spacepoint_table_data.append({
                    "Event Type": event_type,
                    "Event Index": event_i,
                    "Event Plot": wandb.Image(fig),
                    "Predicted Signal Probability": event_predictions[event_i],
                    "True Signal Label": event_true_labels[event_i]
                })
                
                fig.savefig(f'{args.outdir}/plots/spacepoint_plots/{event_type.lower()}_event_{event_i}.png', dpi=300)
                plt.close(fig)
        
        # Create wandb table
        if args.wandb and spacepoint_table_data:
            spacepoint_table = wandb.Table(columns=["Event Type", "Event Index", "Event Plot", "Predicted Signal Probability", "True Signal Label"], 
                                           data=[[row["Event Type"], row["Event Index"], row["Event Plot"], 
                                                 row["Predicted Signal Probability"], 
                                                 row["True Signal Label"]] for row in spacepoint_table_data])

        # Create point probability distribution plot
        # Flatten all probabilities and labels
        flattened_probs = []
        flattened_labels = []
        for event_probs, event_labels in zip(all_point_probabilities, all_point_true_labels):
            for point_probs, point_label in zip(event_probs, event_labels):
                flattened_probs.append(point_probs)
                flattened_labels.append(point_label)
        
        flattened_probs = np.array(flattened_probs)  # [total_points, num_classes]
        flattened_labels = np.array(flattened_labels)  # [total_points]
        
        # Create subplots for each class
        class_names = ['Gamma1', 'Gamma2', 'Other', 'Cosmic']
        num_classes = len(class_names)
        
        point_prob_fig, point_prob_axes = plt.subplots(2, 2, figsize=(12, 10))
        point_prob_axes = point_prob_axes.flatten()
        
        bins = np.linspace(0, 1, 51)
        
        for class_idx in range(num_classes):
            ax = point_prob_axes[class_idx]
            
            # Get probabilities for this class
            class_probs = flattened_probs[:, class_idx]
            
            # Separate by true label
            for true_class_idx in range(num_classes):
                mask = (flattened_labels == true_class_idx)
                if np.sum(mask) > 0:
                    true_class_probs = class_probs[mask]
                    ax.hist(true_class_probs, bins=bins, histtype='step', 
                           label=f'True {class_names[true_class_idx]}', 
                           linewidth=2, density=True, alpha=0.8)
            
            ax.set_xlabel(f'Predicted Probability for {class_names[class_idx]}')
            ax.set_ylabel('Density')
            ax.set_title(f'Distribution of {class_names[class_idx]} Predictions')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, 1)
        
        point_prob_fig.tight_layout()
        point_prob_fig.savefig(f'{args.outdir}/plots/point_probability_distribution.jpg', dpi=300)

        plt.close(point_confusion_fig)
        plt.close(event_score_fig)
        plt.close(point_category_fig)
        plt.close(point_prob_fig)
        plt.close(efficiency_fig)
        plt.close(auc_fig)

        if args.wandb:
            wandb.log({
                'test/point_confusion_matrix': wandb.Image(point_confusion_fig),
                'test/event_score_histogram': wandb.Image(event_score_fig),
                'test/point_category_histogram': wandb.Image(point_category_fig),
                'test/point_probability_distribution': wandb.Image(point_prob_fig),
                'test/spacepoint_visualization_table': spacepoint_table if spacepoint_table_data else None,
                'test/efficiency_curve': wandb.Image(efficiency_fig),
                'test/auc_curve': wandb.Image(auc_fig),
            })

    loss = total_test_loss / num_test_events
    point_loss = total_test_point_loss / num_test_events
    event_loss = total_test_event_loss / num_test_events
    point_accuracy = total_test_correct_points / total_test_num_points
    
    return loss, point_loss, event_loss, point_accuracy


if __name__ == "__main__":

    start_time = datetime.now()

    parser = argparse.ArgumentParser(description="Train spacepoint SSV neural network.")
    parser.add_argument('-f', '--input_file', type=str, required=False, help='Path to root file to pre-process.', default='intermediate_files/downsampled_spacepoints.pkl')
    parser.add_argument('-o', '--outdir', type=str, required=False, help='Path to directory to save logs and checkpoints.', default=f"training_files/{datetime.now().strftime("%Y_%m_%d-%H:%M:%S")}")
    parser.add_argument('-n', '--num_events', type=int, required=False, help='Number of training events to use.')
    parser.add_argument('-tf', '--train_fraction', type=float, required=False, help='Fraction of training events to use.', default=0.9)
    parser.add_argument('-b', '--batch_size', type=int, required=False, help='Batch size for training.', default=5_000)
    parser.add_argument('-e', '--num_epochs', type=int, required=False, help='Number of epochs to train for.', default=50)
    parser.add_argument('-w', '--num_workers', type=int, required=False, help='Number of worker processes for data loading.', default=0)
    parser.add_argument('-ns', '--no_save', action='store_true', required=False, help='Do not save checkpoints.')
    parser.add_argument('--random_seed', type=int, required=False, help='Random seed for training.', default=42)

    parser.add_argument('-wb', '--wandb', action='store_true', required=False, help='Use wandb to track training.')
    parser.add_argument('--wandb_project', type=str, required=False, help='Wandb project name.', default='spacepoint-ssv')
    parser.add_argument('--wandb_entity', type=str, required=False, help='Wandb entity/username.', default=None)
    parser.add_argument('--name', type=str, required=False, help='Wandb run name.', default=None)

    parser.add_argument('--event_loss_weight', type=float, required=False, help='Weight of event loss.', default=1.0)

    parser.add_argument('--model_settings', type=str, required=False, help='Model settings.', default={'type': 'MultiTaskPointTransformerV3', 'grid_size': 0.01})

    parser.add_argument('-lr', '--learning_rate', type=float, required=False, help='Learning rate for training.', default=1e-3)
    parser.add_argument('--weight_decay', type=float, required=False, help='Weight decay for training.', default=1e-2)
    parser.add_argument('--scheduler_settings', type=str, required=False, help='Scheduler type settings.', default={'type': 'CosineAnnealingLR'})

    parser.add_argument('--spacepoints_type', type=str, required=False, help='Type of spacepoints, either all_points, only_photons, only_neutrinos, or only_two_photons.', default='all_points')
    parser.add_argument('--with_charge', action='store_true', required=False, help='Whether to use charge information.')

    parser.add_argument('--gamma_separation_loss_weight', type=float, required=False, help='Weight of gamma separation loss.', default=0)
    parser.add_argument('--gamma_KL_loss_weight', type=float, required=False, help='Weight of gamma KL loss.', default=0)
    parser.add_argument('--entropy_loss_weight', type=float, required=False, help='Weight of entropy loss.', default=0)
    parser.add_argument('--variance_loss_weight', type=float, required=False, help='Weight of variance loss.', default=0)
    parser.add_argument('--near_05_loss_weight', type=float, required=False, help='Weight of near 0.5 loss.', default=0)
    parser.add_argument('--gamma_one_side_loss_weight', type=float, required=False, help='Weight of gamma one side loss.', default=0)

    args = parser.parse_args()

    if args.name is not None:
        args.wandb = True
        args.outdir = f"training_files/{args.name}"
    else:
        args.name = f"spacepoint-ssv-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    if not args.no_save:
        os.makedirs(args.outdir, exist_ok=True)
        os.makedirs(f'{args.outdir}/plots', exist_ok=True)
        os.makedirs(f'{args.outdir}/plots/spacepoint_plots', exist_ok=True)

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

    if args.model_settings['type'] == 'MultiTaskPointTransformerV3' and device != torch.device("cuda:0"):
        raise ValueError("MultiTaskPointTransformerV3 is only supported on CUDA GPUs")
        
    # Create comprehensive config with all parameter log information
    config = {
        # Basic training parameters
        'input_file': args.input_file,
        'outdir': args.outdir,
        'num_events': args.num_events,
        'train_fraction': args.train_fraction,
        'batch_size': args.batch_size,
        'num_epochs': args.num_epochs,
        'num_workers': args.num_workers,
        'random_seed': args.random_seed,
        
        'model_settings': args.model_settings,
        'scheduler_settings': args.scheduler_settings,
        'event_loss_weight': args.event_loss_weight,

        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,

        'spacepoints_type': args.spacepoints_type,
        'with_charge': args.with_charge,

        'gamma_separation_loss_weight': args.gamma_separation_loss_weight,
        'gamma_KL_loss_weight': args.gamma_KL_loss_weight,
        'entropy_loss_weight': args.entropy_loss_weight,
        'variance_loss_weight': args.variance_loss_weight,
        'near_05_loss_weight': args.near_05_loss_weight,
        'gamma_one_side_loss_weight': args.gamma_one_side_loss_weight,
        # System information
        'pytorch_version': torch.__version__,
        'device': str(device),
        'cuda_available': torch.cuda.is_available(),
        'training_start_time': start_time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    
    if torch.cuda.is_available():
        config.update({
            'cuda_device_name': torch.cuda.get_device_name(0),
            'cuda_version': torch.version.cuda,
        })
        
    if args.wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.name,
        )
        print(f"Initialized wandb run: {wandb.run.name}")
        
        config.update({
            'wandb_project': args.wandb_project,
            'wandb_entity': args.wandb_entity,
            'name': args.name,
            'wandb_run_url': wandb.run.url,
        })

    rng = torch.Generator()
    rng.manual_seed(args.random_seed)

    # Create dataloaders
    print("Creating dataloaders")
    train_dataloader, test_dataloader, num_train_events, num_test_events = create_dataloaders(
        pickle_file=args.input_file,
        batch_size=args.batch_size,
        num_events=args.num_events,
        train_fraction=args.train_fraction,
        num_workers=args.num_workers,
        random_seed=args.random_seed,
        out_dir=args.outdir,
        no_save=args.no_save,
        spacepoints_type=args.spacepoints_type,
        with_charge=args.with_charge,
        rng=rng,
    )
    
    print(f"Training batches: {len(train_dataloader)}")
    print(f"Testing batches: {len(test_dataloader)}")
    
    # Get a sample batch to determine input dimensions
    sample_batch_coords, sample_batch_labels, sample_batch_event_y, sample_batch_indices, sample_batch_pair_conversion_coords = next(iter(train_dataloader))
    
    # Determine number of classes from the data
    num_point_classes = 4
    num_event_classes = 2

    if args.with_charge:
        in_channels = 4 # xyzq
    else:
        in_channels = 3 # xyz
    
    model = MultiTaskPointTransformerV3(
        num_classes=4,          # true gamma 1, true gamma 2, other particles, cosmic
        num_event_classes=2,    # signal 1g, background 2g
        event_loss_weight=1.0,
        in_channels=in_channels,
        gamma_separation_loss_weight=args.gamma_separation_loss_weight,
        gamma_KL_loss_weight=args.gamma_KL_loss_weight,
        entropy_loss_weight=args.entropy_loss_weight,
        variance_loss_weight=args.variance_loss_weight,
        near_05_loss_weight=args.near_05_loss_weight,
        gamma_one_side_loss_weight=args.gamma_one_side_loss_weight,
    )
    model = model.to(device)
    
    print(f"created model with {sum(p.numel() for p in model.parameters())} parameters")
    

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Update wandb config with model parameters and data information
    config.update({
        'num_train_events': num_train_events,
        'num_test_events': num_test_events,            
        'num_training_batches': len(train_dataloader),
        'num_testing_batches': len(test_dataloader),
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'random_seed': args.random_seed,
    })
    
    # tracks model parameters and gradients
    if args.wandb:
        wandb.watch(model, log="all", log_freq=100)
        
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    if args.scheduler_settings['type'] == 'CosineAnnealingLR':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs, eta_min=args.learning_rate * 0.01)
    else:
        raise ValueError(f"Scheduler type {args.scheduler_settings['type']} not supported")

    # save config information to a local file as well as wandb
    with open(f"{args.outdir}/config.json", "w") as f:
        json.dump(config, f, indent=4)
    print(f"Configuration saved to: {args.outdir}/config.json")
    if args.wandb:
        wandb.config.update(config)
    
    print(f"Starting training for {args.num_epochs} epochs...")

    best_test_loss = float('inf')
    for epoch in range(-1, args.num_epochs):

        if epoch == -1: # testing the random network before any training
            test_loss, test_point_loss, test_event_loss, test_point_accuracy = test_step(model, test_dataloader, device, epoch, args)
            # train and test should be the same for the inital random network before any training happens
            train_loss, train_point_loss, train_event_loss, train_point_accuracy = test_loss, test_point_loss, test_event_loss, test_point_accuracy
        else:
            train_loss, train_point_loss, train_event_loss, train_point_accuracy = train_step(model, train_dataloader, optimizer, device, epoch, args)
            test_loss, test_point_loss, test_event_loss, test_point_accuracy = test_step(model, test_dataloader, device, epoch, args)

        if test_loss < best_test_loss: # save the best model
            best_test_loss = test_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'test_loss': test_loss,
            }, f'{args.outdir}/best_model.pth')

        if (epoch + 1) % 10 == 0: # save a checkpoint of the model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'test_loss': test_loss,
            }, f'{args.outdir}/checkpoint_epoch_{epoch+1}.pth')

        if args.wandb:
            wandb.log({
                'epoch': epoch,
                'learning_rate': optimizer.param_groups[0]['lr'],
                'train/loss': train_loss,
                'train/point_loss': train_point_loss,
                'train/event_loss': train_event_loss,
                'train/point_accuracy': train_point_accuracy,
                'test/loss': test_loss,
                'test/point_loss': test_point_loss,
                'test/event_loss': test_event_loss,
                'test/point_accuracy': test_point_accuracy,
            })

        if epoch >= 0: # don't step the scheduler for the first loop, since that's just the random untrained network
            scheduler.step()
            
    print("Training completed!")
    if not args.no_save:
        print(f"Best model saved to: {args.outdir}/best_model.pth")
    
    # Finish wandb run
    if args.wandb:
        wandb.finish()
        print("Wandb run completed")

