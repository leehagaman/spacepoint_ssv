
import torch
import torch.nn as nn
from typing import List, Dict, Any
from models.PointTransformerV3_model import PointTransformerV3, Point

class EventCategorizationHead(nn.Module):
    """
    Event categorization head that processes global features to classify entire point clouds.
    """
    def __init__(
        self,
        in_channels: int,
        num_event_classes: int,
        hidden_channels: List[int] = [512, 256],
        dropout: float = 0.5,
        norm_layer=nn.LayerNorm,
        act_layer=nn.GELU
    ):
        super().__init__()
        self.num_event_classes = num_event_classes
        
        # Classification head
        layers = []
        prev_channels = in_channels
        
        for hidden_ch in hidden_channels:
            layers.extend([
                nn.Linear(prev_channels, hidden_ch),
                norm_layer(hidden_ch),
                act_layer(),
                nn.Dropout(dropout)
            ])
            prev_channels = hidden_ch
            
        layers.append(nn.Linear(prev_channels, num_event_classes))
        self.classifier = nn.Sequential(*layers)
        
    def forward(self, point_features):
        """
        Args:
            point_features: Point object with .feat and .offset attributes
        Returns:
            event_logits: [B, num_event_classes] where B is batch size
        """
        # Extract global features per batch item using offset information
        batch_features = []
        offsets = point_features.offset
        
        for i in range(len(offsets) - 1):
            start_idx = offsets[i]
            end_idx = offsets[i + 1]
            batch_feat = point_features.feat[start_idx:end_idx]  # [N_i, C]
            # Global average pooling over points in this batch item
            global_feat = torch.mean(batch_feat, dim=0)  # [C]
            batch_features.append(global_feat)
        
        # Stack to get [B, C]
        global_features = torch.stack(batch_features, dim=0)
        return self.classifier(global_features)


class MultiTaskPointTransformerV3(PointTransformerV3):
    """
    Extended PointTransformerV3 that adds event categorization capability.
    Inherits all functionality from the original model with minimal code duplication.
    """
    def __init__(
        self,
        # Event categorization parameters
        num_event_classes: int = 4,
        event_hidden_channels: List[int] = [512, 256],
        event_dropout: float = 0.5,
        enable_event_classification: bool = True,
        event_loss_weight: float = 1.0,
        # All original PTv3 parameters
        **kwargs
    ):
        # Initialize the base PointTransformerV3 with all original parameters
        super().__init__(**kwargs)
        
        # Store event classification parameters
        self.enable_event_classification = enable_event_classification
        self.event_loss_weight = event_loss_weight
        self.num_event_classes = num_event_classes
        
        # Add event categorization head if enabled
        if self.enable_event_classification:
            # Use the encoder's final channel count for event classification
            final_enc_channels = kwargs.get('enc_channels', (32, 64, 128, 256, 512))[-1]
            
            self.event_head = EventCategorizationHead(
                in_channels=final_enc_channels,
                num_event_classes=num_event_classes,
                hidden_channels=event_hidden_channels,
                dropout=event_dropout,
                norm_layer=nn.LayerNorm,
                act_layer=nn.GELU
            )
    
    def forward(self, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Forward pass that returns both point-wise and event-level predictions.
        
        Args:
            data_dict: Input data dictionary for point cloud
            
        Returns:
            Dictionary containing:
            - 'point_features': Point-wise features/predictions from original model
            - 'event_logits': Event classification logits (if enabled)
        """
        # Get point-wise features from the base model
        point_features = super().forward(data_dict)
        
        results = {'point_features': point_features}
        
        # Add event classification if enabled
        if self.enable_event_classification:
            # Use encoder output for event classification (before decoder)
            # We need to run the encoder part separately to get intermediate features
            point = Point(data_dict)
            point.serialization(order=self.order, shuffle_orders=self.shuffle_orders)
            point.sparsify()
            
            point = self.embedding(point)
            encoded_point = self.enc(point)
            
            # Get event classification from encoded features
            event_logits = self.event_head(encoded_point)
            results['event_logits'] = event_logits
        
        return results
    
    def compute_loss(self, predictions: Dict[str, Any], targets: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Compute multi-task loss combining point-wise and event-level losses.
        
        Args:
            predictions: Model predictions containing 'point_features' and optionally 'event_logits'
            targets: Ground truth containing 'point_labels' and optionally 'event_labels'
            
        Returns:
            Dictionary of losses
        """
        losses = {}
        
        # Point-wise loss (implement based on your specific task)
        if 'point_labels' in targets:
            # Example: cross-entropy loss for point classification
            point_loss = nn.CrossEntropyLoss()(
                predictions['point_features'].feat,  # Assuming .feat contains logits
                targets['point_labels']
            )
            losses['point_loss'] = point_loss
        
        # Event-level loss
        if self.enable_event_classification and 'event_labels' in targets:
            event_loss = nn.CrossEntropyLoss()(
                predictions['event_logits'],
                targets['event_labels']
            )
            losses['event_loss'] = event_loss
        
        # Combined loss
        total_loss = losses.get('point_loss', 0)
        if 'event_loss' in losses:
            total_loss = total_loss + self.event_loss_weight * losses['event_loss']
        
        losses['total_loss'] = total_loss
        return losses
