import torch
import torch.nn as nn
from models.PointTransformerV3.model import PointTransformerV3, Point

class EventCategorizationHead(nn.Module):
    def __init__(self, in_channels, num_event_classes, hidden_channels=[512, 256], dropout=0.5, norm_layer=nn.LayerNorm, act_layer=nn.GELU):
        super().__init__()
        self.num_event_classes = num_event_classes

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
        batch_features = []
        offsets = point_features.offset

        if len(offsets) > 0 and offsets[0] != 0:
            offsets = torch.cat([torch.tensor([0], device=offsets.device, dtype=offsets.dtype), offsets])

        for i in range(len(offsets) - 1):
            start_idx = offsets[i]
            end_idx = offsets[i + 1]
            batch_feat = point_features.feat[start_idx:end_idx]
            event_feat = torch.mean(batch_feat, dim=0)
            batch_features.append(event_feat)

        event_features = torch.stack(batch_features, dim=0)
        return self.classifier(event_features)


class MultiTaskPointTransformerV3(PointTransformerV3):
    def __init__(self, num_event_classes=10, event_hidden_channels=[512, 256], event_dropout=0.5, enable_event_classification=True, event_loss_weight=1.0, **kwargs):
        self.num_point_classes = kwargs.pop('num_classes', 4)
        super().__init__(**kwargs)

        self.enable_event_classification = enable_event_classification
        self.event_loss_weight = event_loss_weight
        self.num_event_classes = num_event_classes

        final_dec_channels = kwargs.get('dec_channels', (64, 64, 128, 256))[0]
        self.point_classifier = nn.Linear(final_dec_channels, self.num_point_classes)

        if self.enable_event_classification:
            final_enc_channels = kwargs.get('enc_channels', (32, 64, 128, 256, 512))[-1]
            self.event_head = EventCategorizationHead(
                in_channels=final_enc_channels,
                num_event_classes=num_event_classes,
                hidden_channels=event_hidden_channels,
                dropout=event_dropout,
                norm_layer=nn.LayerNorm,
                act_layer=nn.GELU
            )

    def forward(self, data_dict):
        point_features = super().forward(data_dict)

        point_logits = self.point_classifier(point_features.feat)
        point_features.feat = point_logits

        results = {'point_features': point_features}

        if self.enable_event_classification:
            point = Point(data_dict)
            point.serialization(order=self.order, shuffle_orders=self.shuffle_orders)
            point.sparsify()
            point = self.embedding(point)
            encoder_output = self.enc(point)

            event_logits = self.event_head(encoder_output)
            results['event_logits'] = event_logits

        return results

    def compute_loss(self, predictions, targets):

        losses = {}

        point_logits = predictions['point_features'].feat
        point_labels = targets['point_labels']
        point_loss = nn.CrossEntropyLoss()(point_logits, point_labels)

        swapped_point_labels = point_labels.clone()
        mask_gamma1 = (point_labels == 0)
        mask_gamma2 = (point_labels == 1)
        swapped_point_labels[mask_gamma1] = 1
        swapped_point_labels[mask_gamma2] = 0
        swapped_point_loss = nn.CrossEntropyLoss()(point_logits, swapped_point_labels)

        # don't care if gamma1 and gamma2 points are swapped, we just want to separate them into two categories
        point_loss = torch.min(point_loss, swapped_point_loss)
        losses['point_loss'] = point_loss

        if self.enable_event_classification and 'event_labels' in targets:
            event_loss = nn.CrossEntropyLoss()(predictions['event_logits'], targets['event_labels'])
            losses['event_loss'] = event_loss

        total_loss = losses.get('point_loss', 0)
        if 'event_loss' in losses:
            total_loss = total_loss + self.event_loss_weight * losses['event_loss']

        losses['total_loss'] = total_loss
        return losses
