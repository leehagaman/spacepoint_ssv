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
    def __init__(self, num_event_classes=10, event_hidden_channels=[512, 256], event_dropout=0.5, 
                 event_loss_weight=1, gamma_separation_loss_weight=0, gamma_KL_loss_weight=0, 
                 entropy_loss_weight=0.1, variance_loss_weight=0.1, near_05_loss_weight=0.1, gamma_one_side_loss_weight=0, **kwargs):
        self.num_point_classes = kwargs.pop('num_classes', 4)
        super().__init__(**kwargs)

        self.event_loss_weight = event_loss_weight
        self.gamma_separation_loss_weight = gamma_separation_loss_weight
        self.gamma_KL_loss_weight = gamma_KL_loss_weight
        self.entropy_loss_weight = entropy_loss_weight
        self.variance_loss_weight = variance_loss_weight
        self.near_05_loss_weight = near_05_loss_weight
        self.gamma_one_side_loss_weight = gamma_one_side_loss_weight
        self.num_event_classes = num_event_classes

        final_dec_channels = kwargs.get('dec_channels', (64, 64, 128, 256))[0]
        self.point_classifier = nn.Linear(final_dec_channels, self.num_point_classes)

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

        point = Point(data_dict)
        point.serialization(order=self.order, shuffle_orders=self.shuffle_orders)
        point.sparsify()
        point = self.embedding(point)
        encoder_output = self.enc(point)

        event_logits = self.event_head(encoder_output)
        results['event_logits'] = event_logits

        return results

    def compute_loss(self, predictions, targets, coords):

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

        event_loss = nn.CrossEntropyLoss()(predictions['event_logits'], targets['event_labels'])
        losses['event_loss'] = event_loss

        point_probs = torch.softmax(point_logits, dim=1)
        gamma1_probs = point_probs[:, 0]
        gamma2_probs = point_probs[:, 1]

        # this tries to punish the case where all the probability is concentrated in one class
        if self.gamma_separation_loss_weight > 0:
            gamma_separation_loss = -torch.mean(gamma1_probs * gamma2_probs)
            losses['gamma_separation_loss'] = self.gamma_separation_loss_weight * gamma_separation_loss

        # this tries to punish the case where gamma1 and gamma2 probability distributions are very similar (always guessing 0.5 probability for both gammas)
        # although similar probability distributions probably aren't actually bad, we just don't want them to be similar for the same points...
        if self.gamma_KL_loss_weight > 0:

            gamma1_distribution = gamma1_probs / (torch.sum(gamma1_probs) + 1e-8)
            gamma2_distribution = gamma2_probs / (torch.sum(gamma2_probs) + 1e-8)

            KL_1 = torch.sum(gamma1_distribution * torch.log(gamma1_distribution / (gamma2_distribution + 1e-8)))
            KL_2 = torch.sum(gamma2_distribution * torch.log(gamma2_distribution / (gamma1_distribution + 1e-8)))
            KL_div = (KL_1 + KL_2) / 2

            losses['gamma_KL_loss'] = - self.gamma_KL_loss_weight * KL_div

        # entropy loss to punish uncertain guesses (always guessing 0.5 probability for two out of four classes)
        if self.entropy_loss_weight > 0:
            entropy = torch.mean(torch.sum(-point_probs * torch.log(point_probs + 1e-8), dim=1))
            losses['entropy_loss'] = - self.entropy_loss_weight * entropy

        # variance loss to punish the case where the probability is the same for every spacepoint
        if self.variance_loss_weight > 0:
            variance = torch.mean(torch.var(point_probs, dim=1))
            losses['variance_loss'] = - self.variance_loss_weight * variance

        # loss to punish probabilities near 0.5
        if self.near_05_loss_weight > 0:
            near_05_loss = torch.mean(torch.abs(point_probs - 0.5))
            losses['near_05_loss'] = - self.near_05_loss_weight * near_05_loss

        # loss to punish when all the gamma probabilities are on one side of 0.5
        if self.gamma_one_side_loss_weight > 0:
            # Use sigmoid to create smooth transitions around 0.5
            # sigmoid((x - 0.5) / temperature) gives values close to 0 for x < 0.5 and close to 1 for x > 0.5
            temperature = 0.1  # Controls how sharp the transition is (smaller = sharper)
            
            gamma1_below = torch.mean(torch.sigmoid((0.5 - gamma1_probs) / temperature))
            gamma1_above = torch.mean(torch.sigmoid((gamma1_probs - 0.5) / temperature))
            gamma2_below = torch.mean(torch.sigmoid((0.5 - gamma2_probs) / temperature))
            gamma2_above = torch.mean(torch.sigmoid((gamma2_probs - 0.5) / temperature))
            
            gamma1_imbalance = torch.abs(gamma1_below - gamma1_above)
            gamma2_imbalance = torch.abs(gamma2_below - gamma2_above)

            losses['gamma_one_side_loss'] = self.gamma_one_side_loss_weight * (gamma1_imbalance + gamma2_imbalance)

        total_loss = losses.get('point_loss', 0)
        if 'event_loss' in losses:
            total_loss = total_loss + self.event_loss_weight * losses['event_loss']
            if self.gamma_separation_loss_weight > 0:
                total_loss = total_loss + losses['gamma_separation_loss']
            if self.gamma_KL_loss_weight > 0:
                total_loss = total_loss + losses['gamma_KL_loss']
            if self.entropy_loss_weight > 0:
                total_loss = total_loss + losses['entropy_loss']
            if self.variance_loss_weight > 0:
                total_loss = total_loss + losses['variance_loss']
            if self.near_05_loss_weight > 0:
                total_loss = total_loss + losses['near_05_loss']
            if self.gamma_one_side_loss_weight > 0:
                total_loss = total_loss + losses['gamma_one_side_loss']

        losses['total_loss'] = total_loss
        return losses
