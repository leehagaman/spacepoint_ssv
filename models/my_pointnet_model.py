# modified from https://github.com/yanx27/Pointnet_Pointnet2_pytorch/blob/master/models/pointnet2_part_seg_ssg.py

import torch.nn as nn
import torch
import torch.nn.functional as F
from models.pointnet2_utils import PointNetSetAbstraction,PointNetFeaturePropagation


class get_model(nn.Module):
    def __init__(self, num_classes=4, global_feature_dim=256):
        super(get_model, self).__init__()
        self.global_feature_dim = global_feature_dim
        
        # Set Abstraction layers
        self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=32, in_channel=6, mlp=[64, 64, 128], group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=128 + 3, mlp=[128, 128, 256], group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256 + 3, mlp=[256, 512, 1024], group_all=True)
        
        # Feature Propagation layers
        self.fp3 = PointNetFeaturePropagation(in_channel=1280, mlp=[256, 256])
        self.fp2 = PointNetFeaturePropagation(in_channel=384, mlp=[256, 128])
        self.fp1 = PointNetFeaturePropagation(in_channel=128+16+6, mlp=[128, 128, 128])
        
        # Point-level prediction layers
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, num_classes, 1)
        
        # Global classification layers (binary classification)
        self.global_fc1 = nn.Linear(1024, 512)
        self.global_bn1 = nn.BatchNorm1d(512)
        self.global_drop1 = nn.Dropout(0.5)
        self.global_fc2 = nn.Linear(512, 256)
        self.global_bn2 = nn.BatchNorm1d(256)
        self.global_drop2 = nn.Dropout(0.5)
        self.global_fc3 = nn.Linear(256, 1)  # Binary classification output

    def forward(self, xyz, cls_label):
        # Set Abstraction layers
        B,C,N = xyz.shape
        l0_points = xyz
        l0_xyz = xyz
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        
        # Global classification prediction
        global_feat = l3_points.view(B, -1)  # [B, 1024]
        global_feat = F.relu(self.global_bn1(self.global_fc1(global_feat)))
        global_feat = self.global_drop1(global_feat)
        global_feat = F.relu(self.global_bn2(self.global_fc2(global_feat)))
        global_feat = self.global_drop2(global_feat)
        global_output = self.global_fc3(global_feat)  # [B, 1] - binary classification logits
        
        # Feature Propagation layers
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        cls_label_one_hot = cls_label.view(B,16,1).repeat(1,1,N)
        l0_points = self.fp1(l0_xyz, l1_xyz, torch.cat([cls_label_one_hot,l0_xyz,l0_points],1), l1_points)
        
        # Point-level prediction
        feat =  F.relu(self.bn1(self.conv1(l0_points)))
        x = self.drop1(feat)
        x = self.conv2(x)
        x = F.log_softmax(x, dim=1)
        x = x.permute(0, 2, 1)
        
        return x, global_output


class get_loss(nn.Module):
    def __init__(self, point_loss_weight=1.0, global_loss_weight=1.0):
        super(get_loss, self).__init__()
        self.point_loss_weight = point_loss_weight
        self.global_loss_weight = global_loss_weight

    def forward(self, pred, target, global_pred, global_target):
        point_loss = F.nll_loss(pred, target)
        
        # Global binary classification loss
        # global_pred has shape (B, 1), global_target has shape (B,)
        # We need to squeeze global_pred and convert global_target to float for BCE
        global_pred_squeezed = global_pred.squeeze(1)  # Shape: (B,)
        global_target_float = global_target.float()  # Shape: (B,)
        global_loss = F.binary_cross_entropy_with_logits(global_pred_squeezed, global_target_float)
        
        total_loss = self.point_loss_weight * point_loss + self.global_loss_weight * global_loss
        
        return total_loss, point_loss, global_loss
    