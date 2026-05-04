import torch.nn as nn
import torch.nn.functional as F
from .pointnetpp_utils import PointNetSetAbstraction,PointNetFeaturePropagation
import torch

class PointPP(nn.Module):
    def __init__(self):
        super(PointPP, self).__init__()
        self.sa1 = PointNetSetAbstraction(800, 0.08, 32, 6, [32, 32, 64], False)
        self.sa2 = PointNetSetAbstraction(400, 0.16, 32, 64 + 3, [64, 64, 128], False)
        self.sa3 = PointNetSetAbstraction(200, 0.32, 32, 128 + 3, [128, 128, 256], False)

        self.fp3 = PointNetFeaturePropagation(384, [256, 256])
        self.fp2 = PointNetFeaturePropagation(320, [256, 128])
        self.fp1 = PointNetFeaturePropagation(128, [128, 128, 128])
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.final_conv_diag = nn.Conv1d(128, 3, kernel_size=1)
        self.final_conv_triang = nn.Conv1d(128, 3, kernel_size=1)
        self.softplus = nn.Softplus()

    def forward(self, xyz):
        l0_points = xyz
        l0_xyz = xyz[:,:3,:]

        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)

        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points)

        x = F.relu(self.bn1(self.conv1(l0_points)))

        diag = self.final_conv_diag(x)
        triang = self.final_conv_triang(x)
        diag = self.softplus(diag)
        out= torch.cat([diag,triang],dim=1)

        return out