import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from SkinCancer.utils.logging import debug_feature_map, debug_tensor


class LightweightTopoBlock(nn.Module):
    def __init__(self, in_channels, reduced_dim=32):
        super().__init__()
        self.reduce = nn.Conv2d(in_channels, reduced_dim, kernel_size=1, bias=False)
        self.streak_h = nn.Conv2d(reduced_dim, reduced_dim, kernel_size=(1, 3), padding=(0, 1), groups=reduced_dim)
        self.streak_v = nn.Conv2d(reduced_dim, reduced_dim, kernel_size=(3, 1), padding=(1, 0), groups=reduced_dim)
        self.grid_dilated = nn.Conv2d(reduced_dim, reduced_dim, kernel_size=3, padding=2, dilation=2, groups=reduced_dim)
        self.fusion = nn.Conv2d(reduced_dim * 3, in_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(in_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        debug_feature_map("LightweightTopoBlock.input", x)
        residual = x
        x_red = self.reduce(x)
        debug_feature_map("LightweightTopoBlock.reduce.after", x_red)
        feat_h = self.streak_h(x_red)
        debug_feature_map("LightweightTopoBlock.streak_h.after", feat_h)
        feat_v = self.streak_v(x_red)
        debug_feature_map("LightweightTopoBlock.streak_v.after", feat_v)
        feat_grid = self.grid_dilated(x_red)
        debug_feature_map("LightweightTopoBlock.grid_dilated.after", feat_grid)
        combined = torch.cat([feat_h, feat_v, feat_grid], dim=1)
        debug_feature_map("LightweightTopoBlock.concat.after", combined)
        out = self.act(self.bn(self.fusion(combined)))
        debug_feature_map("LightweightTopoBlock.fusion_bn_act.after", out)
        output = out + residual
        debug_feature_map("LightweightTopoBlock.output", output)
        return output


class MultiScaleMicroEncoder(nn.Module):
    def __init__(self, in_channels=256, out_channels=256):
        super().__init__()

        def branch(k, padding, dilation=1):
            return nn.Sequential(
                nn.Conv2d(in_channels, 64, k, padding=padding, dilation=dilation, bias=False),
                nn.BatchNorm2d(64),
                nn.GELU(),
            )

        self.branch1 = branch(1, 0)
        self.branch2 = branch(3, 1, dilation=1)
        self.branch3 = branch(3, 2, dilation=2)
        self.branch4 = branch(3, 4, dilation=4)
        self.fuse = nn.Sequential(
            nn.Conv2d(64 * 4, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )
        self.gamma = nn.Parameter(torch.tensor(0.2))

    def forward(self, x):
        debug_feature_map("MultiScaleMicroEncoder.input", x)
        b1 = self.branch1(x)
        debug_feature_map("MultiScaleMicroEncoder.branch1.after", b1)
        b2 = self.branch2(x)
        debug_feature_map("MultiScaleMicroEncoder.branch2.after", b2)
        b3 = self.branch3(x)
        debug_feature_map("MultiScaleMicroEncoder.branch3.after", b3)
        b4 = self.branch4(x)
        debug_feature_map("MultiScaleMicroEncoder.branch4.after", b4)
        ms = torch.cat([b1, b2, b3, b4], dim=1)
        debug_feature_map("MultiScaleMicroEncoder.concat.after", ms)
        out = self.fuse(ms)
        debug_feature_map("MultiScaleMicroEncoder.fuse.after", out)
        output = x + self.gamma * out
        debug_feature_map("MultiScaleMicroEncoder.output", output)
        return output


class GraphConvolutionLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = nn.Parameter(torch.FloatTensor(out_features)) if bias else None
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x, adj):
        debug_tensor("GraphConvolutionLayer.x.before", x)
        debug_tensor("GraphConvolutionLayer.adj.before", adj)
        out = torch.matmul(adj, torch.matmul(x, self.weight))
        if self.bias is not None:
            out = out + self.bias
        debug_tensor("GraphConvolutionLayer.out.after", out)
        return out


class CachedSpatialGCNModule(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, k_neighbors=8, downsample_ratio=2):
        super().__init__()
        self.topo_enhancer = LightweightTopoBlock(in_channels, reduced_dim=max(16, in_channels // 4))
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.k_neighbors = k_neighbors
        self.downsample = nn.Conv2d(in_channels, in_channels, kernel_size=downsample_ratio, stride=downsample_ratio, bias=False)
        self.gcn1 = GraphConvolutionLayer(in_channels, hidden_channels)
        self.gcn2 = GraphConvolutionLayer(hidden_channels, out_channels)
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(0.2)
        self.upsample = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=downsample_ratio, stride=downsample_ratio)
        self.register_buffer("adj_cache", None)
        self.cached_shape = None

    def _get_adjacency_matrix(self, H, W, device):
        if self.adj_cache is not None and self.cached_shape == (H, W):
            return self.adj_cache
        N = H * W
        y_coords = torch.arange(H, device=device, dtype=torch.float32)
        x_coords = torch.arange(W, device=device, dtype=torch.float32)
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing="ij")
        coords = torch.stack([yy.flatten(), xx.flatten()], dim=1)
        k = min(self.k_neighbors + 1, N)
        adj = torch.zeros(N, N, device=device, dtype=torch.float16)
        chunk_size = 1024
        for i in range(0, N, chunk_size):
            end_i = min(i + chunk_size, N)
            dist_chunk = torch.cdist(coords[i:end_i], coords, p=2)
            _, indices = torch.topk(dist_chunk, k, largest=False, dim=1)
            row_indices = torch.arange(i, end_i, device=device).unsqueeze(1).expand(-1, k)
            adj[row_indices, indices] = 1.0
        adj = (adj + adj.t()) / 2.0
        adj = adj + torch.eye(N, device=device, dtype=torch.float16)
        degree = adj.sum(dim=1).float()
        d_inv_sqrt = torch.pow(degree, -0.5)
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
        d_inv_sqrt = d_inv_sqrt.to(torch.float16)
        adj_norm = (d_inv_sqrt.unsqueeze(1) * adj * d_inv_sqrt.unsqueeze(0)).float()
        self.adj_cache = adj_norm.unsqueeze(0)
        self.cached_shape = (H, W)
        return self.adj_cache

    def forward(self, x):
        debug_feature_map("CachedSpatialGCNModule.input", x)
        if x.dim() == 4 and x.shape[1] != self.in_channels and x.shape[-1] == self.in_channels:
            x = x.permute(0, 3, 1, 2).contiguous()
            debug_feature_map("CachedSpatialGCNModule.input.permuted", x)
        x = self.topo_enhancer(x)
        debug_feature_map("CachedSpatialGCNModule.topo_enhancer.after", x)
        B, C, H, W = x.shape
        x_down = self.downsample(x)
        debug_feature_map("CachedSpatialGCNModule.downsample.after", x_down)
        _, _, H_d, W_d = x_down.shape
        N = H_d * W_d
        adj = self._get_adjacency_matrix(H_d, W_d, x.device).expand(B, -1, -1)
        debug_tensor("CachedSpatialGCNModule.adj", adj)
        node_features = x_down.view(B, C, N).transpose(1, 2)
        debug_tensor("CachedSpatialGCNModule.node_features", node_features)
        h = self.gcn1(node_features, adj)
        h = F.relu(self.bn1(h.transpose(1, 2)).transpose(1, 2))
        debug_tensor("CachedSpatialGCNModule.gcn1_bn_relu.after", h)
        h = self.dropout(h)
        h = self.gcn2(h, adj)
        h = F.relu(self.bn2(h.transpose(1, 2)).transpose(1, 2))
        debug_tensor("CachedSpatialGCNModule.gcn2_bn_relu.after", h)
        output = h.transpose(1, 2).reshape(B, self.out_channels, H_d, W_d)
        debug_feature_map("CachedSpatialGCNModule.reshape.after", output)
        output = self.upsample(output)
        debug_feature_map("CachedSpatialGCNModule.upsample.after", output)
        if output.shape[2:] != (H, W):
            output = F.interpolate(output, size=(H, W), mode="bilinear", align_corners=False)
            debug_feature_map("CachedSpatialGCNModule.interpolate.after", output)
        debug_feature_map("CachedSpatialGCNModule.output", output)
        return output


def smart_reshape(x, channels):
    if x.dim() == 4:
        if x.shape[-1] == channels and x.shape[1] != channels:
            return x.permute(0, 3, 1, 2).contiguous()
        return x
    if x.dim() == 3:
        B, L, C = x.shape
        H = int(np.sqrt(L))
        if H * H == L:
            return x.view(B, H, H, C).permute(0, 3, 1, 2).contiguous()
        return x.view(B, C, H, H)
    return x
