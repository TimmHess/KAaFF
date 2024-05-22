import torch
import torch.nn as nn

# class BarlowTwinLoss(nn.Module):
#     """
#     Code adapted from: https://github.com/IgorSusmelj/barlowtwins/blob/main/loss.py
#     """
#     def __init__(self, projection_dim, device, lambd=5e-3, scale_factor=0.025):
#         super(BarlowTwinLoss, self).__init__()

#         self.lambd = lambd
#         self.scale_factor = scale_factor

#         self.bn = nn.BatchNorm1d(projection_dim, affine=False) # projection_dim = sizes[-1]
#         self.bn = self.bn.to(device)
#         return

#     def off_diagonal(self, x):
#         # return a flattened view of the off-diagonal elements of a square matrix
#         n, m = x.shape
#         assert n == m
#         return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

#     def forward(self, ft_x1, ft_x2):
#         """
#         Code by https://github.com/facebookresearch/barlowtwins/blob/main/main.py
#         """
#         c = torch.mm(self.bn(ft_x1), self.bn(ft_x2).T)
#         c.div_(ft_x1.shape[0])

#         on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
#         off_diag = self.off_diagonal(c).pow_(2).sum()
#         loss = self.scale_factor*(on_diag + self.lambd * off_diag)
#         return loss
    

class BarlowTwinLoss(nn.Module):
    def __init__(self, batch_size, lambda_coeff=5e-3, projection_dim=128):
        super().__init__()

        self.z_dim = projection_dim
        self.batch_size = batch_size
        self.lambda_coeff = lambda_coeff

    def off_diagonal_ele(self, x):
        # taken from: https://github.com/facebookresearch/barlowtwins/blob/main/main.py
        # return a flattened view of the off-diagonal elements of a square matrix
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def forward(self, z1, z2):
        # N x D, where N is the batch size and D is output dim of projection head
        z1_norm = (z1 - torch.mean(z1, dim=0)) / torch.std(z1, dim=0)
        z2_norm = (z2 - torch.mean(z2, dim=0)) / torch.std(z2, dim=0)

        cross_corr = torch.matmul(z1_norm.T, z2_norm) / self.batch_size

        on_diag = torch.diagonal(cross_corr).add_(-1).pow_(2).sum()
        off_diag = self.off_diagonal_ele(cross_corr).pow_(2).sum()

        return on_diag + self.lambda_coeff * off_diag