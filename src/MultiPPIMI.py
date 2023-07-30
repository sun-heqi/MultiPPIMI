import torch
from torch import nn
from torch_geometric.nn import global_mean_pool
import torch.nn.functional as F
import math
from ban import BANLayer
from torch.nn.utils.weight_norm import weight_norm



class MultiPPIMI(nn.Module):
    def __init__(self, modulator_model, modulator_emb_dim, ppi_emb_dim, 
                 h_dim, n_heads, 
                 output_dim=2, dropout=0.2, device=None, attention=False):
        super(MultiPPIMI, self).__init__()
        self.attention = attention
        self.modulator_emb_dim = modulator_emb_dim
        self.ppi_emb_dim = ppi_emb_dim
        self.modulator_model = modulator_model

        ##### bilinear attention #####
        self.bcn = weight_norm(
            BANLayer(v_dim=modulator_emb_dim, q_dim=ppi_emb_dim, h_dim=h_dim, h_out=n_heads, k=3),
            name='h_mat', dim=None)

        self.fc1 = nn.Linear(h_dim, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.out = nn.Linear(256, output_dim)
        self.pool = global_mean_pool
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, modulator, rdkit_descriptors, ppi_feats):
        modulator_node_repr = self.modulator_model(modulator)
        modulator_repr = self.pool(modulator_node_repr, modulator.batch)
        modulator_repr = torch.cat((modulator_repr, rdkit_descriptors), 1)

        x, att = self.bcn(modulator_repr, ppi_feats)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.out(x)

        return x
        