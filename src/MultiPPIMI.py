import torch
from torch import nn
from torch_geometric.nn import global_mean_pool
import torch.nn.functional as F
import math
from ban import BANLayer
from torch.nn.utils.weight_norm import weight_norm

from ipdb import set_trace



class ProteinModel(nn.Module):
    def __init__(self):
        super(ProteinModel, self).__init__()

    def forward(self, x):

        return x


class MoleculeProteinModel(nn.Module):
    def __init__(self, molecule_model, protein_model, molecule_emb_dim, protein_emb_dim, 
                 h_dim, n_heads, 
                 output_dim=2, dropout=0.2, device=None, attention=False):
        super(MoleculeProteinModel, self).__init__()
        self.attention = attention
        self.molecule_emb_dim = molecule_emb_dim
        self.protein_emb_dim = protein_emb_dim
        self.molecule_model = molecule_model
        self.protein_model = protein_model

        ##### bilinear attention #####
        self.bcn = weight_norm(
            BANLayer(v_dim=molecule_emb_dim, q_dim=protein_emb_dim, h_dim=h_dim, h_out=n_heads, k=3),
            name='h_mat', dim=None)

        self.fc1 = nn.Linear(h_dim, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.out = nn.Linear(256, output_dim)
        self.pool = global_mean_pool
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, molecule, rdkit_descriptors, ppi_feats):
        molecule_node_repr = self.molecule_model(molecule)
        molecule_repr = self.pool(molecule_node_repr, molecule.batch)
        molecule_repr = torch.cat((molecule_repr, rdkit_descriptors), 1)
        protein_repr = self.protein_model(ppi_feats)

        x, att = self.bcn(molecule_repr, protein_repr)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.out(x)

        return x
        