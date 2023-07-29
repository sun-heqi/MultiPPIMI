import os

import numpy as np
import pandas as pd
import torch
from rdkit.Chem import AllChem
from torch_geometric.data import InMemoryDataset
from torch.autograd import Variable
from sklearn.metrics import roc_auc_score, auc, precision_recall_curve, \
        precision_score, recall_score, \
        f1_score, confusion_matrix, accuracy_score, matthews_corrcoef
from .molecule_datasets import mol_to_graph_data_obj_simple

from ipdb import set_trace

seq_voc = "ABCDEFGHIKLMNOPQRSTUVWXYZ"
seq_dict = {v:(i+1) for i,v in enumerate(seq_voc)}
seq_dict_len = len(seq_dict)
max_seq_len = 1000


def seq_cat(prot):
    x = np.zeros(max_seq_len)
    for i, ch in enumerate(prot[:max_seq_len]):
        x[i] = seq_dict[ch]
    return x

def get_best_threshold(output, labels):
    preds = output[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(labels, preds)
    f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-20)
    best_threshold = thresholds[f1_scores.argmax()]
    return best_threshold


def performance_evaluation(output, labels):
    output = torch.softmax(torch.from_numpy(output), dim=1)
    pred_scores = output[:, 1]
    roc_auc = roc_auc_score(labels, pred_scores)
    prec, reca, _ = precision_recall_curve(labels, pred_scores)
    aupr = auc(reca, prec)

    best_threshold = get_best_threshold(output, labels)
    pred_labels = output[:, 1] > best_threshold
    precision = precision_score(labels, pred_labels)
    accuracy = accuracy_score(labels, pred_labels)
    recall = recall_score(labels, pred_labels)
    f1 = f1_score(labels, pred_labels)
    (tn, fp, fn, tp) = confusion_matrix(labels, pred_labels).ravel()
    specificity = tn / (tn + fp)
    mcc = matthews_corrcoef(labels, pred_labels)

    return roc_auc, aupr, precision, accuracy, recall, f1, specificity, mcc, pred_labels


class MoleculeProteinDataset(InMemoryDataset):
    def __init__(self, mode, setting, fold):
        super(InMemoryDataset, self).__init__()
        datapath = f"./data/folds/{setting}/{mode}_fold{fold}.csv"
        print('datapath\t', datapath)

        self.process_molecule()
        self.process_protein()

        df = pd.read_csv(datapath)
        self.molecule_index_list = df['SMILES'].tolist()
        self.protein_index1_list = df['uniprot_id1'].tolist()
        self.protein_index2_list = df['uniprot_id2'].tolist()
        self.label_list = df['label'].tolist()
        self.label_list = torch.LongTensor(self.label_list)

        return

    def process_molecule(self):
        rdkit_descriptors_path = './data/features/compound_phy.tsv'
        rdkit_descriptors = pd.read_csv(rdkit_descriptors_path, sep=' ')
        rdkit_descriptors = rdkit_descriptors.rename(columns={"smiles":"SMILES"})
        self.rdkit_descriptors = torch.FloatTensor(rdkit_descriptors.drop('SMILES', axis=1).to_numpy())

        smiles_list = rdkit_descriptors['SMILES']
        self.smiles_list = smiles_list.tolist()

        rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]
        preprocessed_rdkit_mol_objs_list = [m if m != None else None for m in rdkit_mol_objs_list]
        preprocessed_smiles_list = [AllChem.MolToSmiles(m) if m != None else None for m in preprocessed_rdkit_mol_objs_list]
        assert len(smiles_list) == len(preprocessed_rdkit_mol_objs_list)
        assert len(smiles_list) == len(preprocessed_smiles_list)

        smiles_list, rdkit_mol_objs = preprocessed_smiles_list, preprocessed_rdkit_mol_objs_list

        data_list = []
        for i in range(len(smiles_list)):
            rdkit_mol = rdkit_mol_objs[i]
            if rdkit_mol != None:
                data = mol_to_graph_data_obj_simple(rdkit_mol)
                data.id = torch.tensor([i])
                data_list.append(data)

        self.molecule_graph_list = data_list

        return

    def process_protein(self):
        ### Load pre-trained bert data
        ESM2 = pd.read_csv(f"./data/features/protein_esm2.csv", header=None)
        ESM2 = ESM2.rename(columns={0:"uniprot_id"})
        ESM2.sort_values('uniprot_id', inplace=True)
        pfeature = pd.read_csv('./data/features/protein_phy.csv')
        pfeature.sort_values('uniprot_id', inplace=True)
        assert pfeature.uniprot_id.tolist() == ESM2.uniprot_id.tolist()

        protein_feats_list = np.concatenate((ESM2.iloc[:, 1:], pfeature.iloc[:, 1:]), 1)
        self.uniprot_id_list = ESM2['uniprot_id'].tolist()
        self.protein_feats_list = torch.FloatTensor(protein_feats_list)
        return

    def __getitem__(self, idx):
        molecule_idx = self.smiles_list.index(self.molecule_index_list[idx])
        molecule_graph = self.molecule_graph_list[molecule_idx]
        rdkit_descriptors = self.rdkit_descriptors[molecule_idx]

        protein_idx1 = self.uniprot_id_list.index(self.protein_index1_list[idx])
        protein_feats1 = self.protein_feats_list[protein_idx1]    # for target 1
        protein_idx2 = self.uniprot_id_list.index(self.protein_index2_list[idx])
        protein_feats2 = self.protein_feats_list[protein_idx2]    # for target 2
        protein_feats = torch.cat((protein_feats1, protein_feats2), 0)

        label = self.label_list[idx]
        return molecule_graph, rdkit_descriptors, protein_feats, label

    def __len__(self):
        return len(self.label_list)
