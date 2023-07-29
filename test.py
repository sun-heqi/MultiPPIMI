import argparse
import copy
import sys
import time
import random

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import DataLoader

sys.path.insert(0, './src')
from datasets.PPIM_datasets_CV import MoleculeProteinDataset, performance_evaluation
from molecule_gnn_model import GNNComplete
from MultiPPIMI import MoleculeProteinModel, ProteinModel

from ipdb import set_trace

def predicting(repurpose_model, device, dataloader):
    repurpose_model.eval()
    total_preds = []
    total_labels = []
    with torch.no_grad():
        for batch in dataloader:
            molecule, rdkit_descriptors, protein_bert, label = batch
            molecule = molecule.to(device)
            rdkit_descriptors = rdkit_descriptors.to(device)
            protein_bert = protein_bert.to(device)
            label = label.to(device)
            pred = repurpose_model(molecule, rdkit_descriptors, protein_bert).squeeze()
            if pred.ndim == 1:
                pred = pred.unsqueeze(0)
            total_preds.append(pred.detach().cpu())
            total_labels.append(label.detach().cpu())

    total_preds = torch.cat(total_preds, dim=0)
    total_labels = torch.cat(total_labels, dim=0)
    return total_labels.numpy(), total_preds.numpy()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch implementation of MultiPPIMI')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--eval_setting', type=str, default='S1', choices=['S1', 'S2', 'S3', 'S4'])
    parser.add_argument('--fold', type=str)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--runseed', type=int, default=123)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--learning_rate', type=float, default=0.0005)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--input_model_file', type=str, default='')
    parser.add_argument('--output_model_file', type=str, default='')
    parser.add_argument('--out_path', type=str)
    ########## For compound embedding ##########
    parser.add_argument('--num_layer', type=int, default=5)
    parser.add_argument('--emb_dim', type=int, default=300)
    parser.add_argument('--dropout_ratio', type=float, default=0.)
    parser.add_argument('--JK', type=str, default='last')
    parser.add_argument('--gnn_type', type=str, default='gin')
    ########## For protein embedding ##########
    parser.add_argument('--protein_hidden_dim', type=int, default=1318)
    parser.add_argument('--num_features', type=int, default=25)
    ########## For attention module ##########
    parser.add_argument('--h_dim', type=int, default=512)
    parser.add_argument('--n_heads', type=int, default=2)
    args = parser.parse_args()

    ### set random seeds
    torch.manual_seed(args.runseed)
    np.random.seed(args.runseed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    device = torch.device(f'cuda:{args.device}')
    print(device)

    ########## Set up dataset and dataloader ##########
    test_dataset = MoleculeProteinDataset(mode='test', setting=args.eval_setting, fold=args.fold)
    print('size of test: {}'.format(len(test_dataset)))
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)

    ########## Set up model ##########
    print('Model path: ', args.input_model_file)
    molecule_model = GNNComplete(args.num_layer, args.emb_dim, JK=args.JK, drop_ratio=args.dropout_ratio, gnn_type=args.gnn_type)
    protein_model = ProteinModel()
    PPIMI_model = MoleculeProteinModel(
        molecule_model, protein_model,
        molecule_emb_dim=310, 
        protein_emb_dim=args.protein_hidden_dim, 
        device=device,
        h_dim=args.h_dim, n_heads=args.n_heads
        ).to(device)
    PPIMI_model.load_state_dict(torch.load(args.input_model_file))
    print('MultiPPIMI model\n', PPIMI_model)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(PPIMI_model.parameters(), lr=args.learning_rate)

    start_time = time.time()
    print('Test results:')
    G, P = predicting(PPIMI_model, device, test_dataloader)
    roc_auc, aupr, precision, accuracy, recall, f1, specificity, mcc, pred_labels = performance_evaluation(P, G)
    print('AUC:\t{}'.format(roc_auc))
    print('AUPR:\t{}'.format(aupr))
    print('precision:\t{}'.format(precision))
    print('accuracy:\t{}'.format(accuracy))
    print('recall:\t{}'.format(recall))
    print('f1:\t{}'.format(f1))
    print('specificity:\t{}'.format(specificity))
    print('mcc:\t{}'.format(mcc))
    print('Took {:.5f}s.'.format(time.time() - start_time))
