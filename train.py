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


def train(repurpose_model, device, dataloader, optimizer):
    repurpose_model.train()
    loss_accum = 0
    for step_idx, batch in enumerate(dataloader):
        molecule, rdkit_descriptors, protein_bert, label = batch
        molecule = molecule.to(device)
        rdkit_descriptors = rdkit_descriptors.to(device)
        protein_bert = protein_bert.to(device)
        label = label.to(device)
        pred = repurpose_model(molecule, rdkit_descriptors, protein_bert).squeeze()

        optimizer.zero_grad()
        loss = criterion(pred, label)
        loss.backward()
        optimizer.step()
        loss_accum += loss.detach().item()
    print('Loss:\t{}'.format(loss_accum / len(dataloader)))


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
    parser.add_argument('--pretrained_model_file', type=str, default='./src/GraphMVP_C.model')
    parser.add_argument('--output_model_file', type=str, default='')
    parser.add_argument('--out_path', type=str, default='.')
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
    train_dataset = MoleculeProteinDataset(mode='train', setting=args.eval_setting, fold=args.fold)    
    valid_dataset = MoleculeProteinDataset(mode='valid', setting=args.eval_setting, fold=args.fold)
    print('size of train: {}\tval: {}'.format(len(train_dataset), len(valid_dataset)))

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)

    ########## Set up model ##########
    molecule_model = GNNComplete(args.num_layer, args.emb_dim, JK=args.JK, drop_ratio=args.dropout_ratio, gnn_type=args.gnn_type)
    if not args.pretrained_model_file == '':
        print('========= Loading from {}'.format(args.pretrained_model_file))
        molecule_model.load_state_dict(torch.load(args.pretrained_model_file))
    protein_model = ProteinModel()
    PPIMI_model = MoleculeProteinModel(
        molecule_model, protein_model,
        molecule_emb_dim=310, 
        protein_emb_dim=args.protein_hidden_dim, 
        device=device,
        h_dim=args.h_dim, n_heads=args.n_heads
        ).to(device)
    print('MultiPPIMI model\n', PPIMI_model)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(PPIMI_model.parameters(), lr=args.learning_rate)

    best_model = None
    best_roc_auc = 0
    best_epoch = 0

    print('Start training...')
    train_start_time = time.time()
    for epoch in range(1, 1+args.epochs):
        start_time = time.time()
        print('Start training at epoch: {}'.format(epoch))
        train(PPIMI_model, device, train_dataloader, optimizer)

        G, P = predicting(PPIMI_model, device, valid_dataloader)
        current_roc_auc, current_aupr, precision, accuracy, recall, f1, specificity, mcc, pred_labels = performance_evaluation(P, G)
        print('Val AUC:\t{}'.format(current_roc_auc))
        print('Val AUPR:\t{}'.format(current_aupr))
        if current_roc_auc > best_roc_auc:
            best_model = copy.deepcopy(PPIMI_model)
            best_roc_auc = current_roc_auc
            best_epoch = epoch
            print('ROC-AUC improved at epoch {}\tbest ROC-AUC: {}'.format(best_epoch, best_roc_auc))
        else:
            print('No improvement since epoch {}\tbest ROC-AUC: {}'.format(best_epoch, best_roc_auc))
        print('Took {:.5f}s.'.format(time.time() - start_time))
        print()

    print('Finish training!')
    print('Total training time: {:.5f} hours'.format((time.time()-train_start_time)/3600))
    start_time = time.time()
    print('Last epoch validation results: {}'.format(args.epochs))
    G, P = predicting(PPIMI_model, device, valid_dataloader)
    roc_auc, aupr, precision, accuracy, recall, f1, specificity, mcc, pred_labels = performance_evaluation(P, G)
    print('AUC:\t{}'.format(roc_auc))
    print('AUPR:\t{}'.format(aupr))
    print('precision:\t{}'.format(precision))
    print('accuracy:\t{}'.format(accuracy))
    print('recall:\t{}'.format(recall))
    print('f1:\t{}'.format(f1))
    print('specificity:\t{}'.format(specificity))
    print('mcc:\t{}'.format(mcc))
    print('')
    print('Took {:.5f}s.'.format(time.time() - start_time))

    start_time = time.time()
    print('Best epoch validation results: {}'.format(best_epoch))
    G, P = predicting(best_model, device, valid_dataloader)
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

    # save best model
    model_path = args.out_path + f'/setting_{args.eval_setting}_fold{args.fold}.model'
    torch.save(best_model.state_dict(), model_path)
