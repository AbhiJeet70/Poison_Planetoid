import torch
import numpy as np
import random

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures

def load_planetoid_data(dataset_name):
    dataset = Planetoid(root=f'/tmp/{dataset_name}', name=dataset_name, transform=NormalizeFeatures())
    data = dataset[0]
    return data


import torch
import numpy as np

def split_indices(num_nodes, train_ratio=0.7, val_ratio=0.1):
    indices = np.random.permutation(num_nodes)
    train_end = int(train_ratio * num_nodes)
    val_end = int((train_ratio + val_ratio) * num_nodes)
    train_idx = torch.tensor(indices[:train_end], dtype=torch.long)
    val_idx = torch.tensor(indices[train_end:val_end], dtype=torch.long)
    test_idx = torch.tensor(indices[val_end:], dtype=torch.long)
    return train_idx, val_idx, test_idx

import torch
import torch.nn.functional as F

def train_model(model, pyg_data, lr, weight_decay):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pyg_data = pyg_data.to(device)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_val_acc = 0
    patience = 100
    patience_counter = 0

    for epoch in range(1, 501):
        model.train()
        optimizer.zero_grad()
        out = model(pyg_data.x, pyg_data.edge_index)
        loss = F.cross_entropy(out[pyg_data.train_mask], pyg_data.y[pyg_data.train_mask])
        loss.backward()
        optimizer.step()

        model.eval()
        _, pred = model(pyg_data.x, pyg_data.edge_index).max(dim=1)
        val_correct = float(pred[pyg_data.val_mask].eq(pyg_data.y[pyg_data.val_mask]).sum().item())
        val_acc = val_correct / pyg_data.val_mask.sum().item()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            best_model_state = model.state_dict()
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f'Early stopping at epoch {epoch}')
            break

    model.load_state_dict(best_model_state)
    model.eval()
    _, pred = model(pyg_data.x, pyg_data.edge_index).max(dim=1)
    correct = float(pred[pyg_data.test_mask].eq(pyg_data.y[pyg_data.test_mask]).sum().item())
    acc = correct / pyg_data.test_mask.sum().item()
    return acc

def print_dataset_statistics(data, dataset_name):
    num_nodes = data.num_nodes
    num_edges = data.num_edges
    num_features = data.num_node_features
    num_classes = data.y.max().item() + 1
    class_distribution = torch.bincount(data.y).cpu().numpy()
    print(f"Statistics for {dataset_name}:")
    print(f"  Number of nodes: {num_nodes}")
    print(f"  Number of edges: {num_edges}")
    print(f"  Number of features: {num_features}")
    print(f"  Number of classes: {num_classes}")
    print(f"  Class distribution: {class_distribution}")

import torch

def evenly_perturb_edges(data, perturbation_percentage, ascending=False):
    device = data.edge_index.device
    edge_index = data.edge_index.clone().to(device)
    num_edges = edge_index.size(1)
    num_perturbations = int(num_edges * perturbation_percentage)
    total_perturbations = 0

    degrees = torch.zeros(data.num_nodes, dtype=torch.long, device=device)
    degrees.scatter_add_(0, edge_index[0], torch.ones(edge_index.size(1), dtype=torch.long, device=device))
    sorted_nodes = torch.argsort(degrees)
    
    if not ascending:
        sorted_nodes = sorted_nodes.flip(dims=[0])
    
    perturbations_per_node = num_perturbations // data.num_nodes

    for node in sorted_nodes:
        if total_perturbations >= num_perturbations:
            break
        connected_edges = (edge_index[0] == node) | (edge_index[1] == node)
        num_node_edges = connected_edges.sum().item()
        num_perturb_node_edges = min(perturbations_per_node, num_node_edges)

        if num_perturb_node_edges > 0:
            perturb_edges_idx = torch.nonzero(connected_edges, as_tuple=False).view(-1)
            perturb_edges_idx = perturb_edges_idx[torch.randperm(perturb_edges_idx.size(0))[:num_perturb_node_edges]]
            edge_index[:, perturb_edges_idx] = torch.randint(0, data.num_nodes, edge_index[:, perturb_edges_idx].shape, dtype=torch.long, device=device)
            total_perturbations += num_perturb_node_edges

    data.edge_index = edge_index
    return data

def concentrated_perturb_edges(data, perturbation_percentage, top_k=20):
    device = data.edge_index.device
    edge_index = data.edge_index.clone().to(device)
    num_edges = edge_index.size(1)
    num_perturbations = int(num_edges * perturbation_percentage)
    total_perturbations = 0

    degrees = torch.zeros(data.num_nodes, dtype=torch.long, device=device)
    degrees.scatter_add_(0, edge_index[0], torch.ones(edge_index.size(1), dtype=torch.long, device=device))
    sorted_nodes = torch.argsort(degrees, descending=True)
    
    top_k_nodes = sorted_nodes[:top_k]

    for node in top_k_nodes:
        if total_perturbations >= num_perturbations:
            break
        connected_edges = (edge_index[0] == node) | (edge_index[1] == node)
        perturb_edges_idx = torch.nonzero(connected_edges, as_tuple=False).view(-1)
        num_perturb_node_edges = min(len(perturb_edges_idx), num_perturbations - total_perturbations)

        if num_perturb_node_edges > 0:
            perturb_edges_idx = perturb_edges_idx[torch.randperm(len(perturb_edges_idx))[:num_perturb_node_edges]]
            edge_index[:, perturb_edges_idx] = torch.randint(0, data.num_nodes, edge_index[:, perturb_edges_idx].shape, dtype=torch.long, device=device)
            total_perturbations += num_perturb_node_edges

    data.edge_index = edge_index
    return data, top_k_nodes



