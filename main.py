
import torch
import matplotlib.pyplot as plt
import pandas as pd
from model import GCNNet
from utils import set_seed, load_planetoid_data, split_indices, train_model, print_dataset_statistics, evenly_perturb_edges, concentrated_perturb_edges

def plot_accuracies(results, dataset_name):
    for strategy, accuracies in results.items():
        perturbation_percentages = list(accuracies.keys())
        values = list(accuracies.values())
        plt.plot(perturbation_percentages, values, label=strategy)
    plt.xlabel('Perturbation Percentage')
    plt.ylabel('Accuracy')
    plt.title(f'{dataset_name} Accuracy vs. Perturbation Percentage')
    plt.legend()
    plt.savefig(f'{dataset_name}_accuracies.png')
    plt.show()

def save_accuracies_to_csv(results, dataset_name):
    df = pd.DataFrame(results)
    df.to_csv(f'{dataset_name}_accuracies.csv', index_label='Perturbation Percentage')

def main():
    set_seed(42)

    dataset_name = 'Cora'
    data = load_planetoid_data(dataset_name)
    print_dataset_statistics(data, dataset_name)

    num_nodes = data.num_nodes
    train_idx, val_idx, test_idx = split_indices(num_nodes)
    data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    data.train_mask[train_idx] = True
    data.val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    data.val_mask[val_idx] = True
    data.test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    data.test_mask[test_idx] = True

    in_channels = data.num_features
    hidden_channels = 16
    out_channels = len(data.y.unique())

    model = GCNNet(in_channels, hidden_channels, out_channels)

    clean_acc = train_model(model, data, lr=0.01, weight_decay=5e-4)
    print(f'Clean accuracy: {clean_acc}')

    perturbation_percentages = [0.05, 0.10, 0.15, 0.20, 0.25]
    results = {
        'Clean': {},
        'Evenly Perturbed': {},
        'Concentrated Perturbed': {}
    }

    results['Clean'][0.0] = clean_acc

    for percentage in perturbation_percentages:
        perturbed_data = evenly_perturb_edges(data.clone(), percentage)
        acc = train_model(model, perturbed_data, lr=0.01, weight_decay=5e-4)
        results['Evenly Perturbed'][percentage] = acc

        perturbed_data, _ = concentrated_perturb_edges(data.clone(), percentage)
        acc = train_model(model, perturbed_data, lr=0.01, weight_decay=5e-4)
        results['Concentrated Perturbed'][percentage] = acc

    save_accuracies_to_csv(results, dataset_name)
    plot_accuracies(results, dataset_name)

if __name__ == '__main__':
    main()
