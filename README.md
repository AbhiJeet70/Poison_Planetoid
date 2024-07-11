# Poison_Planetoid

This repository contains code for conducting experiments on the Planetoid datasets (Cora, Pubmed, CiteSeer) using Graph Convolutional Networks (GCNs) with edge perturbations. The experiments include hyperparameter tuning and evaluation of the impact of evenly and concentrated edge perturbations on the model performance.


## Code Structure
```bash
main.py: Main script for running the experiments.
data_utils.py: Utility functions for loading and processing data.
models.py: Definition of the GCN model.
train_eval.py: Functions for training and evaluating the model.
perturbation_utils.py: Functions for applying edge perturbations.
```

## Installation

Install the necessary dependencies:

  ```bash
  pip install torch torch_geometric numpy pandas matplotlib
  ```

## Usage
To run the experiments, execute the following command:

  ```bash
  python main.py
  ```

This will load the datasets, train the models, apply perturbations, and save the results to perturbation_results.csv.


## Results
The results of the experiments, including plots of the test accuracy against the perturbation percentage, will be saved in the current directory.

## Files
perturbation_results.csv: Contains the results of the experiments.

## License
This project is licensed under the MIT License - see the LICENSE file for details.
