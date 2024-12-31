import torch
import pandas as pd
from torch_geometric.data import Data
from prediction_gat import GATNet
from training_p_model import cross_validate
from datatcreate_p_model import *

data = pd.read_csv('GCGR_P_train_test.csv')

smiles_list = data['standard ismiles'].tolist()
labels = data['ActivityValue'].tolist()

min_label = min(labels)
if min_label <= 0:
    labels = [label - min_label + 1 for label in labels]

# Convert to log scale
log_labels = np.log(labels)

min_label = min(log_labels)
max_label = max(log_labels)

normalized_labels = normalize_labels(log_labels, min_label, max_label)

graph_data_list = []
for i, smiles in enumerate(smiles_list):
    result = smile_to_graph(smiles)
    if result is not None:
        c_size, features, edge_index = result
        x = torch.tensor(features, dtype=torch.float)
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        y = torch.tensor([normalized_labels[i]], dtype=torch.float)  # Use normalized labels
        data = Data(x=x, edge_index=edge_index, y=y)
        graph_data_list.append(data)

if __name__ == '__main__':
    num_folds = 5
    min_label = min(normalized_labels)
    max_label = max(normalized_labels)

    model = GATNet(num_features_xd=78, output_dim=128, dropout=0.2)
    mean_mse, std_mse, mean_mae, std_mae, mean_r2, std_r2 = cross_validate(model, graph_data_list, min_label, max_label,
                                                                           k=5, epochs=200)

    print(f'Final Cross-Validation Results: MSE: {mean_mse:.4f}, MAE: {mean_mae:.4f}, R²: {mean_r2:.4f}')
