import torch
import pandas as pd
from rdkit import Chem
from torch_geometric.data import Data
import numpy as np
from prediction_gat import GATNet
import networkx as nx
import pickle

# Loading min_label and max_label
with open('label_range.pkl', 'rb') as f:
    min_label, max_label = pickle.load(f)

# Define the antinomial and inverse normalisation functions
def denormalize_labels(labels, min_val, max_val):
    """Anti-standardisation labelling"""
    return labels * (max_val - min_val) + min_val

def inverse_log(labels):
    """Anticode conversion"""
    return np.exp(labels)

df = pd.read_csv('T2DM_TCM.csv')

smiles_list = df['i_SMILES'].tolist()
ids = df['CID'].tolist()

def atom_features(atom):
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),
                                          ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca',
                                           'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag',
                                           'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni',
                                           'Cd', 'In', 'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb', 'Unknown']) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    [atom.GetIsAromatic()])

def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))

def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

def smile_to_graph(smile):
    mol = Chem.MolFromSmiles(smile)
    if mol is None:
        print(f"Warning: Failed to parse SMILES: {smile}")
        return None, None, None
    
    c_size = mol.GetNumAtoms()
    features = [atom_features(atom) for atom in mol.GetAtoms()]
    
    edges = [[bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()] for bond in mol.GetBonds()]
    
    # If there are no valid edges, skip this SMILES
    if len(edges) == 0:
        print(f"Skipping SMILES {smile} due to no edges.")
        return None, None, None 


    g = nx.Graph(edges).to_directed()

    edge_index = np.array([[e1, e2] for e1, e2 in g.edges], dtype=np.long).T  

    # Check that the edge_index meets the requirements
    if edge_index.shape[0] != 2:
        print(f"Skipping SMILES {smile} due to invalid edge_index shape {edge_index.shape}")
        return None, None, None  

    print(f"Edge index shape: {edge_index.shape}")
    print(f"x shape: {np.array(features).shape}")

    return c_size, np.array(features), edge_index


# Converting SMILES to graph data
graph_data_list = []
for i, smiles in enumerate(smiles_list):
    result = smile_to_graph(smiles)
    if result is not None:
        c_size, features, edge_index = result
        if features is not None:
            x = torch.tensor(features, dtype=torch.float)
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
            data = Data(x=x, edge_index=edge_index) 
            graph_data_list.append(data)
        else:
            print(f"Skipping SMILES with invalid features: {smiles}")  # Printing unmanageable SMILES
    else:
        print(f"Skipping SMILES {smiles} due to invalid graph.")  # Printing unprocessable SMILES (cannot be parsed as a graph)

# Define a function to load the model for each fold
def load_model(fold_num):
    model = GATNet(num_features_xd=78, output_dim=128, dropout=0.2)
    model.load_state_dict(torch.load(f'fold_{fold_num}_model.pth'))
    model.eval()
    return model

def predict_smiles(model, smile):
    result = smile_to_graph(smile)
    if result is None:
        print(f"Skipping SMILES {smile} due to invalid graph.")
        return np.nan  # If it cannot be parsed, return NaN to skip the SMILES.

    c_size, features, edge_index = result

    if edge_index is None or edge_index.shape[1] == 0:
        print(f"Skipping SMILES {smile} due to invalid edge_index shape {edge_index.shape if edge_index is not None else 'None'}")
        return np.nan  # If edge_index is invalid, skip the SMILES.
    
    x = torch.tensor(features, dtype=torch.float)
    edge_index = torch.tensor(edge_index, dtype=torch.long)

    data = Data(x=x, edge_index=edge_index)

    with torch.no_grad():
        output = model(data)
    
    # Anti-standardisation
    denormalized_output = denormalize_labels(output.item(), min_label, max_label)
    
    # Anticode conversion
    restored_output = inverse_log(denormalized_output)

    return restored_output

# Predictions for each SMILES
predictions = []
for i, smile in enumerate(smiles_list):
    fold_predictions = []
    for fold_num in range(1, 6):
        model = load_model(fold_num)
        pred = predict_smiles(model, smile)
        if pred is not None:
            fold_predictions.append(pred)
    if fold_predictions:
        avg_pred = np.mean(fold_predictions)
        predictions.append(avg_pred)
    else:
        predictions.append(np.nan)

df['Predicted_Value'] = predictions

df.to_csv('output_predictions.csv', index=False, encoding='utf-8-sig')


