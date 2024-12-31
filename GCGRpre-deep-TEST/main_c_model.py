from torch_geometric.loader import DataLoader
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from classification_gcn import GCNNet
from training_c_model import train_and_validate
from test_c_model import test
from datacreate_c_model import *

train_val_data = pd.read_csv('GCGR_C_train_validation.csv', dtype={'standard_ismiles': str})
train_val_smiles = train_val_data['standard_ismiles'].tolist()
train_val_labels = train_val_data['agonist1&antagonist0'].tolist()

test_data = pd.read_csv('GCGR_C_test.csv')
test_smiles = test_data['standard_ismiles'].tolist()
test_labels = test_data['agonist1&antagonist0'].tolist()

train_val_graph_data = smiles_to_graph_data(train_val_smiles, train_val_labels)
test_graph_data = smiles_to_graph_data(test_smiles, test_labels)

train_graph_data, val_graph_data = train_test_split(train_val_graph_data, test_size=0.2, random_state=42)

train_loader = DataLoader(train_graph_data, batch_size=64, shuffle=True)
val_loader = DataLoader(val_graph_data, batch_size=64, shuffle=False)
test_loader = DataLoader(test_graph_data, batch_size=64, shuffle=False)

if __name__ == '__main__':
    model = GCNNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)
    criterion = torch.nn.BCELoss()
    epochs = 200

    model_path = 'best_model.pth'
    train_fpr, train_tpr, train_roc_auc, val_fpr, val_tpr, val_roc_auc = train_and_validate(model, train_loader, val_loader, optimizer, criterion, epochs, model_path)
    test_fpr, test_tpr, test_roc_auc = test(model, test_loader, model_path)

    plt.figure()
    plt.plot(train_fpr, train_tpr, color='darkorange', lw=2, label=f'Train ROC curve (area = {train_roc_auc:.2f})')
    plt.plot(val_fpr, val_tpr, color='darkgreen', lw=2, label=f'Validation ROC curve (area = {val_roc_auc:.2f})')
    plt.plot(test_fpr, test_tpr, color='darkblue', lw=2, label=f'Test ROC curve (area = {test_roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig('all_roc_curves.png', format='png')
