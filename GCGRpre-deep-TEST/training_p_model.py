import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from sklearn.model_selection import KFold
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from prediction_gat import GATNet


def normalize_labels(labels, min_val, max_val):
    labels_array = np.array(labels)
    return (labels_array - min_val) / (max_val - min_val)

def denormalize_labels(labels, min_val, max_val):
    return labels * (max_val - min_val) + min_val

def train(model, train_loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for data in train_loader:
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output.squeeze(), data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

def test(model, test_loader, min_label, max_label, fold_num):
    model.eval()
    predictions = []
    true_labels = []

    with torch.no_grad():
        for data in test_loader:
            output = model(data)
            output = output.squeeze()
            predictions.extend(output.tolist())
            true_labels.extend(data.y.tolist())

    # Denormalize predictions and true labels
    denorm_predictions = denormalize_labels(np.array(predictions), min_label, max_label)
    denorm_true_labels = denormalize_labels(np.array(true_labels), min_label, max_label)

    # Compute regression metrics
    mse = mean_squared_error(denorm_true_labels, denorm_predictions)
    mae = mean_absolute_error(denorm_true_labels, denorm_predictions)
    r2 = r2_score(denorm_true_labels, denorm_predictions)

    # Plot True Values vs Predictions
    plt.figure()
    plt.scatter(denorm_true_labels, denorm_predictions, alpha=0.5)
    plt.xlabel("True Values")
    plt.ylabel("Predictions")
    plt.title(f"True Values vs Predictions - Fold {fold_num}")
    plt.savefig(f"fold_{fold_num}_plot.png")  # Save figure with fold number in filename

    return mse, mae, r2  # Return MSE as validation loss


def cross_validate(model, dataset, min_label, max_label, k=5, epochs=200):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    mse_list, mae_list, r2_list = [], [], []
    fold_train_losses = []

    for fold_num, (train_index, test_index) in enumerate(kf.split(dataset), 1):
        train_data = [dataset[i] for i in train_index]
        test_data = [dataset[i] for i in test_index]

        train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

        model = GATNet(num_features_xd=78, output_dim=128, dropout=0.2)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        fold_train_loss = []

        for epoch in range(epochs):
            train_loss = train(model, train_loader, optimizer, criterion)
            fold_train_loss.append(train_loss)
            print(f'Fold {fold_num}, Epoch {epoch + 1}/{epochs}, Training Loss: {train_loss:.4f}')

        fold_train_losses.append(fold_train_loss)

        torch.save(model.state_dict(), f'fold_{fold_num}_model.pth')

        model.load_state_dict(torch.load(f'fold_{fold_num}_model.pth'))


        mse, mae, r2 = test(model, test_loader, min_label, max_label, fold_num)

        print(f"Fold {fold_num} Test Results - MSE: {mse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
        mse_list.append(mse)
        mae_list.append(mae)
        r2_list.append(r2)

    mean_mse = np.mean(mse_list)
    std_mse = np.std(mse_list)
    mean_mae = np.mean(mae_list)
    std_mae = np.std(mae_list)
    mean_r2 = np.mean(r2_list)
    std_r2 = np.std(r2_list)

    print(f'Cross-Validation MSE: {mean_mse:.4f} ± {std_mse:.4f}')
    print(f'Cross-Validation MAE: {mean_mae:.4f} ± {std_mae:.4f}')
    print(f'Cross-Validation R²: {mean_r2:.4f} ± {std_r2:.4f}')

    print(f'Final Cross-Validation Results: MSE: {mean_mse:.4f}, MAE: {mean_mae:.4f}, R²: {mean_r2:.4f}')

    plt.figure(figsize=(10, 5))
    for fold_num, fold_train_loss in enumerate(fold_train_losses, 1):
        plt.plot(range(1, epochs + 1), fold_train_loss, label=f'Fold {fold_num} Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Epochs for Each Fold')
    plt.legend()
    plt.grid(True)
    plt.savefig('cross_validation_loss_curve.png')

    return mean_mse, std_mse, mean_mae, std_mae, mean_r2, std_r2
