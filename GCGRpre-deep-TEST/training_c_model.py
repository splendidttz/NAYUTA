import torch
from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve, accuracy_score,auc, RocCurveDisplay
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt


def train_and_validate(model, train_loader, val_loader, optimizer, criterion, epochs, model_path):
    best_val_loss = float('inf')  
    train_losses = []
    val_losses = []
    train_precisions = []
    val_precisions = []
    train_recalls = []
    val_recalls = []
    train_f1s = []
    val_f1s = []
    train_accuracies = []
    val_accuracies = []
    true_train_labels = []
    train_scores = []
    true_val_labels = []
    val_scores = []
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        train_preds = []
        train_targets = []

        for data in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output.squeeze(), data.y)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

            output = output.squeeze()
            train_preds.extend(output.tolist())
            train_targets.extend(data.y.tolist())

        train_losses.append(total_train_loss / len(train_loader))
        train_scores.extend(train_preds)
        true_train_labels.extend(train_targets)

        # Calculate precision, recall, F1 score, and accuracy for training set
        train_predictions = [1 if val > 0.5 else 0 for val in train_scores]
        train_precision = precision_score(true_train_labels, train_predictions)
        train_precisions.append(train_precision)
        train_recall = recall_score(true_train_labels, train_predictions)
        train_recalls.append(train_recall)
        train_f1 = f1_score(true_train_labels, train_predictions)
        train_f1s.append(train_f1)
        train_accuracy = accuracy_score(true_train_labels, train_predictions)
        train_accuracies.append(train_accuracy)

        model.eval()
        total_val_loss = 0
        val_preds = []
        val_targets = []

        with torch.no_grad():
            for data in val_loader:
                output = model(data)
                loss = criterion(output.squeeze(), data.y)
                total_val_loss += loss.item()

                output = output.squeeze()
                val_preds.extend(output.tolist())
                val_targets.extend(data.y.tolist())

        val_losses.append(total_val_loss / len(val_loader))
        val_scores.extend(val_preds)
        true_val_labels.extend(val_targets)

        # Calculate precision, recall, F1 score, and accuracy for validation set
        val_predictions = [1 if val > 0.5 else 0 for val in val_scores]
        val_precision = precision_score(true_val_labels, val_predictions)
        val_precisions.append(val_precision)
        val_recall = recall_score(true_val_labels, val_predictions)
        val_recalls.append(val_recall)
        val_f1 = f1_score(true_val_labels, val_predictions)
        val_f1s.append(val_f1)
        val_accuracy = accuracy_score(true_val_labels, val_predictions)
        val_accuracies.append(val_accuracy)
        val_auc = roc_auc_score(true_val_labels, val_scores)

        print(f'Epoch {epoch + 1}/{epochs} - Training loss: {train_losses[-1]:.4f} - Validation loss: {val_losses[-1]:.4f}')
        print(f'Training Precision: {train_precision:.4f} - Validation Precision: {val_precision:.4f}')
        print(f'Training Recall: {train_recall:.4f} - Validation Recall: {val_recall:.4f}')
        print(f'Training F1 Score: {train_f1:.4f} - Validation F1 Score: {val_f1:.4f}')
        print(f'Training Accuracy: {train_accuracy:.4f} - Validation Accuracy: {val_accuracy:.4f}')
        print(f'Validation AUC: {val_auc:.4f}')

        scheduler.step(total_val_loss / len(val_loader))

        if val_losses[-1] < best_val_loss:
            best_val_loss = val_losses[-1]

            # Save model parameters to file
            torch.save(model.state_dict(), model_path)

            print("Saved best model with validation loss:", best_val_loss)

        if (epoch + 1) % 10 == 0:
            plot_roc(true_train_labels, train_scores, true_val_labels, val_scores, epoch + 1)

    train_fpr, train_tpr, train_roc_auc = compute_roc(true_train_labels, train_scores)
    val_fpr, val_tpr, val_roc_auc = compute_roc(true_val_labels, val_scores)

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, epochs + 1), train_losses, label='Training Loss')
    plt.plot(range(1, epochs + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig('loss_curve.png')

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, epochs + 1), train_precisions, label='Training Precision')
    plt.plot(range(1, epochs + 1), val_precisions, label='Validation Precision')
    plt.xlabel('Epochs')
    plt.ylabel('Precision')
    plt.title('Training and Validation Precision Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig('precision_curve.png')

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, epochs + 1), train_recalls, label='Training Recall')
    plt.plot(range(1, epochs + 1), val_recalls, label='Validation Recall')
    plt.xlabel('Epochs')
    plt.ylabel('Recall')
    plt.title('Training and Validation Recall Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig('recall_curve.png')

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, epochs + 1), train_f1s, label='Training F1 Score')
    plt.plot(range(1, epochs + 1), val_f1s, label='Validation F1 Score')
    plt.xlabel('Epochs')
    plt.ylabel('F1 Score')
    plt.title('Training and Validation F1 Score Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig('f1_curve.png')

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, epochs + 1), train_accuracies, label='Training Accuracy')
    plt.plot(range(1, epochs + 1), val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig('accuracy_curve.png')

    return train_fpr, train_tpr, train_roc_auc, val_fpr, val_tpr, val_roc_auc

def compute_roc(true_labels, scores):
    fpr, tpr, _ = roc_curve(true_labels, scores)
    roc_auc = auc(fpr, tpr)
    return fpr, tpr, roc_auc

def plot_roc(true_train_labels, train_scores, true_val_labels, val_scores, epoch):
    train_fpr, train_tpr, train_roc_auc = compute_roc(true_train_labels, train_scores)
    val_fpr, val_tpr, val_roc_auc = compute_roc(true_val_labels, val_scores)

    plt.figure()
    plt.plot(train_fpr, train_tpr, color='darkorange', lw=2, label=f'Train ROC curve (area = {train_roc_auc:.2f})')
    plt.plot(val_fpr, val_tpr, color='darkgreen', lw=2, label=f'Validation ROC curve (area = {val_roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve (Epoch {epoch})')
    plt.legend(loc="lower right")
    plt.savefig(f'roc_curve_epoch_{epoch}.png', format='png')

