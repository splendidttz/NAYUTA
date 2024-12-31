import torch
from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve, accuracy_score,auc, RocCurveDisplay
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

def test(model, test_loader, model_path):
    model.load_state_dict(torch.load(model_path))
    model.eval()
    predictions = []
    true_labels = []
    scores = []

    with torch.no_grad():
        for data in test_loader:
            output = model(data)
            output = output.squeeze()
            pred = torch.round(output)
            predictions.extend(pred.tolist())
            true_labels.extend(data.y.tolist())
            scores.extend(output.tolist())

    precision = precision_score(true_labels, predictions)
    recall = recall_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions)
    accuracy = accuracy_score(true_labels, predictions)
    fpr, tpr, roc_auc = compute_roc(true_labels, scores)

    plt.figure()
    RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name='Test Model').plot()
    plt.savefig('test_roc_curve.png', format='png')

    print("True Labels:", true_labels)
    print("Predictions:", predictions)
    print("Prediction Probabilities:", scores)
    print("Confusion Matrix:\n", confusion_matrix(true_labels, predictions))

    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')
    print(f'Accuracy: {accuracy:.4f}')
    print(f'AUC: {roc_auc:.4f}')

    return fpr, tpr, roc_auc
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
