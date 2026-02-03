from sklearn.metrics import roc_curve, auc
import numpy as np
import random
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

DNA_ALPHABET = ['A', 'C', 'G', 'T']

def random_substitution_top_kmers(
    model,
    seq,
    encoding_func,
    seq_len,
    window_size=6,
    top_k=5,
    device='cpu'
):
    model.eval()

    # original prediction
    x_orig = encoding_func(seq, seq_len)
    x_orig = torch.from_numpy(x_orig).unsqueeze(0).to(device)

    with torch.no_grad():
        base_prob = model(x_orig).item()

    drops = []

    for start in range(0, seq_len - window_size + 1):
        seq_list = list(seq)

        # randomize window
        for i in range(start, start + window_size):
            seq_list[i] = random.choice(DNA_ALPHABET)

        new_seq = "".join(seq_list)

        x_mut = encoding_func(new_seq, seq_len)
        x_mut = torch.from_numpy(x_mut).unsqueeze(0).to(device)

        with torch.no_grad():
            new_prob = model(x_mut).item()

        drop = base_prob - new_prob
        drops.append(drop)

    drops = np.array(drops)

    # get indices of most important windows
    top_idx = np.argsort(drops)[::-1][:top_k]

    important_kmers = []
    for i in top_idx:
        kmer = seq[i:i+window_size]
        important_kmers.append((i, kmer, drops[i]))

    return important_kmers


def calculate_metrics(true_labels, predictions, threshold):
    """
    Calculates and returns various classification metrics as a dictionary.

    Args:
        true_labels (np.array): Array of true binary labels.
        predictions (np.array): Array of target scores (e.g., probabilities) for the positive class.
        threshold (float): The threshold to binarize predictions.

    Returns:
        dict: A dictionary containing accuracy, precision, recall (sensitivity), specificity, and F1-score.
    """
    binary_predictions = (predictions >= threshold).astype(int)

    accuracy = accuracy_score(true_labels, binary_predictions)
    precision = precision_score(true_labels, binary_predictions)
    recall = recall_score(true_labels, binary_predictions)  # Also known as sensitivity
    f1 = f1_score(true_labels, binary_predictions)

    # Calculate specificity (negative recall)
    tn, fp, fn, tp = confusion_matrix(true_labels, binary_predictions).ravel()
    specificity = tn / (tn + fp)

    metrics = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall (Sensitivity)": recall,
        "Negative Recall (Specificity)": specificity,
        "F1-Score": f1
    }
    return metrics

def plot_confusion_matrix(true_labels, predictions, threshold=0.5, title='Confusion Matrix'):
    """
    Calculates and plots the confusion matrix.

    Args:
        true_labels (np.array): Array of true binary labels.
        predictions (np.array): Array of target scores (e.g., probabilities) for the positive class.
        threshold (float): The threshold to binarize predictions.
        title (str): Title for the confusion matrix plot.
    """
    binary_predictions = (predictions >= threshold).astype(int)
    cm = confusion_matrix(true_labels, binary_predictions)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Predicted Non-Promoter', 'Predicted Promoter'],
                yticklabels=['Actual Non-Promoter', 'Actual Promoter'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(title)
    plt.show()

def find_optimal_threshold(true_labels, predictions, thresholds):
    """
    Finds the optimal threshold that maximizes the F1-score.

    Args:
        true_labels (np.array): Array of true binary labels.
        predictions (np.array): Array of target scores (e.g., probabilities) for the positive class.
        thresholds (np.array): Array of thresholds from roc_curve.

    Returns:
        tuple: (optimal_threshold, max_f1_score)
    """
    optimal_threshold = 0.5 # Default starting point
    max_f1_score = -1

    # Iterate through the thresholds to find the one that maximizes F1-score
    # Exclude the first threshold which corresponds to a TPR of 1 and FPR of 1 (all samples classified as positive)
    for t in thresholds:
        binary_predictions_at_t = (predictions >= t).astype(int)
        f1 = f1_score(true_labels, binary_predictions_at_t)

        if f1 > max_f1_score:
            max_f1_score = f1
            optimal_threshold = t

    return optimal_threshold, max_f1_score

def plot_roc_curve(fpr, tpr, roc_auc, title="Receiver Operating Characteristic (ROC) Curve"):
    """
    Plots the ROC curve given FPR, TPR, and AUC.

    Args:
        fpr (np.array): Array of False Positive Rates.
        tpr (np.array): Array of True Positive Rates.
        roc_auc (float): Area Under the Curve.
        title (str): Title for the ROC plot.
    """
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.show()

def calculate_roc_data(model, test_loader, plot=False,device='cuda'):
    model.eval()  # Set the model to evaluation mode
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs).squeeze()

            # Ensure outputs are 1D before converting to numpy and extending
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(outputs.view(-1).cpu().numpy())

    # Convert lists to numpy arrays
    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions)

    # Compute ROC curve and AUC
    fpr, tpr, thresholds = roc_curve(all_labels, all_predictions)
    roc_auc = auc(fpr, tpr)
    
    if plot:
      plot_roc_curve(fpr,tpr,roc_auc)

    return fpr, tpr, roc_auc, all_labels, all_predictions


