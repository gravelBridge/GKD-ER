import numpy as np
import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def accuracy(model, x, y, device):
    """Compute accuracy for a model on given data."""
    model.eval()
    with torch.no_grad():
        xb = torch.tensor(x, device=device, dtype=torch.float)
        out = model(xb)
        pred = out.argmax(dim=1).cpu().numpy()
        acc = (pred == y).mean()*100
    return acc

def test_all_tasks(model, tasks_x_test, tasks_y_test, device):
    """Test model on all tasks."""
    accs = []
    for i in range(len(tasks_x_test)):
        accs.append(accuracy(model, tasks_x_test[i], tasks_y_test[i], device))
    return accs

def final_avg_acc(accs):
    """Compute final average accuracy."""
    return np.mean(accs[-1])

def forgetting_metric(accs):
    """Compute forgetting metric."""
    arr = np.array(accs)
    final_acc = arr[-1]
    max_acc = arr.max(axis=0)
    forgetting = np.mean(max_acc - final_acc)
    return forgetting

def forward_transfer(accs):
    """Compute forward transfer."""
    arr = np.array(accs)
    ft_values = [arr[j,j] for j in range(len(accs))]
    return np.mean(ft_values), ft_values

def backward_transfer(accs):
    """Compute backward transfer."""
    fg = forgetting_metric(accs)
    bt = -fg
    return bt

def print_stats(name, accs):
    """Print all metrics for a method."""
    fa = final_avg_acc(accs)
    fg = forgetting_metric(accs)
    ft_mean, ft_vals = forward_transfer(accs)
    bt_val = backward_transfer(accs)
    print(f"{name}:")
    print(f"  Final Avg Accuracy: {fa:.2f}%")
    print(f"  Forgetting: {fg:.2f}%")
    print(f"  Forward Transfer: {ft_mean:.2f}% (vals: {ft_vals})")
    print(f"  Backward Transfer: {bt_val:.2f}%")

def compute_confusion_matrix(model, x, y, device, num_classes=10):
    """Compute confusion matrix for a model on given data."""
    model.eval()
    with torch.no_grad():
        xb = torch.tensor(x, device=device, dtype=torch.float)
        out = model(xb)
        preds = out.argmax(dim=1).cpu().numpy()
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for i in range(len(y)):
        cm[y[i], preds[i]] += 1
    return cm

def save_accuracies_csv(accs, filename):
    """Save accuracies to CSV file."""
    df = pd.DataFrame(accs, columns=[f"Task {i}" for i in range(len(accs[0]))])
    df.loc["Average"] = df.mean()
    df.to_csv(filename)

def plot_confusion_matrix(cm, title, filename):
    """Plot and save confusion matrix."""
    df_cm = pd.DataFrame(cm, index=[i for i in range(cm.shape[0])],
                        columns=[i for i in range(cm.shape[1])])
    plt.figure(figsize=(8,6))
    sns.heatmap(df_cm, annot=True, fmt="d", cmap="Blues")
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig(filename)
    plt.close()
