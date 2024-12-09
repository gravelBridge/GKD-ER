import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_results(accs_list, labels, filename="cl_results_comparison.png", title="Continual Learning Performance"):
    """Plot learning curves for all methods."""
    plt.figure(figsize=(8,6))
    tasks_range = range(1, len(accs_list[0][0])+1)
    for accs, label in zip(accs_list, labels):
        mean_accs = [np.mean(a) for a in accs]
        plt.plot(tasks_range, mean_accs, '-o', label=label)
    plt.xlabel("Task Number")
    plt.ylabel("Average Accuracy (%)")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.close()

def plot_forgetting(labels, accs_list, filename="forgetting_bar.png"):
    """Plot forgetting metric comparison."""
    forgets = []
    for accs in accs_list:
        fg = forgetting_metric(accs)
        forgets.append(fg)
    plt.figure(figsize=(8,6))
    x = np.arange(len(labels))
    plt.bar(x, forgets, color='salmon')
    plt.xticks(x, labels, rotation=45)
    plt.ylabel("Forgetting (%)")
    plt.title("Forgetting Metric Comparison")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def plot_final_avg_acc(labels, accs_list, filename="final_avg_acc_bar.png"):
    """Plot final average accuracy comparison."""
    final_accs = [final_avg_acc(a) for a in accs_list]
    plt.figure(figsize=(8,6))
    x = np.arange(len(labels))
    plt.bar(x, final_accs, color='skyblue')
    plt.xticks(x, labels, rotation=45)
    plt.ylabel("Final Average Accuracy (%)")
    plt.title("Final Average Accuracy Comparison")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def plot_task_evolution(accs_list, labels, task_id=0, filename_prefix="task_evolution"):
    """Plot evolution of accuracy on a specific task."""
    plt.figure(figsize=(8,6))
    tasks_range = range(1, len(accs_list[0])+1)
    for accs, label in zip(accs_list, labels):
        evol = [a[task_id] for a in accs]
        plt.plot(tasks_range, evol, '-o', label=label)
    plt.xlabel("Number of Tasks Learned")
    plt.ylabel(f"Accuracy on Task {task_id} (%)")
    plt.title(f"Accuracy Evolution on Task {task_id}")
    plt.legend()
    plt.grid(True)
    fname = f"{filename_prefix}_task{task_id}.png"
    plt.savefig(fname)
    plt.close()

def plot_accuracy_boxplot(final_accs_dict, filename="final_task_accuracies_boxplot.png"):
    """Plot boxplot of final accuracies."""
    df_final_acc = pd.DataFrame(final_accs_dict)
    df_melt = df_final_acc.drop("Average").melt(var_name="Method", value_name="Accuracy")
    plt.figure(figsize=(8,6))
    sns.boxplot(data=df_melt, x="Method", y="Accuracy")
    plt.title("Distribution of Final Task Accuracies per Method")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
