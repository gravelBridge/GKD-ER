import os
import torch
import random
import numpy as np
from models import ResNetSmall, GKD_ER_Full_Model
from utils import (load_mnist, create_permuted_mnist, print_stats,
                  plot_results, plot_forgetting, plot_final_avg_acc,
                  plot_task_evolution, compute_confusion_matrix,
                  plot_confusion_matrix, save_accuracies_csv,
                  plot_accuracy_boxplot)
from methods import (run_naive, run_ewc, run_si, run_er, run_gkder_full)

# Global Configurations
TRAINING_TIME = 5  # epochs per task
MEM_SIZE = 2000
LAMBDA_EWC = 500
LAMBDA_KD = 1.0
BATCH_SIZE = 256
VAR_THRESHOLD = 0.9
NUM_TASKS = 5
NUM_CLASSES = 10

def set_seed(seed=42):
    """Set seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main():
    # Set device and seed
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed()
    
    print(f"Running methods with TRAINING_TIME = {TRAINING_TIME} epochs per task.")
    print(f"Using device: {device}")
    
    # Load and preprocess data
    x_train, y_train, x_test, y_test = load_mnist()
    tasks_x_train, tasks_y_train, tasks_x_test, tasks_y_test = create_permuted_mnist(
        x_train, y_train, x_test, y_test, NUM_TASKS
    )
    
    # Create output directories
    os.makedirs("results", exist_ok=True)
    os.makedirs("confusion_matrices", exist_ok=True)
    
    # Initialize models
    model_naive = ResNetSmall(output_dim=NUM_CLASSES).to(device)
    model_ewc = ResNetSmall(output_dim=NUM_CLASSES).to(device)
    model_si = ResNetSmall(output_dim=NUM_CLASSES).to(device)
    model_er = ResNetSmall(output_dim=NUM_CLASSES).to(device)
    model_gkder = GKD_ER_Full_Model(output_dim=NUM_CLASSES, num_tasks=NUM_TASKS).to(device)
    
    # Run experiments
    print("\nRunning Naive...")
    accs_naive = run_naive(model_naive, tasks_x_train, tasks_y_train,
                          tasks_x_test, tasks_y_test, device,
                          epochs=TRAINING_TIME, batch_size=BATCH_SIZE)
    
    print("\nRunning EWC...")
    accs_ewc = run_ewc(model_ewc, tasks_x_train, tasks_y_train,
                       tasks_x_test, tasks_y_test, device,
                       lambda_ewc=LAMBDA_EWC, epochs=TRAINING_TIME,
                       batch_size=BATCH_SIZE)
    
    print("\nRunning SI...")
    accs_si = run_si(model_si, tasks_x_train, tasks_y_train,
                     tasks_x_test, tasks_y_test, device,
                     lambda_si=0.1, epochs=TRAINING_TIME,
                     batch_size=BATCH_SIZE)
    
    print("\nRunning ER...")
    accs_er = run_er(model_er, tasks_x_train, tasks_y_train,
                     tasks_x_test, tasks_y_test, device,
                     memory_size=MEM_SIZE, epochs=TRAINING_TIME,
                     batch_size=BATCH_SIZE)
    
    print("\nRunning GKD-ER-Full...")
    accs_gkder = run_gkder_full(model_gkder, tasks_x_train, tasks_y_train,
                                tasks_x_test, tasks_y_test, device,
                                lambda_ewc=LAMBDA_EWC, lambda_kd=LAMBDA_KD,
                                mem_size=MEM_SIZE, var_threshold=VAR_THRESHOLD,
                                epochs=TRAINING_TIME, batch_size=BATCH_SIZE)
    
    # Collect results
    methods = ["Naive", "EWC", "SI", "ER", "GKD-ER-Full"]
    accs_list = [accs_naive, accs_ewc, accs_si, accs_er, accs_gkder]
    
    print("\nFinal Results:")
    for name, accs in zip(methods, accs_list):
        print_stats(name, accs)
    
    # Generate plots
    plot_results(accs_list, methods, filename="results/cl_results_comparison.png",
                title="CL Performance on Permuted MNIST")
    plot_forgetting(methods, accs_list, filename="results/forgetting_bar.png")
    plot_final_avg_acc(methods, accs_list, filename="results/final_avg_acc_bar.png")
    
    for t in range(NUM_TASKS):
        plot_task_evolution(accs_list, methods, task_id=t,
                          filename_prefix=f"results/task_evolution_task{t}")
    
    # Save accuracies
    for name, accs in zip(methods, accs_list):
        save_accuracies_csv(accs, f"results/{name.lower()}_accuracies.csv")
    
    # Generate confusion matrices for GKD-ER-Full
    for t in range(NUM_TASKS):
        cm = compute_confusion_matrix(model_gkder, tasks_x_test[t],
                                    tasks_y_test[t], device)
        plot_confusion_matrix(cm,
                            title=f"GKD-ER-Full Confusion Matrix for Task {t}",
                            filename=f"confusion_matrices/gkder_confusion_task_{t}.png")
    
    # Create final accuracy comparison
    final_accs_all = {}
    for name, accs in zip(methods, accs_list):
        final_accs_all[name] = accs[-1]
    
    plot_accuracy_boxplot(final_accs_all,
                         filename="results/final_task_accuracies_boxplot.png")

if __name__ == "__main__":
    main()
