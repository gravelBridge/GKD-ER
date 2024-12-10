# GKD-ER: Gradient-space Knowledge Distillation with Episodic Replay for Mitigating Catastrophic Forgetting in Continual Learning

This repository contains an implementation of our paper "GKD-ER: Gradient-space Knowledge Distillation with Episodic Replay for Mitigating Catastrophic Forgetting in Continual Learning" and experiments with various continual learning methods for the Permuted MNIST task, including our novel GKD-ER (Gradient-space Knowledge Distillation with Episodic Replay) approach.

## Methods Implemented

1. Naive (Fine-tuning)
2. EWC (Elastic Weight Consolidation)
3. SI (Synaptic Intelligence)
4. ER (Experience Replay)
5. GKD-ER-Full (Our method)

## Project Structure

```
GKD-ER/
├── data/                  # Data directory for MNIST
├── models/               # Neural network model implementations
├── methods/              # Continual learning method implementations
├── utils/               # Utility functions and helpers
├── results/             # Results directory for metrics and plots
├── confusion_matrices/  # Confusion matrix plots
├── requirements.txt     # Project dependencies
└── main.py             # Main training script
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

To run the experiments:

```bash
python main.py
```

## Configuration

Key hyperparameters can be modified in `main.py`:

- `TRAINING_TIME`: Number of epochs per task
- `MEM_SIZE`: Memory buffer size for replay methods
- `LAMBDA_EWC`: EWC regularization strength
- `LAMBDA_KD`: Knowledge distillation loss weight
- `BATCH_SIZE`: Training batch size
- `VAR_THRESHOLD`: Variance threshold for gradient subspace
- `NUM_TASKS`: Number of tasks
- `NUM_CLASSES`: Number of classes per task

## Results

The code will generate:
- Learning curves comparing all methods
- Forgetting metrics visualization
- Final average accuracy comparisons
- Task evolution plots
- Confusion matrices
- Detailed CSV files with metrics

Results will be saved in the `results/` and `confusion_matrices/` directories.

## Citation

If you use this code in your research, please cite:

```bibtex
@article{gkd-er2024,
  title={GKD-ER: Gradient-space Knowledge Distillation with Episodic Replay for Mitigating Catastrophic Forgetting in Continual Learning},
  author={John Tian},
  journal={viXra preprint viXra:2412.0057},
  year={2024}
}
```

## License

MIT License
