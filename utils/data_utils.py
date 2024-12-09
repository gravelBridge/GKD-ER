import numpy as np
import torch
from torchvision.datasets import MNIST
import torchvision.transforms as transforms

def load_mnist():
    """Load and preprocess MNIST dataset."""
    transform = transforms.ToTensor()
    mnist_train = MNIST(root='./data', train=True, download=True, transform=transform)
    mnist_test = MNIST(root='./data', train=False, download=True, transform=transform)
    
    x_train = mnist_train.data.numpy().astype(np.float32)/255.0
    y_train = mnist_train.targets.numpy()
    x_test = mnist_test.data.numpy().astype(np.float32)/255.0
    y_test = mnist_test.targets.numpy()
    
    x_train = x_train.reshape(-1, 1, 28, 28)
    x_test = x_test.reshape(-1, 1, 28, 28)
    
    return x_train, y_train, x_test, y_test

def create_permuted_mnist(x_train, y_train, x_test, y_test, num_tasks, seed=42, seed_offset=0):
    """Create permuted MNIST tasks."""
    tasks_x_train = []
    tasks_y_train = []
    tasks_x_test = []
    tasks_y_test = []
    h, w = 28, 28
    
    for t in range(num_tasks):
        np.random.seed(t + seed + seed_offset)
        perm = np.arange(h*w)
        np.random.shuffle(perm)
        
        x_train_perm = x_train.copy().reshape(-1, h*w)
        x_train_perm = x_train_perm[:, perm].reshape(-1, 1, 28, 28)
        x_test_perm = x_test.copy().reshape(-1, h*w)
        x_test_perm = x_test_perm[:, perm].reshape(-1, 1, 28, 28)
        
        tasks_x_train.append(x_train_perm)
        tasks_y_train.append(y_train.copy())
        tasks_x_test.append(x_test_perm)
        tasks_y_test.append(y_test.copy())
    
    return tasks_x_train, tasks_y_train, tasks_x_test, tasks_y_test

def reservoir_sampling(buffer_x, buffer_y, new_x, new_y, mem_size=2000):
    """Implement reservoir sampling for experience replay."""
    for i in range(len(new_y)):
        if len(buffer_y) < mem_size:
            buffer_x.append(new_x[i])
            buffer_y.append(new_y[i])
        else:
            idx = np.random.randint(0, len(buffer_y))
            buffer_x[idx] = new_x[i]
            buffer_y[idx] = new_y[i]
