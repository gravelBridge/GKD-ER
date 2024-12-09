import torch
import torch.nn as nn
import numpy as np

def save_params(model):
    """Save model parameters."""
    old_params = {}
    for p in model.parameters():
        old_params[p] = p.data.clone()
    return old_params

def compute_fisher(model, x, y, device, samples=2000, batch_size=256):
    """Compute Fisher Information Matrix."""
    model.eval()
    fisher = {}
    for p in model.parameters():
        fisher[p] = torch.zeros_like(p.data)
    
    criterion = nn.CrossEntropyLoss()
    indices = np.random.choice(len(y), min(samples, len(y)), replace=False)
    x_s = x[indices]
    y_s = y[indices]
    
    for start in range(0, len(y_s), batch_size):
        end = start + batch_size
        xb = torch.tensor(x_s[start:end], device=device, dtype=torch.float)
        yb = torch.tensor(y_s[start:end], device=device, dtype=torch.long)
        model.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        for p in model.parameters():
            if p.grad is not None:
                fisher[p] += p.grad.data**2
    
    for p in fisher:
        fisher[p] /= (len(y_s)/batch_size)
    return fisher

def ewc_loss(model, old_params, fisher, lambda_ewc=1000):
    """Compute EWC loss."""
    loss = 0.0
    for p in model.parameters():
        if p in old_params:
            loss += (fisher[p]*(p - old_params[p])**2).sum()
    return lambda_ewc*loss

def knowledge_distillation_loss(new_logits, old_probs, T=2.0):
    """Compute Knowledge Distillation loss."""
    criterion_kd = nn.KLDivLoss(reduction='batchmean')
    new_probs = nn.LogSoftmax(dim=1)(new_logits/T)
    soft_targets = torch.tensor(old_probs, device=new_logits.device, dtype=torch.float)
    kd_loss = criterion_kd(new_probs, soft_targets)*(T*T)
    return kd_loss

def store_distillation_data(model, x, y, device, n_samples=500):
    """Store data for knowledge distillation."""
    idx = np.random.choice(len(y), min(n_samples, len(y)), replace=False)
    xb = torch.tensor(x[idx], device=device, dtype=torch.float)
    model.eval()
    with torch.no_grad():
        logits = model(xb)
        soft_targets = nn.Softmax(dim=1)(logits)
    return xb.cpu().numpy(), soft_targets.cpu().numpy(), y[idx]

def get_param_grads(model):
    """Get flattened gradients of model parameters."""
    grads = []
    for p in model.parameters():
        if p.grad is not None:
            grads.append(p.grad.view(-1))
    if len(grads) > 0:
        return torch.cat(grads)
    else:
        return None

def compute_subspace_directions(model, x, y, device, batch_size=256, samples=2000, var_threshold=0.9):
    """Compute gradient subspace directions."""
    criterion = nn.CrossEntropyLoss()
    indices = np.random.choice(len(y), min(samples, len(y)), replace=False)
    x_s = x[indices]
    y_s = y[indices]
    grad_list = []
    model.eval()
    
    for start in range(0, len(y_s), batch_size):
        end = start + batch_size
        xb = torch.tensor(x_s[start:end], device=device, dtype=torch.float)
        yb = torch.tensor(y_s[start:end], device=device, dtype=torch.long)
        model.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        g = get_param_grads(model)
        if g is not None:
            grad_list.append(g)
    
    if len(grad_list) == 0:
        return None
        
    G = torch.stack(grad_list, dim=0)  # shape MxD
    U, S, Vt = torch.linalg.svd(G, full_matrices=False)
    var_expl = (S**2).cumsum(dim=0)/(S**2).sum()
    idxs = (var_expl <= var_threshold).sum() + 1
    idxs = min(idxs, Vt.size(0))
    directions = Vt[:idxs]  # top directions
    
    for i in range(directions.size(0)):
        directions[i] = directions[i]/(directions[i].norm() + 1e-12)
    
    return directions.to(device)

def project_gradients(model, G_list):
    """Project gradients onto orthogonal space of previous gradients."""
    if len(G_list) == 0:
        return
        
    G_mat = G_list[0]  # shape(k,D)
    param_list = [p for p in model.parameters() if p.grad is not None]
    current_grad = torch.cat([p.grad.view(-1) for p in param_list], dim=0)
    coeff = torch.matmul(G_mat, current_grad)  # shape(k)
    projected = (coeff.unsqueeze(1)*G_mat).sum(dim=0)  # D-dim vector
    corrected_grad = current_grad - projected
    
    idx = 0
    for p in param_list:
        sz = p.grad.numel()
        p.grad.data.copy_(corrected_grad[idx:idx+sz].view_as(p.grad))
        idx += sz
