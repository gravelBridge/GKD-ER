import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from .base import *
from ..utils.metrics import test_all_tasks

def train_epoch(model, optimizer, x, y, device, batch_size=256, epochs=1):
    """Basic training epoch."""
    criterion = nn.CrossEntropyLoss()
    model.train()
    for epoch in range(epochs):
        perm = np.random.permutation(len(y))
        x = x[perm]
        y = y[perm]
        for start in range(0, len(y), batch_size):
            end = start + batch_size
            xb = torch.tensor(x[start:end], device=device, dtype=torch.float)
            yb = torch.tensor(y[start:end], device=device, dtype=torch.long)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()

def run_naive(model, tasks_x_train, tasks_y_train, tasks_x_test, tasks_y_test, 
              device, epochs=5, batch_size=256):
    """Run naive fine-tuning."""
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    all_accs = []
    
    for t in range(len(tasks_x_train)):
        train_epoch(model, optimizer, tasks_x_train[t], tasks_y_train[t], 
                   device, batch_size, epochs)
        accs = test_all_tasks(model, tasks_x_test, tasks_y_test, device)
        all_accs.append(accs)
    
    return all_accs

def run_ewc(model, tasks_x_train, tasks_y_train, tasks_x_test, tasks_y_test,
            device, lambda_ewc=1000, epochs=5, batch_size=256):
    """Run EWC method."""
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    old_params_list = []
    fishers_list = []
    all_accs = []
    criterion = nn.CrossEntropyLoss()
    
    for t in range(len(tasks_x_train)):
        for epoch in range(epochs):
            perm = np.random.permutation(len(tasks_y_train[t]))
            x_c = tasks_x_train[t][perm]
            y_c = tasks_y_train[t][perm]
            
            for start in range(0, len(y_c), batch_size):
                end = start + batch_size
                xb = torch.tensor(x_c[start:end], device=device, dtype=torch.float)
                yb = torch.tensor(y_c[start:end], device=device, dtype=torch.long)
                
                optimizer.zero_grad()
                out = model(xb)
                ce_loss = criterion(out, yb)
                
                penalty = 0.0
                if len(old_params_list) > 0:
                    for i in range(len(old_params_list)):
                        penalty += ewc_loss(model, old_params_list[i], fishers_list[i], 
                                         lambda_ewc=lambda_ewc)
                    penalty = penalty / len(old_params_list)
                
                loss = ce_loss + penalty
                loss.backward()
                optimizer.step()
        
        old_params = save_params(model)
        fisher = compute_fisher(model, tasks_x_train[t], tasks_y_train[t], device)
        old_params_list.append(old_params)
        fishers_list.append(fisher)
        
        accs = test_all_tasks(model, tasks_x_test, tasks_y_test, device)
        all_accs.append(accs)
    
    return all_accs

def run_si(model, tasks_x_train, tasks_y_train, tasks_x_test, tasks_y_test,
           device, lambda_si=1.0, epochs=5, batch_size=256):
    """Run Synaptic Intelligence method."""
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    criterion = nn.CrossEntropyLoss()
    
    param_names = [n for n,_ in model.named_parameters()]
    w0 = {}
    w_importance = {}
    for n,p in model.named_parameters():
        w0[n] = p.data.clone()
        w_importance[n] = torch.zeros_like(p.data)
    
    all_accs = []
    
    def update_importance():
        for n,p in model.named_parameters():
            w_importance[n] += torch.abs(p.data - w0[n])
            w0[n] = p.data.clone()
    
    for t in range(len(tasks_x_train)):
        for epoch in range(epochs):
            perm = np.random.permutation(len(tasks_y_train[t]))
            x_c = tasks_x_train[t][perm]
            y_c = tasks_y_train[t][perm]
            
            for start in range(0, len(y_c), batch_size):
                end = start + batch_size
                xb = torch.tensor(x_c[start:end], device=device, dtype=torch.float)
                yb = torch.tensor(y_c[start:end], device=device, dtype=torch.long)
                
                optimizer.zero_grad()
                out = model(xb)
                ce_loss = criterion(out, yb)
                
                si_term = 0.0
                if t > 0:
                    for n,p in model.named_parameters():
                        si_term += (w_importance[n]*((p - w0[n])**2)).sum()
                    si_term = lambda_si * si_term
                
                loss = ce_loss + si_term
                loss.backward()
                optimizer.step()
        
        update_importance()
        accs = test_all_tasks(model, tasks_x_test, tasks_y_test, device)
        all_accs.append(accs)
    
    return all_accs

def run_er(model, tasks_x_train, tasks_y_train, tasks_x_test, tasks_y_test,
           device, memory_size=1000, epochs=5, batch_size=256):
    """Run Experience Replay method."""
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    criterion = nn.CrossEntropyLoss()
    buffer_x = []
    buffer_y = []
    all_accs = []
    
    for t in range(len(tasks_x_train)):
        reservoir_sampling(buffer_x, buffer_y, tasks_x_train[t], tasks_y_train[t], 
                         mem_size=memory_size)
        
        for epoch in range(epochs):
            perm = np.random.permutation(len(tasks_y_train[t]))
            x_c = tasks_x_train[t][perm]
            y_c = tasks_y_train[t][perm]
            
            for start in range(0, len(y_c), batch_size):
                end = start + batch_size
                xb = torch.tensor(x_c[start:end], device=device, dtype=torch.float)
                yb = torch.tensor(y_c[start:end], device=device, dtype=torch.long)
                
                if len(buffer_y) > 0:
                    idxs = np.random.choice(len(buffer_y), 
                                          min(len(buffer_y), batch_size), 
                                          replace=False)
                    xb_buf = torch.tensor(np.array([buffer_x[i] for i in idxs]), 
                                        device=device, dtype=torch.float)
                    yb_buf = torch.tensor(np.array([buffer_y[i] for i in idxs]), 
                                        device=device, dtype=torch.long)
                    xb = torch.cat([xb,xb_buf], dim=0)
                    yb = torch.cat([yb,yb_buf], dim=0)
                
                optimizer.zero_grad()
                out = model(xb)
                loss = criterion(out, yb)
                loss.backward()
                optimizer.step()
        
        accs = test_all_tasks(model, tasks_x_test, tasks_y_test, device)
        all_accs.append(accs)
    
    return all_accs

def run_gkder_full(model, tasks_x_train, tasks_y_train, tasks_x_test, tasks_y_test,
                   device, lambda_ewc=500, lambda_kd=1.0, mem_size=2000, 
                   var_threshold=0.9, epochs=5, batch_size=256):
    """Run GKD-ER-Full method."""
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    old_params_list = []
    fishers_list = []
    old_distill_data = []
    buffer_x = []
    buffer_y = []
    G_list = []
    
    criterion = nn.CrossEntropyLoss()
    all_accs = []
    
    for t in range(len(tasks_x_train)):
        model.set_active_task(t)
        
        for epoch in range(epochs):
            reservoir_sampling(buffer_x, buffer_y, tasks_x_train[t], tasks_y_train[t], 
                            mem_size=mem_size)
            x_cur = tasks_x_train[t]
            y_cur = tasks_y_train[t]
            perm = np.random.permutation(len(y_cur))
            x_c = x_cur[perm]
            y_c = y_cur[perm]
            
            steps = (len(y_c) + batch_size - 1) // batch_size
            
            for step in range(steps):
                start = step*batch_size
                end = start + batch_size
                xb = torch.tensor(x_c[start:end], device=device, dtype=torch.float)
                yb = torch.tensor(y_c[start:end], device=device, dtype=torch.long)
                
                if len(buffer_y) > 0:
                    idxs = np.random.choice(len(buffer_y), 
                                          min(len(buffer_y), batch_size), 
                                          replace=False)
                    xb_buf = torch.tensor(np.array([buffer_x[i] for i in idxs]), 
                                        device=device, dtype=torch.float)
                    yb_buf = torch.tensor(np.array([buffer_y[i] for i in idxs]), 
                                        device=device, dtype=torch.long)
                    xb = torch.cat([xb, xb_buf], dim=0)
                    yb = torch.cat([yb, yb_buf], dim=0)
                
                optimizer.zero_grad()
                out = model(xb)
                ce_loss = criterion(out, yb)
                
                ewc_term = 0.0
                if len(old_params_list) > 0:
                    for i in range(len(old_params_list)):
                        ewc_term += ewc_loss(model, old_params_list[i], fishers_list[i], 
                                          lambda_ewc=lambda_ewc)
                    ewc_term = ewc_term / len(old_params_list)
                
                kd_term = 0.0
                if len(old_distill_data) > 0:
                    kd_xb_all = []
                    kd_pb_all = []
                    for (xx,pp,yy) in old_distill_data:
                        kd_idxs = np.random.choice(len(pp), 
                                                 min(len(pp), batch_size), 
                                                 replace=False)
                        kd_xb_all.append(xx[kd_idxs])
                        kd_pb_all.append(pp[kd_idxs])
                    if len(kd_xb_all) > 0:
                        kd_xb = torch.tensor(np.concatenate(kd_xb_all), 
                                           device=device, dtype=torch.float)
                        kd_pb = np.concatenate(kd_pb_all)
                        out_kd = model(kd_xb)
                        kd_term = lambda_kd * knowledge_distillation_loss(out_kd, kd_pb, T=2.0)
                
                loss = ce_loss + ewc_term + kd_term
                loss.backward(retain_graph=True)
                
                if len(G_list) > 0:
                    project_gradients(model, G_list)
                
                optimizer.step()
        
        old_params = save_params(model)
        fisher = compute_fisher(model, tasks_x_train[t], tasks_y_train[t], device)
        old_params_list.append(old_params)
        fishers_list.append(fisher)
        
        dist_x, dist_p, dist_y = store_distillation_data(model, tasks_x_train[t], 
                                                        tasks_y_train[t], device, 
                                                        n_samples=500)
        old_distill_data.append((dist_x, dist_p, dist_y))
        
        new_dirs = compute_subspace_directions(model, tasks_x_train[t], tasks_y_train[t], 
                                             device, var_threshold=var_threshold)
        if new_dirs is not None:
            if len(G_list) == 0:
                G_list = [new_dirs]
            else:
                combined = torch.cat([G_list[0], new_dirs], dim=0)
                Q, R = torch.linalg.qr(combined.T)
                Q = Q.T
                G_list = [Q]
        
        model.set_active_task(None)
        accs = test_all_tasks(model, tasks_x_test, tasks_y_test, device)
        all_accs.append(accs)
    
    return all_accs
