# Import required libraries
import numpy as np
import torch
from qml_modules import *
import os
from multiprocessing import Pool

def label_check(args):
    model, data, target = args
    data = data.view(data.size(0), -1)
    output = model(data)
    _, predicted = torch.max(output.data, 1)
    total = target.size(0)
    correct = (predicted == target).sum().item()
    
    return total, correct

def loss_eval(args):
    model, criterion, data, target, batch_size = args
    output = model(data)
    loss = criterion(output, target)
    loss = loss / batch_size * len(data)
    loss.backward()  # Accumulate gradients
    
    return loss.item()

#torch.autograd.set_detect_anomaly(True) 
# MPI-based parallel batch processing for CPU
def process_batch_cpu_multiprocessing(model, data, target, batch_size, batch_idx, minibatch_size, criterion):
    
    size = len(data)
    total_loss = 0
    nan_counts = 0
    
    num_threads = size // minibatch_size + 1
    
    mini_data_list = [data[i:i+minibatch_size] for i in range(0, size, minibatch_size)]
    mini_target_list = [target[i:i+minibatch_size] for i in range(0, size, minibatch_size)]
    data_target_pairs_list = list(zip(mini_data_list, mini_target_list))
    
    with Pool(num_threads) as p:
        for result in p.map(loss_eval, [(model, criterion, mini_data, mini_target, batch_size) for mini_data, mini_target in data_target_pairs_list]):
            #result = p.map(loss_eval, mini_data_list)
            #print(f'Batch {batch_idx}, processed {len(result)} samples', flush=True)
            total_loss += result
    
    total_loss *= batch_size/(size-nan_counts)
    
    return total_loss, nan_counts


# Modified evaluation and training code with per-sample processing
def process_batch(model, data, target, batch_size, batch_idx, minibatch_size, criterion):
    
    size = len(data)
    total_loss = 0
    nan_counts = 0
    # Process one sample at a time
    for i in range(0, size, minibatch_size):
        mini_data = data[i:i+minibatch_size]  # Keep batch dimension (shape [1, 16])
        mini_target = target[i:i+minibatch_size]
        #print(mini_target, flush=True)
        # Forward pass with gradient tracking
        output = model(mini_data)
        
        if torch.any(torch.isnan(output)):
            nan_counts += 1
            print(f'{batch_idx * batch_size + i + minibatch_size} nan encountered', flush=True)
            #with open(f'noisy_QNN_test/nan_image_{batch_idx * batch_size + i + minibatch_size}.pkl', 'wb') as file:
            #    pickle.dump(mini_data, file, protocol=pickle.HIGHEST_PROTOCOL)
                        
            del mini_data
            del mini_target
            del output
            continue
        
        loss = criterion(output, mini_target)
        
        #print(f'output: {output}', flush=True)
        print(f'{batch_idx * batch_size + i + minibatch_size} done', flush=True)
        
        # Accumulate gradients (scaled by 1/batch_size)
        loss = loss / batch_size * len(mini_data)  # Scale loss for gradient accumulation
        loss.backward()  # Gradients accumulate across samples
        total_loss += loss.item()
        
        #print(optimizer.param_groups[0]['params'][0].grad.data.cpu().numpy(), flush=True)
        #print(model.clamped_classical.weight.grad, flush=True)
        #for param in model.parameters():
        #    if param.shape == (3*layers, num_qubits):
        #        print(torch.mean(param.grad**2))
    
    total_loss *= batch_size/(size-nan_counts)
    
    return total_loss, nan_counts



def load_checkpoint(model, optimizer, device, filename='noisy_QNN_test/checkpoint.pth.tar'):
    # Note: Input model & optimizer should be pre-defined.  This routine only updates their states.
    if os.path.isfile(filename):
        print("=> loading checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        last_image_index = checkpoint['last_image_index']
        print("=> loaded checkpoint '{}'"
                  .format(filename))
    else:
        print("=> no checkpoint found at '{}'".format(filename))
        
        
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)

    return model.to(device), optimizer, last_image_index


def calc_amplitude_damping_params(decay_rate: float, relative_T: int|float, dt: float) -> tuple:
    """
    Calculate the amplitude damping parameters based on decay rate, temperature, and time step.
    
    Args:
        decay_rate (float): The decay rate of the amplitude damping at 0 K
        relative (float): The relative temperature = hw/2kT.
        dt (float): The time step.
    
    Returns:
        tuple: Parameters for amplitude damping.
    """
    p_damping = np.exp(relative_T)/(2*np.cosh(relative_T))
    
    Gamma = decay_rate/np.tanh(relative_T)
    gamma = 1 - np.exp(-Gamma * dt)
    
    return p_damping, gamma, Gamma * dt