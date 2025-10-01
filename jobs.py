# Import required libraries
import pennylane as qml
from pennylane import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision import datasets, transforms
from custom_datasets import CustomWaveDataset, CustomFFTInputs
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from qml_modules import HybridModel
from util_funcs import process_batch, process_minibatch, process_batch_cpu_multiprocessing, process_batch_cpu_mpi, label_check
from custom_loss_functions import SoftMaxSmoothedPQCLoss
import pickle
from multiprocessing import Pool
from mpi4py import MPI


def fourier_analysis_torch_parallel_job(rank: int, 
                                        size: int, 
                                        epoch_size: int,
                                        batch_size: int, 
                                        minibatch_size: int,
                                        test_size: int,
                                        num_qubits: int,
                                        p_depolarising_idx: int, 
                                        spectrum_size: int, 
                                        fft_density: int,
                                        noise_std: float,
                                        learning_rate: float) -> None:
    """
    Like depolarising_cpu_parallel_job, but uses torch.distributed for parallelisation within each batch.
    """
    dist.init_process_group(backend="nccl", rank=rank, world_size=size, init_method='env://')
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu", rank)
    torch.cuda.set_device(rank)
    #device = torch.device("cpu")
    
    if rank == 0:
        print(f"Using device: {device}", flush=True)
    
    #epoch_size = 100000
    #batch_size = 100
    #test_size = 500
    #num_qubits = 1
    #spectrum_size = 10
    
    transform = transforms.Compose([
        #transforms.Resize((32, 32)),
        #transforms.ToTensor(),
        transforms.Lambda(lambda x: torch.from_numpy(x).float()),
        #transforms.Lambda(lambda x: x.flatten()),
        #transforms.Lambda(lambda x: x / torch.norm(x)),
        transforms.Lambda(lambda x: x.to(device))
    ])
    
    if rank == 0:
        total_dataset = CustomWaveDataset(input_dim=num_qubits, spectrum_size=spectrum_size, num_samples_train=epoch_size, num_samples_test=test_size, noise_std=noise_std, transform=transform)
    dist.barrier(device_ids=[rank])
    
    total_dataset = CustomWaveDataset(input_dim=num_qubits, spectrum_size=spectrum_size, num_samples_train=epoch_size, num_samples_test=test_size, noise_std=noise_std, transform=transform)
    
    #if rank == 0:
    #    subset = Subset(total_dataset, indices=range(0,2000))
    #    loader = DataLoader(subset, batch_size=2000, shuffle=True)
    #    for _, (x, y) in enumerate(loader):
    #        training_data = {'data': torch.flatten(x).cpu().numpy(), 'target': torch.flatten(y).cpu().numpy()}
    #        with open(f'temp/training_function.pkl', 'wb') as file:
    #            pickle.dump(training_data, file, protocol=pickle.HIGHEST_PROTOCOL)
                
    #fft_inputs_dataset = CustomFFTInputs(input_dim=num_qubits, fft_density=fft_density, transform=transform)
    #test_dataset = CustomWaveDataset(input_dim=num_qubits, spectrum_size=spectrum_size, num_samples=test_size, noise_std=0, transform=transform)
    
    depolarising_rates = np.logspace(-5,np.log10(3/4),10)

    print(p_depolarising_idx, flush=True)
    
    if p_depolarising_idx == -1:
        p_depolarising = 0
    else:
        p_depolarising = depolarising_rates[p_depolarising_idx]
        
    if p_depolarising == 0:
        #try:
        #    # Try to use GPU simulator
        #    dev = qml.device("lightning.gpu", wires=num_qubits)
        #    print("Using lightning.gpu quantum simulator", flush=True)
        #except:
        #    #Fallback to CPU with GPU classical components
        dev = qml.device("default.qubit", wires=num_qubits)
        if rank == 0:
            print("Using default.qubit simulator - install lightning.gpu for GPU acceleration", flush=True)
    
    else:
        dev = qml.device("default.mixed", wires=num_qubits)
        if rank == 0:
            print("Using default.mixed simulator - install lightning.gpu for GPU acceleration", flush=True)
            
            
    layers = spectrum_size + 1
    weight_shapes = {"weights": (int(layers*3*2), num_qubits)}
    
    if rank == 0:
        print(f'depolarising noise rate: {p_depolarising}', flush=True)
    
    if p_depolarising == 0:
        model = HybridModel(dev=dev, 
                            device=device, 
                            num_qubits=num_qubits, 
                            weight_shapes=weight_shapes, 
                            noise_model=None, 
                            repeated_encoding=True).to(device)
    else:
        model = HybridModel(dev=dev, 
                            device=device, 
                            num_qubits=num_qubits, 
                            weight_shapes=weight_shapes, 
                            noise_model='depolarising',
                            p_depolarising=p_depolarising,
                            repeated_encoding=True).to(device)
        
    model = DDP(model, device_ids=[rank])
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    
    results = {}
    test_loss_values = {}
    training_samples = [0]
    loss_values = []
    gradients_list = []
    frequencies_list = []
    fitting_data_list = []
    trained_samples = 0

    
    #for batch_idx, (data, target) in enumerate(train_loader):
    for batch_idx in range(epoch_size//batch_size):
        
        train_subset = Subset(total_dataset, indices=range(batch_idx*batch_size, (batch_idx+1)*batch_size))
        train_sampler = DistributedSampler(train_subset, num_replicas=size, rank=rank)
        train_loader = DataLoader(train_subset, batch_size=minibatch_size, shuffle=False, sampler=train_sampler)
        
        total_loss = 0
        for minibatch_idx, (data, target) in enumerate(train_loader):
            model.train()
            data = data.to(device)
            target = target.to(device)

            optimizer.zero_grad()
            loss = process_minibatch(model, data, target, batch_size, minibatch_size, criterion)
            total_loss += loss

            trained_samples += len(data)*size
            if rank == 0:
                print(f'trained samples {trained_samples}', flush=True)
                
        total_loss = torch.tensor(total_loss, device=device)
        dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        total_loss = total_loss.item()
        
        optimizer.step()
        
        if rank == 0:
            print(f'Batch {batch_idx}, trained samples {trained_samples}, Total Loss: {total_loss:.4f}', flush=True)
            #gradients_linear_weights = optimizer.param_groups[0]['params'][0].grad.data.cpu().numpy()
            #gradients_linear_offsets = optimizer.param_groups[0]['params'][1].grad.data.cpu().numpy()
            #gradients_quantum = optimizer.param_groups[0]['params'][2].grad.data.cpu().numpy()
            #gradients_list.append({'linear weights': gradients_linear_weights,
            #                        'linear offsets': gradients_linear_offsets,
            #                        'quantum': gradients_quantum}
            #                        )
            gradients_quantum = optimizer.param_groups[0]['params'][0].grad.data.cpu().numpy()
            gradients_list.append({'quantum': gradients_quantum})
            
            loss_values.append(total_loss)
            training_samples.append(training_samples[-1] + batch_size)
            
        #homogeneous_inputs_generators = [torch.linspace(0, 2*np.pi, fft_density, device=device) for _ in range(num_qubits)]
        #homogeneous_inputs = torch.cartesian_prod(*homogeneous_inputs_generators)
        
        #model.eval()
        #with torch.no_grad():
        #    
        #    fft_inputs_sampler = DistributedSampler(fft_inputs_dataset, num_replicas=size, rank=rank)
        #    fft_inputs_loader = DataLoader(fft_inputs_dataset, batch_size=minibatch_size, shuffle=False, sampler=fft_inputs_sampler)
        #    
        #    current_rank_inputs = []
        #    current_rank_outputs = []
        #    for input_idx, input in enumerate(fft_inputs_loader):
        #        input = input.to(device)
        #        output = model(input)
        #        current_rank_inputs.append(input)
        #        current_rank_outputs.append(output)
        #    current_rank_inputs = torch.cat(current_rank_inputs)
        #    current_rank_outputs = torch.cat(current_rank_outputs)
        #    
        #    #current_rank_inputs = homogeneous_inputs[rank::size].to(device)
        #    #current_rank_outputs = model(current_rank_inputs)### need to be revised for single-value output instead of a prob vector
        #    
        #    total_inputs = [torch.empty_like(current_rank_inputs) for _ in range(size)]
        #    dist.all_gather(total_inputs, current_rank_inputs)
        #    total_inputs = torch.cat(total_inputs, dim=0)
#
        #    total_outputs = [torch.empty_like(current_rank_outputs) for _ in range(size)]
        #    dist.all_gather(total_outputs, current_rank_outputs)
        #    total_outputs = torch.cat(total_outputs, dim=0)
#
        #    if rank == 0:
        #        outputs_sorted = torch.zeros([fft_density*fft_inputs_dataset.sampling_overhead_multiplier]*num_qubits, device=device)
        #        for location, output in zip(total_inputs, total_outputs):
        #            indices_float = (location*(fft_density-1)/(2*np.pi)).tolist()
        #            indices_int = tuple([int(x) for x in indices_float])
        #            outputs_sorted[indices_int] = output.item()
        #        
        #        frequencies = torch.fft.rfftn(outputs_sorted)
        #        frequencies_list.append(frequencies.cpu().numpy())
        
        
        model.eval()
        with torch.no_grad():
            if rank == 0:
                coefficients = qml.fourier.coefficients(f = model.module.quantum_forward, 
                                                        n_inputs = num_qubits, 
                                                        degree = spectrum_size-1,
                                                        #lowpass_filter = True,
                                                        #filter_threshold = spectrum_size-1
                                                        )
                frequencies_list.append(coefficients)
        

        if batch_idx % 4 == 0:
            model.eval()
                        
            with torch.no_grad():
                test_subset = Subset(total_dataset, indices=range(epoch_size, epoch_size+test_size))
                
                test_MSE = 0
                fitting_data = []
                fitting_output = []
                
                test_sampler = DistributedSampler(test_subset, num_replicas=size, rank=rank)
                test_loader = DataLoader(test_subset, batch_size=minibatch_size, shuffle=False, sampler=test_sampler)
                for test_idx, (data_, target_) in enumerate(test_loader):
                    data_ = data_.to(device)
                    target_ = target_.to(device)
                    output = model(data_)
                    
                    #MSE = process_minibatch(model, data_, target_, test_size, minibatch_size, criterion, backpropagate=False)
                    MSE = criterion(output, target_)/batch_size*len(data_)
                    test_MSE += MSE
                    
                    fitting_data.append(data_)
                    fitting_output.append(output)
                    if rank == 0:
                        print(f'samples tested: {test_idx*minibatch_size*size}', flush=True)
                
                test_MSE = torch.tensor(test_MSE, device=device)
                dist.all_reduce(test_MSE, op=dist.ReduceOp.SUM)
                test_MSE = test_MSE.item()
                
                if rank == 0:
                    print(f'Test MSE: {test_MSE:.5f}\n', flush=True)
                    test_loss_values[trained_samples] = test_MSE
                
                fitting_data = torch.cat(fitting_data)
                fitting_output = torch.cat(fitting_output)
                
                total_fitting_data = [torch.empty_like(fitting_data) for _ in range(size)]
                dist.all_gather(total_fitting_data, fitting_data)
                total_fitting_data = torch.cat(total_fitting_data, dim=0)
                total_fitting_data = total_fitting_data.cpu().numpy()

                total_fitting_output = [torch.empty_like(fitting_output) for _ in range(size)]
                dist.all_gather(total_fitting_output, fitting_output)
                total_fitting_output = torch.cat(total_fitting_output, dim=0)
                total_fitting_output = total_fitting_output.cpu().numpy()
                
                if rank == 0:
                    fitting_data = {'data': total_fitting_data, 'target': total_fitting_output}
                    fitting_data_list.append(fitting_data)
                
                #if batch_idx == 996 and rank == 0:
                #    current_model_quantum_weights = list(model.module.quantum.parameters())[0].detach().cpu().numpy()
                #    print("Current model parameters shape:", current_model_quantum_weights.shape)
                #    current_model_quantum_weights = current_model_quantum_weights.reshape(layers, 6, num_qubits)
                #    print("Analytic fourier model parameters shape:", current_model_quantum_weights.shape)
                #
                #    encoding = partial(EncodingLayer, n_qubits=num_qubits)
                #    analytic_fourier_model = Model(
                #        n_qubits = num_qubits,
                #        n_layers = spectrum_size,
                #        circuit_type = VariationalLayer,
                #        data_reupload = True,  # Re-upload data at each layer
                #        encoding = encoding,
                #        output_qubit = 0, 
                #        #params = current_model_quantum_weights#, dtype=torch.float32, requires_grad=True)
                #        )
                #    analytic_fourier_model.params = torch.tensor(current_model_quantum_weights, dtype=torch.float32, requires_grad=True)
                #    print(analytic_fourier_model.params.shape)
                #    
                #    fourier_tree = FourierTree(analytic_fourier_model)
                #    an_coeffs, an_freqs = fourier_tree.get_spectrum(force_mean=True)
                #    print("Analytic Fourier Coefficients:", an_coeffs)
                #    print("Analytic Fourier Frequencies:", an_freqs)

    if rank == 0:
        print(f'trained samples: {trained_samples}', flush=True)
        results['training samples'] = training_samples[1:]
        results['test loss values'] = test_loss_values
        results['loss values'] = loss_values
        results['gradients'] = gradients_list
        results['frequencies'] = frequencies_list
        results['fitting data'] = fitting_data_list
        with open(f'noisy_QNN_test/repeated_angle_encoding_QVC{layers}_{num_qubits}q_{batch_size}batch_{epoch_size}epoch_{learning_rate}lr_depol{p_depolarising_idx}_smoothedPQC_gpu_torch.pkl', 'wb') as file:
            pickle.dump(results, file, protocol=pickle.HIGHEST_PROTOCOL)
    return

def depolarising_smoothedPQC_torch_parallel_job(rank: int, size: int, minibatch_size: int, p_depolarising: float) -> None:
    """
    Like depolarising_cpu_parallel_job, but uses torch.distributed for parallelisation within each batch.
    """
    
    dist.init_process_group(backend="nccl", rank=rank, world_size=size, init_method='env://')
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu", rank)
    torch.cuda.set_device(rank)
    #device = torch.device("cpu")
    
    if rank == 0:
        print(f"Using device: {device}", flush=True)
    
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.flatten()),
        transforms.Lambda(lambda x: x / torch.norm(x)),
        transforms.Lambda(lambda x: x.to(device))
    ])
    
    epoch_size = 15000
    batch_size = 50
    test_size = 250
    num_qubits = 10
    
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    #train_subset = Subset(train_dataset, indices=range(epoch_size))
    #train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    if p_depolarising == 0:
        #try:
        #    # Try to use GPU simulator
        #    dev = qml.device("lightning.gpu", wires=num_qubits)
        #    print("Using lightning.gpu quantum simulator", flush=True)
        #except:
        #    #Fallback to CPU with GPU classical components
        dev = qml.device("default.qubit", wires=num_qubits)
        if rank == 0:
            print("Using default.qubit simulator - install lightning.gpu for GPU acceleration", flush=True)
    
    else:
        dev = qml.device("default.mixed", wires=num_qubits)
        if rank == 0:
            print("Using default.mixed simulator - install lightning.gpu for GPU acceleration", flush=True)
            
            
    layers = 50
    weight_shapes = {"weights": (int(layers*3), num_qubits)}
    
    if rank == 0:
        print(f'depolarising noise rate: {p_depolarising}', flush=True)
    
    if p_depolarising == 0:
        model = HybridModel(dev=dev, device=device, num_qubits=num_qubits, weight_shapes=weight_shapes, noise_model=None).to(device)
    else:
        model = HybridModel(dev=dev, device=device, num_qubits=num_qubits, weight_shapes=weight_shapes, noise_model='depolarising',
                            p_depolarising=p_depolarising).to(device)
        
    model = DDP(model, device_ids=[rank])
    
    criterion = SoftMaxSmoothedPQCLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    
    
    results = {}
    accuracies = {}
    training_samples = [0]
    loss_values = []
    gradients_list = []
    trained_images = 0

    
    #for batch_idx, (data, target) in enumerate(train_loader):
    for batch_idx in range(epoch_size//batch_size):
        
        train_subset = Subset(train_dataset, indices=range(batch_idx*batch_size, (batch_idx+1)*batch_size))
        train_sampler = DistributedSampler(train_subset, num_replicas=size, rank=rank)
        train_loader = DataLoader(train_subset, batch_size=minibatch_size, shuffle=False, sampler=train_sampler)
        
        total_loss = 0
        for minibatch_idx, (data, target) in enumerate(train_loader):
            model.train()
            data = data.view(data.size(0), -1).to(device)
            target = target.to(device)
            target_new = torch.zeros(len(data), 10, device=device)
            for j in range(len(target)):
                target_new[j][target[j]] = 1
            optimizer.zero_grad()
            loss = process_minibatch(model, data, target_new, batch_size, minibatch_size, criterion)
            total_loss += loss

            trained_images += len(data)*size
            if rank == 0:
                print(f'trained images {trained_images}', flush=True)
                
        total_loss = torch.tensor(total_loss, device=device)
        dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        total_loss = total_loss.item()
        
        optimizer.step()
        
        if rank == 0:
            print(f'Batch {batch_idx}, trained images {trained_images}, Total Loss: {total_loss:.4f}', flush=True)
            #gradients_linear_weights = optimizer.param_groups[0]['params'][0].grad.data.cpu().numpy()
            #gradients_linear_offsets = optimizer.param_groups[0]['params'][1].grad.data.cpu().numpy()
            #gradients_quantum = optimizer.param_groups[0]['params'][2].grad.data.cpu().numpy()
            #gradients_list.append({'linear weights': gradients_linear_weights,
            #                        'linear offsets': gradients_linear_offsets,
            #                        'quantum': gradients_quantum}
            #                        )
            gradients_quantum = optimizer.param_groups[0]['params'][0].grad.data.cpu().numpy()
            gradients_list.append({'quantum': gradients_quantum})
            
            loss_values.append(total_loss)
            training_samples.append(training_samples[-1] + batch_size)
        
        # Evaluation
        if batch_idx % 4 == 0:
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                test_subset = Subset(test_dataset, indices=range(test_size))
                test_sampler = DistributedSampler(test_subset, num_replicas=size, rank=rank)
                test_loader = DataLoader(test_subset, batch_size=minibatch_size, shuffle=False, sampler=test_sampler)
                for test_idx, (data_, target_) in enumerate(test_loader):
                    data_ = data_.view(data_.size(0), -1).to(device)
                    target_ = target_.to(device)
                    
                    output = model(data_)
                    pred = output.argmax(dim=1)
                    correct += (pred == target_).sum().item()
                    total += len(target_)
                    if rank == 0:
                        print(f'images tested: {test_idx*minibatch_size*size}', flush=True)
            
            total = torch.tensor(total, device=device)
            dist.all_reduce(total, op=dist.ReduceOp.SUM)
            total = total.item()
            
            correct = torch.tensor(correct, device=device)
            dist.all_reduce(correct, op=dist.ReduceOp.SUM)
            correct = correct.item()
        
            if rank == 0:
                print(f'Test Accuracy: {100 * correct / total:.2f}%\n', flush=True)
                accuracies[trained_images] = correct / total

    if rank == 0:
        print(f'trained images: {trained_images}', flush=True)
        results['training samples'] = training_samples[1:]
        results['accuracies'] = accuracies
        results['loss values'] = loss_values
        results['gradients'] = gradients_list
        with open(f'noisy_QNN_test/QVC50_10q_encoded_50batch_15000epoch_0005lr_depol_{p_depolarising:.2e}_smoothedPQC_cpu_torch.pkl', 'wb') as file:
            pickle.dump(results, file, protocol=pickle.HIGHEST_PROTOCOL)
    return


def depolarising_smoothedPQC_cpu_mpi_parallel_job(p_depolarising: float) -> None:
    """
    Like depolarising_cpu_parallel_job, but uses MPI for parallelisation within each batch.
    """
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    if rank == 0:
        device = torch.device("cpu")
        print(f"Using device: {device}", flush=True)
        
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.flatten()),
            transforms.Lambda(lambda x: x / torch.norm(x)),
        ])
        
        epoch_size = 15000
        batch_size = 50
        test_size = 250
        minibatch_size = 1
        num_qubits = 10
        
        train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        train_subset = Subset(train_dataset, indices=range(epoch_size))
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

        dev = qml.device("default.mixed", wires=num_qubits)
        print("Using default.mixed simulator - install lightning.gpu for GPU acceleration", flush=True)
        
        layers = 50
        weight_shapes = {"weights": (int(layers*3), num_qubits)}
        
        print(f'depolarising noise rate: {p_depolarising}', flush=True)
        
        model = HybridModel(dev=dev, device=device, num_qubits=num_qubits, weight_shapes=weight_shapes, noise_model='depolarising',
                            p_depolarising=p_depolarising).to(device)
        
        criterion = SoftMaxSmoothedPQCLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.005)
        
        
        results = {}
        accuracies = {}
        training_samples = [0]
        loss_values = []
        gradients_list = []
        trained_images = 0

        train_loader = comm.bcast(train_loader, root=0)
        test_dataset = comm.bcast(test_dataset, root=0)
        
        batch_size = comm.bcast(batch_size, root=0)
        test_size = comm.bcast(test_size, root=0)
        minibatch_size = comm.bcast(minibatch_size, root=0)
        
        model = comm.bcast(model, root=0)
        criterion = comm.bcast(criterion, root=0)
        optimizer = comm.bcast(optimizer, root=0)
        
    comm.Barrier()
    
    for batch_idx, (data, target) in enumerate(train_loader):
        
        model.train()
        data = data.view(data.size(0), -1)
        target_new = torch.zeros(len(data),10)
        for j in range(len(target)):
            target_new[j][target[j]] = 1
            
        optimizer.zero_grad()
        
        loss = process_batch_cpu_mpi(model, data, target_new, batch_size, minibatch_size, criterion, rank, size)
        comm.Barrier()
        
        if rank == 0:
            local_grads = [torch.zeros_like(p) for p in model.parameters()]
            # Accumulate gradients across all ranks
            for idx, param in enumerate(model.parameters()):
                if param.grad is not None:
                    comm.Allreduce(param.grad, local_grads[idx], op = MPI.SUM)
                    param.grad.data = local_grads[idx]
                else:
                    param.grad.data = local_grads[idx]
            # Average total_loss and nan_counts across ranks
            loss = comm.allreduce(loss, op=MPI.SUM)
            
            optimizer.step()
            model = comm.bcast(model, root=0)
            loss = comm.bcast(loss, root=0)
            optimizer = comm.bcast(optimizer, root=0)
            
        
            trained_images += len(data)
            
            #gradients_linear_weights = optimizer.param_groups[0]['params'][0].grad.data.cpu().numpy()
            #gradients_linear_offsets = optimizer.param_groups[0]['params'][1].grad.data.cpu().numpy()
            #gradients_quantum = optimizer.param_groups[0]['params'][2].grad.data.cpu().numpy()
            #gradients_list.append({'linear weights': gradients_linear_weights,
            #                        'linear offsets': gradients_linear_offsets,
            #                        'quantum': gradients_quantum}
            #                        )
            gradients_quantum = optimizer.param_groups[0]['params'][0].grad.data.cpu().numpy()
            gradients_list.append({'quantum': gradients_quantum})
            
            loss_values.append(loss)
            training_samples.append(training_samples[-1] + len(data))
            print(f"trained images {trained_images}: Loss {loss:.4f}", flush=True)
        comm.Barrier()
        
        
        if batch_idx % 4 == 0:
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                test_subset = Subset(test_dataset, indices=range(test_size))
                test_loader = DataLoader(test_subset, batch_size=test_size, shuffle=False)
                indices = list(range(0, test_size, minibatch_size))
                current_rank_indicies = indices[rank::size]
                
                for data_, target_ in test_loader:
                    for i in current_rank_indicies:
                        mini_data = data_.view(minibatch_size, -1)[i:i+minibatch_size]
                        mini_target = target_[i:i+minibatch_size]
                        
                        output = model(mini_data)
                        pred = output.argmax(dim=1)
                        correct += (pred == mini_target).sum().item()
                        total += len(mini_data)
                comm.Barrier()
                
                if rank == 0:
                    correct = comm.allreduce(correct, op=MPI.SUM)
                    total = comm.allreduce(total, op=MPI.SUM)
                    print(f'Test Accuracy: {100 * correct / total:.2f}%\n', flush=True)
                    accuracies[trained_images] = correct / total
                comm.Barrier()

    if rank == 0:
        print(f'trained images: {trained_images}', flush=True)
        results['training samples'] = training_samples[1:]
        results['accuracies'] = accuracies
        results['loss values'] = loss_values
        results['gradients'] = gradients_list
        with open(f'noisy_QNN_test/QVC50_10q_encoded_50batch_15000epoch_0005lr_depol_{p_depolarising:.2e}_smoothedPQC_cpu_mpi.pkl', 'wb') as file:
            pickle.dump(results, file, protocol=pickle.HIGHEST_PROTOCOL)
    return


def depolarising_smoothedPQC_cpu_parallel_job(p_depolarising: float) -> None:
    """    
    function to run the depolarising noise job with quantum circuits.
    """
    
    # Check CUDA availability
    device = torch.device("cpu")
    print(f"Using device: {device}", flush=True)
    # Step 1: Data Preprocessing
    # --------------------------
    # Resize images to 4x4, convert to tensors, and scale pixel values to [0, π]
    transform = transforms.Compose([
        transforms.Resize((32, 32)),          # Downsample to 16x16 (256 pixels)
        transforms.ToTensor(),              # Convert to tensor
        #transforms.Lambda(lambda x: x * np.pi),  # Scale pixel values to [0, π]
        transforms.Lambda(lambda x: x.flatten()),  # Flatten to 16D vector
        transforms.Lambda(lambda x: x / torch.norm(x)), # Normalize to unit length
        #transforms.Lambda(lambda x: x.to(device)),
    ])

    # Load MNIST dataset
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    # Use smaller subsets for faster training (optional)
    epoch_size = 15000
    batch_size = 50
    test_size = 250
    minibatch_size = 1



    num_qubits = 10  # Using 9 qubits if using amplitude+phase encoding

    #try:
    #    # Try to use GPU simulator
    #    dev = qml.device("lightning.gpu", wires=num_qubits)
    #    print("Using lightning.gpu quantum simulator", flush=True)
    #except:
    # Fallback to CPU with GPU classical components
    #dev = qml.device("default.qubit", wires=num_qubits)
    #print("Using default.qubit simulator - install lightning.gpu for GPU acceleration", flush=True)
    dev = qml.device("default.mixed", wires=num_qubits)
    print("Using default.mixed simulator - install lightning.gpu for GPU acceleration", flush=True)

    layers = 50
    weight_shapes = {"weights": (int(layers*3), num_qubits)}



    # Step 5: Training Loop
    # ---------------------
    #depolarising_rates = np.logspace(-5,np.log10(3/4),10)

    #i = 0
    #print(i, flush=True)

    #p_depolarising = depolarising_rates[i]
    print(f'depolarising noise rate: {p_depolarising}', flush=True)
    #sigma = standard_deviations[i]
    #print(f'standard deviation: {sigma}', flush=True)

    results = {}
    
    model = HybridModel(dev=dev, device=device, num_qubits=num_qubits, weight_shapes=weight_shapes, noise_model='depolarising',
                        p_depolarising=p_depolarising).to(device)
    
    #model, optimizer, last_image_index = load_checkpoint(model=model, optimizer=optimizer, device=device, filename=f'noisy_QNN_test/QVC50_10q_encoded_50batch_15000epoch_0005lr_depol{i}_param_states_part1.pth.tar')
    criterion = SoftMaxSmoothedPQCLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005)

    accuracies = {}
    training_samples = [0]
    loss_values = []
    gradients_list = []
    nan_counts = []


    trained_images = 0
    last_image_index = -1
    finished_loop_counter = 0

    while trained_images < epoch_size:
        train_subset = Subset(train_dataset, indices=range(last_image_index+1, last_image_index+1 + epoch_size - trained_images))
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)#, pin_memory=True)#, pin_memory_device=device)
        for batch_idx, (data, target) in enumerate(train_loader):
                            
            model.train()
            # Flatten images: (batch_size, 1, 4, 4) -> (batch_size, 16)
            data = data.view(data.size(0), -1)#.to(device)
            target_new = torch.zeros(len(data),10).to(device)
            for j in range(len(target)):
                target_new[j][target[j]] = 1

            optimizer.zero_grad()

            loss, nan_count = process_batch_cpu_multiprocessing(model, data, target_new, batch_size, batch_idx, minibatch_size, criterion)
            trained_images += len(data) - nan_count
            nan_counts.append(nan_count)
            
            optimizer.step()

            #gradients_linear_weights = optimizer.param_groups[0]['params'][0].grad.data.cpu().numpy()
            #gradients_linear_offsets = optimizer.param_groups[0]['params'][1].grad.data.cpu().numpy()
            #gradients_quantum = optimizer.param_groups[0]['params'][2].grad.data.cpu().numpy()
            
            #gradients_list.append({'linear weights': gradients_linear_weights,
            #                        'linear offsets': gradients_linear_offsets,
            #                        'quantum': gradients_quantum}
            #                        )
            
            # if no classical layer
            gradients_quantum = optimizer.param_groups[0]['params'][0].grad.data.cpu().numpy()
            gradients_list.append({'quantum': gradients_quantum})

            loss_values.append(loss)
            training_samples.append(training_samples[-1]+len(data))
            last_image_index = training_samples[-1] - 1
            print(f"Finished Loop {finished_loop_counter}, last image index {last_image_index}, trained images {trained_images}: Loss {loss:.4f}", flush=True)
                
            # Step 6: Evaluation
            # ------------------
            if batch_idx % 4 == 0:
                model.eval()
                correct = 0
                total = 0
                last_test_idx = -1
                with torch.no_grad():
                    while total < test_size:
                        test_subset = Subset(test_dataset, indices=range(last_test_idx+1, last_test_idx+1 + test_size - total))
                        test_loader = DataLoader(test_subset, batch_size=minibatch_size, shuffle=False)#, pin_memory=True)#, pin_memory_device=device)
                        
                        data_target_pairs_list = [(data_, target_) for data_, target_ in test_loader]
                        num_threads = len(test_loader)
                        with Pool(num_threads) as p:
                            for batch_total, batch_correct in p.map(label_check, [(model, data_, target_) for data_, target_ in data_target_pairs_list]):
                                total += batch_total
                                correct += batch_correct

                #if batch_idx % 5 == 0:
                print(f'Test Accuracy: {100 * correct / total:.2f}%\n', flush=True)
                accuracies[trained_images] = correct/total
                torch.cuda.empty_cache()  # Clean up GPU memory

        finished_loop_counter += 1


    print(f'last image index: {last_image_index}', flush=True)
    results['nan_counts'] = nan_counts
    results['training samples'] = training_samples[1:]
    results['accuracies'] = accuracies
    results['loss values'] = loss_values
    results['gradients'] = gradients_list


    #state = {'state_dict': model.state_dict(), 
    #         'optimizer': optimizer.state_dict(),
    #         'last_image_index': last_image_index,}

    #torch.save(state, f'noisy_QNN_test/QVC50_10q_encoded_50batch_15000epoch_0005lr_depol{i}_param_states_part1.pth.tar')

    with open(f'noisy_QNN_test/QVC50_10q_encoded_50batch_15000epoch_0005lr_depol_{p_depolarising:.2e}_smoothedPQC_cpu_parallel.pkl', 'wb') as file:
        pickle.dump(results, file, protocol=pickle.HIGHEST_PROTOCOL)
        
    return


def depolarising_smoothedPQC_job(p_depolarising: float) -> None:
    """    
    function to run the depolarising noise job with quantum circuits.
    """
    
    # Check CUDA availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)
    # Step 1: Data Preprocessing
    # --------------------------
    # Resize images to 4x4, convert to tensors, and scale pixel values to [0, π]
    transform = transforms.Compose([
        transforms.Resize((32, 32)),          # Downsample to 16x16 (256 pixels)
        transforms.ToTensor(),              # Convert to tensor
        #transforms.Lambda(lambda x: x * np.pi),  # Scale pixel values to [0, π]
        transforms.Lambda(lambda x: x.flatten()),  # Flatten to 16D vector
        transforms.Lambda(lambda x: x / torch.norm(x)), # Normalize to unit length
        transforms.Lambda(lambda x: x.to(device)),
    ])

    # Load MNIST dataset
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    # Use smaller subsets for faster training (optional)
    epoch_size = 15000
    batch_size = 50
    test_size = 250
    minibatch_size = 1



    num_qubits = 10  # Using 9 qubits if using amplitude+phase encoding
    
    if p_depolarising == 0:
        #try:
        #    # Try to use GPU simulator
        #    dev = qml.device("lightning.gpu", wires=num_qubits)
        #    print("Using lightning.gpu quantum simulator", flush=True)
        #except:
        #    #Fallback to CPU with GPU classical components
        dev = qml.device("default.qubit", wires=num_qubits)
        print("Using default.qubit simulator - install lightning.gpu for GPU acceleration", flush=True)
    
    else:
        dev = qml.device("default.mixed", wires=num_qubits)
        print("Using default.mixed simulator - install lightning.gpu for GPU acceleration", flush=True)

    layers = 50
    weight_shapes = {"weights": (int(layers*3), num_qubits)}



    # Step 5: Training Loop
    # ---------------------
    #depolarising_rates = np.logspace(-5,np.log10(3/4),10)

    #i = 0
    #print(i, flush=True)

    #p_depolarising = depolarising_rates[i]
    print(f'depolarising noise rate: {p_depolarising}', flush=True)
    #sigma = standard_deviations[i]
    #print(f'standard deviation: {sigma}', flush=True)

    results = {}
    if p_depolarising == 0:
        model = HybridModel(dev=dev, device=device, num_qubits=num_qubits, weight_shapes=weight_shapes, noise_model=None).to(device)
    else:
        model = HybridModel(dev=dev, device=device, num_qubits=num_qubits, weight_shapes=weight_shapes, noise_model='depolarising',
                            p_depolarising=p_depolarising).to(device)
    
    #model, optimizer, last_image_index = load_checkpoint(model=model, optimizer=optimizer, device=device, filename=f'noisy_QNN_test/QVC50_10q_encoded_50batch_15000epoch_0005lr_depol{i}_param_states_part1.pth.tar')
    criterion = SoftMaxSmoothedPQCLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005)

    accuracies = {}
    training_samples = [0]
    loss_values = []
    gradients_list = []
    nan_counts = []


    trained_images = 0
    last_image_index = -1
    finished_loop_counter = 0

    while trained_images < epoch_size:
        train_subset = Subset(train_dataset, indices=range(last_image_index+1, last_image_index+1 + epoch_size - trained_images))
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)#, pin_memory=True)#, pin_memory_device=device)
        for batch_idx, (data, target) in enumerate(train_loader):
                            
            model.train()
            # Flatten images: (batch_size, 1, 4, 4) -> (batch_size, 16)
            data = data.view(data.size(0), -1)#.to(device)
            target_new = torch.zeros(len(data),10).to(device)
            for j in range(len(target)):
                target_new[j][target[j]] = 1

            optimizer.zero_grad()

            loss, nan_count = process_batch(model, data, target_new, batch_size, batch_idx, minibatch_size, criterion)
            trained_images += len(data) - nan_count
            nan_counts.append(nan_count)
            
            optimizer.step()

            #gradients_linear_weights = optimizer.param_groups[0]['params'][0].grad.data.cpu().numpy()
            #gradients_linear_offsets = optimizer.param_groups[0]['params'][1].grad.data.cpu().numpy()
            #gradients_quantum = optimizer.param_groups[0]['params'][2].grad.data.cpu().numpy()
            
            #gradients_list.append({'linear weights': gradients_linear_weights,
            #                        'linear offsets': gradients_linear_offsets,
            #                        'quantum': gradients_quantum}
            #                        )
            
            # if no classical layer
            gradients_quantum = optimizer.param_groups[0]['params'][0].grad.data.cpu().numpy()
            gradients_list.append({'quantum': gradients_quantum})

            loss_values.append(loss)
            training_samples.append(training_samples[-1]+len(data))
            last_image_index = training_samples[-1] - 1
            print(f"Finished Loop {finished_loop_counter}, last image index {last_image_index}, trained images {trained_images}: Loss {loss:.4f}", flush=True)
                
            # Step 6: Evaluation
            # ------------------
            if batch_idx % 4 == 0:
                model.eval()
                correct = 0
                total = 0
                last_test_idx = -1
                with torch.no_grad():
                    while total < test_size:
                        test_subset = Subset(test_dataset, indices=range(last_test_idx+1, last_test_idx+1 + test_size - total))
                        test_loader = DataLoader(test_subset, batch_size=minibatch_size, shuffle=False)#, pin_memory=True)#, pin_memory_device=device)
                        for idx, (data_, target_) in enumerate(test_loader):
                            last_test_idx += 1
                            print(f'test index: {last_test_idx}, successful: {total}', flush=True)
                            data_ = data_.view(data_.size(0), -1)#.to(device)
                            target_ = target_.to(device)

                            output = model(data_)
                            if torch.any(torch.isnan(output)):
                                print('test nan encountered', flush=True)
                                del output
                                continue
                                
                            
                            _, predicted = torch.max(output.data, 1)
                            total += target_.size(0)
                            correct += (predicted.to(device) == target_).sum().item()
                            print('test prcoessed', flush=True)
                #if batch_idx % 5 == 0:
                print(f'Test Accuracy: {100 * correct / total:.2f}%\n', flush=True)
                accuracies[trained_images] = correct/total
                torch.cuda.empty_cache()  # Clean up GPU memory

        finished_loop_counter += 1


    print(f'last image index: {last_image_index}', flush=True)
    results['nan_counts'] = nan_counts
    results['training samples'] = training_samples[1:]
    results['accuracies'] = accuracies
    results['loss values'] = loss_values
    results['gradients'] = gradients_list


    #state = {'state_dict': model.state_dict(), 
    #         'optimizer': optimizer.state_dict(),
    #         'last_image_index': last_image_index,}

    #torch.save(state, f'noisy_QNN_test/QVC50_10q_encoded_50batch_15000epoch_0005lr_depol{i}_param_states_part1.pth.tar')

    with open(f'noisy_QNN_test/QVC50_10q_encoded_50batch_15000epoch_0005lr_depol_{p_depolarising:.2e}_smoothedPQC.pkl', 'wb') as file:
        pickle.dump(results, file, protocol=pickle.HIGHEST_PROTOCOL)
        
    return


def depolarising_job(p_depolarising: float) -> None:
    """    
    function to run the depolarising noise job with quantum circuits.
    """
    
    # Check CUDA availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)
    # Step 1: Data Preprocessing
    # --------------------------
    # Resize images to 4x4, convert to tensors, and scale pixel values to [0, π]
    transform = transforms.Compose([
        transforms.Resize((32, 32)),          # Downsample to 16x16 (256 pixels)
        transforms.ToTensor(),              # Convert to tensor
        #transforms.Lambda(lambda x: x * np.pi),  # Scale pixel values to [0, π]
        transforms.Lambda(lambda x: x.flatten()),  # Flatten to 16D vector
        transforms.Lambda(lambda x: x / torch.norm(x)), # Normalize to unit length
        transforms.Lambda(lambda x: x.to(device)),
    ])

    # Load MNIST dataset
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    # Use smaller subsets for faster training (optional)
    epoch_size = 15000
    batch_size = 50
    test_size = 250
    minibatch_size = 1



    num_qubits = 10  # Using 9 qubits if using amplitude+phase encoding

    #try:
    #    # Try to use GPU simulator
    #    dev = qml.device("lightning.gpu", wires=num_qubits)
    #    print("Using lightning.gpu quantum simulator", flush=True)
    #except:
    # Fallback to CPU with GPU classical components
    #dev = qml.device("default.qubit", wires=num_qubits)
    #print("Using default.qubit simulator - install lightning.gpu for GPU acceleration", flush=True)
    dev = qml.device("default.mixed", wires=num_qubits)
    print("Using default.mixed simulator - install lightning.gpu for GPU acceleration", flush=True)

    layers = 75
    weight_shapes = {"weights": (int(layers*3), num_qubits)}



    # Step 5: Training Loop
    # ---------------------
    #depolarising_rates = np.logspace(-5,np.log10(3/4),10)

    #i = 0
    #print(i, flush=True)

    #p_depolarising = depolarising_rates[i]
    print(f'depolarising noise rate: {p_depolarising}', flush=True)
    #sigma = standard_deviations[i]
    #print(f'standard deviation: {sigma}', flush=True)

    results = {}
    
    model = HybridModel(dev=dev, device=device, num_qubits=num_qubits, weight_shapes=weight_shapes, noise_model='depolarising',
                        p_depolarising=p_depolarising).to(device)
    
    #model, optimizer, last_image_index = load_checkpoint(model=model, optimizer=optimizer, device=device, filename=f'noisy_QNN_test/QVC50_10q_encoded_50batch_15000epoch_0005lr_depol{i}_param_states_part1.pth.tar')
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005)

    accuracies = {}
    training_samples = [0]
    loss_values = []
    gradients_list = []
    nan_counts = []


    trained_images = 0
    last_image_index = -1
    finished_loop_counter = 0

    while trained_images < epoch_size:
        train_subset = Subset(train_dataset, indices=range(last_image_index+1, last_image_index+1 + epoch_size - trained_images))
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)#, pin_memory=True)#, pin_memory_device=device)
        for batch_idx, (data, target) in enumerate(train_loader):
                            
            model.train()
            # Flatten images: (batch_size, 1, 4, 4) -> (batch_size, 16)
            data = data.view(data.size(0), -1)#.to(device)
            target_new = torch.zeros(len(data),10).to(device)
            for j in range(len(target)):
                target_new[j][target[j]] = 1

            optimizer.zero_grad()

            loss, nan_count = process_batch(model, data, target_new, batch_size, batch_idx, minibatch_size, criterion)
            trained_images += len(data) - nan_count
            nan_counts.append(nan_count)
            
            optimizer.step()

            #gradients_linear_weights = optimizer.param_groups[0]['params'][0].grad.data.cpu().numpy()
            #gradients_linear_offsets = optimizer.param_groups[0]['params'][1].grad.data.cpu().numpy()
            #gradients_quantum = optimizer.param_groups[0]['params'][2].grad.data.cpu().numpy()
            
            #gradients_list.append({'linear weights': gradients_linear_weights,
            #                        'linear offsets': gradients_linear_offsets,
            #                        'quantum': gradients_quantum}
            #                        )
            
            # if no classical layer
            gradients_quantum = optimizer.param_groups[0]['params'][0].grad.data.cpu().numpy()
            gradients_list.append({'quantum': gradients_quantum})

            loss_values.append(loss)
            training_samples.append(training_samples[-1]+len(data))
            last_image_index = training_samples[-1] - 1
            print(f"Finished Loop {finished_loop_counter}, last image index {last_image_index}, trained images {trained_images}: Loss {loss:.4f}", flush=True)
                
            # Step 6: Evaluation
            # ------------------
            if batch_idx % 4 == 0:
                model.eval()
                correct = 0
                total = 0
                last_test_idx = -1
                with torch.no_grad():
                    while total < test_size:
                        test_subset = Subset(test_dataset, indices=range(last_test_idx+1, last_test_idx+1 + test_size - total))
                        test_loader = DataLoader(test_subset, batch_size=minibatch_size, shuffle=False)#, pin_memory=True)#, pin_memory_device=device)
                        for idx, (data_, target_) in enumerate(test_loader):
                            last_test_idx += 1
                            print(f'test index: {last_test_idx}, successful: {total}', flush=True)
                            data_ = data_.view(data_.size(0), -1)#.to(device)
                            target_ = target_.to(device)

                            output = model(data_)
                            if torch.any(torch.isnan(output)):
                                print('test nan encountered', flush=True)
                                del output
                                continue
                                
                            
                            _, predicted = torch.max(output.data, 1)
                            total += target_.size(0)
                            correct += (predicted.to(device) == target_).sum().item()
                            print('test prcoessed', flush=True)
                #if batch_idx % 5 == 0:
                print(f'Test Accuracy: {100 * correct / total:.2f}%\n', flush=True)
                accuracies[trained_images] = correct/total
                torch.cuda.empty_cache()  # Clean up GPU memory

        finished_loop_counter += 1


    print(f'last image index: {last_image_index}', flush=True)
    results['nan_counts'] = nan_counts
    results['training samples'] = training_samples[1:]
    results['accuracies'] = accuracies
    results['loss values'] = loss_values
    results['gradients'] = gradients_list


    #state = {'state_dict': model.state_dict(), 
    #         'optimizer': optimizer.state_dict(),
    #         'last_image_index': last_image_index,}

    #torch.save(state, f'noisy_QNN_test/QVC50_10q_encoded_50batch_15000epoch_0005lr_depol{i}_param_states_part1.pth.tar')

    with open(f'noisy_QNN_test/QVC75_10q_encoded_50batch_15000epoch_0005lr_depol_{p_depolarising:.2e}_no_classical.pkl', 'wb') as file:
        pickle.dump(results, file, protocol=pickle.HIGHEST_PROTOCOL)
        
    return



def two_qubit_noise_job(i: int) -> None:
    """
    function to run the two qubit noise job with quantum circuits.
    This job uses an extended depolarising noise model on two qubit gates.
    """

    # Check CUDA availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)
    # Step 1: Data Preprocessing
    # --------------------------
    # Resize images to 4x4, convert to tensors, and scale pixel values to [0, π]
    transform = transforms.Compose([
        transforms.Resize((32, 32)),          # Downsample to 16x16 (256 pixels)
        transforms.ToTensor(),              # Convert to tensor
        #transforms.Lambda(lambda x: x * np.pi),  # Scale pixel values to [0, π]
        transforms.Lambda(lambda x: x.flatten()),  # Flatten to 16D vector
        transforms.Lambda(lambda x: x / torch.norm(x)), # Normalize to unit length
        transforms.Lambda(lambda x: x.to(device)),
    ])

    # Load MNIST dataset
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    # Use smaller subsets for faster training (optional)
    epoch_size = 15000
    batch_size = 50
    test_size = 250
    minibatch_size = 1



    num_qubits = 10  # Using 9 qubits if using amplitude+phase encoding

    #try:
    #    # Try to use GPU simulator
    #    dev = qml.device("lightning.gpu", wires=num_qubits)
    #    print("Using lightning.gpu quantum simulator", flush=True)
    #except:
    # Fallback to CPU with GPU classical components
    #dev = qml.device("default.qubit", wires=num_qubits)
    #print("Using default.qubit simulator - install lightning.gpu for GPU acceleration", flush=True)
    dev = qml.device("default.mixed", wires=num_qubits)
    print("Using default.mixed simulator - install lightning.gpu for GPU acceleration", flush=True)

    layers = 75
    weight_shapes = {"weights": (int(layers*3), num_qubits)}



    # Step 5: Training Loop
    # ---------------------

    depolarising_rates = np.logspace(-5,np.log10(3/4),10)

    print(i, flush=True)
    
    if i == -1:
        p_depolarising = 0
    else:
        p_depolarising = depolarising_rates[i]
        
    print(f'depolarising noise rate: {p_depolarising}', flush=True)

    results = {}
    
    model = HybridModel(dev=dev, device=device, num_qubits=num_qubits, weight_shapes=weight_shapes, 
                        noise_model='extended depolarising on two qubit gates', 
                        p_depolarising = p_depolarising,
                        p_depolarising_two_qubit = 0.0019,
                        phi_IX = 0,
                        phi_IY = 0,
                        phi_IZ = 0,
                        phi_XI = 0,
                        phi_XX = 0,
                        phi_XY = 0,
                        phi_XZ = 0,
                        phi_YI = 0,
                        phi_YX = 0,
                        phi_YY = 0,
                        phi_YZ = 0,
                        phi_ZI = 0,
                        phi_ZX = 0,
                        phi_ZY = 0,
                        phi_ZZ = 0.00116).to(device)
    #model, optimizer, last_image_index = load_checkpoint(model=model, optimizer=optimizer, device=device, filename=f'noisy_QNN_test/QVC50_10q_encoded_50batch_15000epoch_0005lr_depol{i}_param_states_part1.pth.tar')
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005)

    accuracies = {}
    training_samples = [0]
    loss_values = []
    gradients_list = []
    nan_counts = []


    trained_images = 0
    last_image_index = -1
    finished_loop_counter = 0

    while trained_images < epoch_size:
        train_subset = Subset(train_dataset, indices=range(last_image_index+1, last_image_index+1 + epoch_size - trained_images))
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)#, pin_memory=True)#, pin_memory_device=device)
        for batch_idx, (data, target) in enumerate(train_loader):
                
            model.train()
            # Flatten images: (batch_size, 1, 4, 4) -> (batch_size, 16)
            data = data.view(data.size(0), -1)#.to(device)
            target_new = torch.zeros(len(data),10).to(device)
            for j in range(len(target)):
                target_new[j][target[j]] = 1

            optimizer.zero_grad()

            loss, nan_count = process_batch(model, data, target_new, batch_size, batch_idx, minibatch_size, criterion)
            trained_images += len(data) - nan_count
            nan_counts.append(nan_count)
            
            optimizer.step()

            #gradients_linear_weights = optimizer.param_groups[0]['params'][0].grad.data.cpu().numpy()
            #gradients_linear_offsets = optimizer.param_groups[0]['params'][1].grad.data.cpu().numpy()
            #gradients_quantum = optimizer.param_groups[0]['params'][2].grad.data.cpu().numpy()
            # if no classical layer
            gradients_quantum = optimizer.param_groups[0]['params'][0].grad.data.cpu().numpy()
            gradients_list.append({#'linear weights': gradients_linear_weights,
                                   # 'linear offsets': gradients_linear_offsets,
                                    'quantum': gradients_quantum}
                                    )
            #gradients_list.append(gradients)

            loss_values.append(loss)
            training_samples.append(training_samples[-1]+len(data))
            last_image_index = training_samples[-1] - 1
            print(f"Finished Loop {finished_loop_counter}, last image index {last_image_index}, trained images {trained_images}: Loss {loss:.4f}", flush=True)
                
            # Step 6: Evaluation
            # ------------------
            if batch_idx % 4 == 0:
                model.eval()
                correct = 0
                total = 0
                last_test_idx = -1
                with torch.no_grad():
                    while total < test_size:
                        test_subset = Subset(test_dataset, indices=range(last_test_idx+1, last_test_idx+1 + test_size - total))
                        test_loader = DataLoader(test_subset, batch_size=minibatch_size, shuffle=False)#, pin_memory=True)#, pin_memory_device=device)
                        for idx, (data_, target_) in enumerate(test_loader):
                            last_test_idx += 1
                            print(f'test index: {last_test_idx}, successful: {total}', flush=True)
                            data_ = data_.view(data_.size(0), -1)#.to(device)
                            target_ = target_.to(device)

                            output = model(data_)
                            if torch.any(torch.isnan(output)):
                                print('test nan encountered', flush=True)
                                del output
                                continue
                                
                            
                            _, predicted = torch.max(output.data, 1)
                            total += target_.size(0)
                            correct += (predicted.to(device) == target_).sum().item()
                            print('test prcoessed', flush=True)
                #if batch_idx % 5 == 0:
                print(f'Test Accuracy: {100 * correct / total:.2f}%\n', flush=True)
                accuracies[trained_images] = correct/total
                torch.cuda.empty_cache()  # Clean up GPU memory

        finished_loop_counter += 1
        

    print(f'last image index: {last_image_index}', flush=True)
    results['nan_counts'] = nan_counts
    results['training samples'] = training_samples[1:]
    results['accuracies'] = accuracies
    results['loss values'] = loss_values
    results['gradients'] = gradients_list


    #state = {'state_dict': model.state_dict(), 
    #         'optimizer': optimizer.state_dict(),
    #         'last_image_index': last_image_index,}

    #torch.save(state, f'noisy_QNN_test/QVC50_10q_encoded_50batch_15000epoch_0005lr_extended_depol{i}_param_states_part1.pth.tar')

    with open(f'noisy_QNN_test/QVC75_10q_encoded_50batch_15000epoch_0005lr_extended_depol{i}_no_classical.pkl', 'wb') as file:
        pickle.dump(results, file, protocol=pickle.HIGHEST_PROTOCOL)
        
    return


def gaussian_job(sigma: float, mu:float = 0) -> None:

    # Check CUDA availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)
    # Step 1: Data Preprocessing
    # --------------------------
    # Resize images to 4x4, convert to tensors, and scale pixel values to [0, π]
    transform = transforms.Compose([
        transforms.Resize((32, 32)),          # Downsample to 16x16 (256 pixels)
        transforms.ToTensor(),              # Convert to tensor
        #transforms.Lambda(lambda x: x * np.pi),  # Scale pixel values to [0, π]
        transforms.Lambda(lambda x: x.flatten()),  # Flatten to 16D vector
        transforms.Lambda(lambda x: x / torch.norm(x)), # Normalize to unit length
        transforms.Lambda(lambda x: x.to(device)),
    ])

    # Load MNIST dataset
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    # Use smaller subsets for faster training (optional)
    epoch_size = 15000
    batch_size = 50
    test_size = 250
    minibatch_size = 1



    num_qubits = 10  # Using 9 qubits if using amplitude+phase encoding

    #try:
    #    # Try to use GPU simulator
    #    dev = qml.device("lightning.gpu", wires=num_qubits)
    #    print("Using lightning.gpu quantum simulator", flush=True)
    #except:
    # Fallback to CPU with GPU classical components
    #dev = qml.device("default.qubit", wires=num_qubits)
    #print("Using default.qubit simulator - install lightning.gpu for GPU acceleration", flush=True)
    dev = qml.device("default.mixed", wires=num_qubits)
    print("Using default.mixed simulator - install lightning.gpu for GPU acceleration", flush=True)

    layers = 50
    weight_shapes = {"weights": (int(layers*3), num_qubits)}


    # Step 5: Training Loop
    # ---------------------
    print(f'standard deviation: {sigma}', flush=True)

    results = {}
    
    model = HybridModel(dev=dev, device=device, num_qubits=num_qubits, weight_shapes=weight_shapes, 
                        noise_model='gaussian over rotation', 
                        sigma=sigma,
                        mean=mu,
                        num_channels=9).to(device)
    #model, optimizer, last_image_index = load_checkpoint(model=model, optimizer=optimizer, device=device, filename=f'noisy_QNN_test/QVC50_10q_encoded_50batch_15000epoch_0005lr_depol{i}_param_states_part1.pth.tar')
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005)

    accuracies = {}
    training_samples = [0]
    loss_values = []
    gradients_list = []
    nan_counts = []


    trained_images = 0
    last_image_index = -1
    finished_loop_counter = 0

    while trained_images < epoch_size:
        train_subset = Subset(train_dataset, indices=range(last_image_index+1, last_image_index+1 + epoch_size - trained_images))
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)#, pin_memory=True)#, pin_memory_device=device)
        for batch_idx, (data, target) in enumerate(train_loader):

            model.train()
            # Flatten images: (batch_size, 1, 4, 4) -> (batch_size, 16)
            data = data.view(data.size(0), -1)#.to(device)
            target_new = torch.zeros(len(data),10).to(device)
            for j in range(len(target)):
                target_new[j][target[j]] = 1

            optimizer.zero_grad()

            loss, nan_count = process_batch(model, data, target_new, batch_size, batch_idx, minibatch_size, criterion)
            trained_images += len(data) - nan_count
            nan_counts.append(nan_count)
            
            optimizer.step()

            #gradients_linear_weights = optimizer.param_groups[0]['params'][0].grad.data.cpu().numpy()
            #gradients_linear_offsets = optimizer.param_groups[0]['params'][1].grad.data.cpu().numpy()
            #gradients_quantum = optimizer.param_groups[0]['params'][2].grad.data.cpu().numpy()
            # if no classical layer
            gradients_quantum = optimizer.param_groups[0]['params'][0].grad.data.cpu().numpy()
            gradients_list.append({#'linear weights': gradients_linear_weights,
                                   # 'linear offsets': gradients_linear_offsets,
                                    'quantum': gradients_quantum}
                                    )
            #gradients_list.append(gradients)

            loss_values.append(loss)
            training_samples.append(training_samples[-1]+len(data))
            last_image_index = training_samples[-1] - 1

            print(f"Finished Loop {finished_loop_counter}, last image index {last_image_index}, trained images {trained_images}: Loss {loss:.4f}", flush=True)

            # Step 6: Evaluation
            # ------------------
            if batch_idx % 4 == 0:
                model.eval()
                correct = 0
                total = 0
                last_test_idx = -1
                with torch.no_grad():
                    while total < test_size:
                        test_subset = Subset(test_dataset, indices=range(last_test_idx+1, last_test_idx+1 + test_size - total))
                        test_loader = DataLoader(test_subset, batch_size=minibatch_size, shuffle=False)#, pin_memory=True)#, pin_memory_device=device)
                        for idx, (data_, target_) in enumerate(test_loader):
                            last_test_idx += 1
                            print(f'test index: {last_test_idx}, successful: {total}', flush=True)
                            data_ = data_.view(data_.size(0), -1)#.to(device)
                            target_ = target_.to(device)

                            output = model(data_)
                            if torch.any(torch.isnan(output)):
                                print('test nan encountered', flush=True)
                                del output
                                continue
                                
                            
                            _, predicted = torch.max(output.data, 1)
                            total += target_.size(0)
                            correct += (predicted.to(device) == target_).sum().item()
                            print('test prcoessed', flush=True)
                #if batch_idx % 5 == 0:
                print(f'Test Accuracy: {100 * correct / total:.2f}%\n', flush=True)
                accuracies[trained_images] = correct/total
                torch.cuda.empty_cache()  # Clean up GPU memory

        finished_loop_counter += 1

    print(f'last image index: {last_image_index}', flush=True)
    results['nan_counts'] = nan_counts
    results['training samples'] = training_samples[1:]
    results['accuracies'] = accuracies
    results['loss values'] = loss_values
    results['gradients'] = gradients_list


    #state = {'state_dict': model.state_dict(), 
    #         'optimizer': optimizer.state_dict(),
    #         'last_image_index': last_image_index,}

    #torch.save(state, f'noisy_QNN_test/QVC50_10q_encoded_50batch_15000epoch_0005lr_depol{i}_param_states_part1.pth.tar')

    with open(f'noisy_QNN_test/QVC50_10q_encoded_50batch_15000epoch_0005lr_gaussian_pi2mean{sigma}_no_classical.pkl', 'wb') as file:
        pickle.dump(results, file, protocol=pickle.HIGHEST_PROTOCOL)
    
    return



def depol_and_damping_job(i_ : int, i : int = 4) -> None:
    """
    function to run the depolarising and damping noise job with quantum circuits.
    Note classical layer is not used in this job.
    """
    
    # Check CUDA availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)
    # Step 1: Data Preprocessing
    # --------------------------
    # Resize images to 4x4, convert to tensors, and scale pixel values to [0, π]
    transform = transforms.Compose([
        transforms.Resize((32, 32)),          # Downsample to 16x16 (256 pixels)
        transforms.ToTensor(),              # Convert to tensor
        #transforms.Lambda(lambda x: x * np.pi),  # Scale pixel values to [0, π]
        transforms.Lambda(lambda x: x.flatten()),  # Flatten to 16D vector
        transforms.Lambda(lambda x: x / torch.norm(x)), # Normalize to unit length
        transforms.Lambda(lambda x: x.to(device)),
    ])

    # Load MNIST dataset
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    # Use smaller subsets for faster training (optional)
    epoch_size = 15000
    batch_size = 50
    test_size = 250
    minibatch_size = 1



    num_qubits = 10  # Using 9 qubits if using amplitude+phase encoding

    #try:
    #    # Try to use GPU simulator
    #    dev = qml.device("lightning.gpu", wires=num_qubits)
    #    print("Using lightning.gpu quantum simulator", flush=True)
    #except:
    # Fallback to CPU with GPU classical components
    #dev = qml.device("default.qubit", wires=num_qubits)
    #print("Using default.qubit simulator - install lightning.gpu for GPU acceleration", flush=True)
    dev = qml.device("default.mixed", wires=num_qubits)
    print("Using default.mixed simulator - install lightning.gpu for GPU acceleration", flush=True)
    #dev = qml.device("default.qubit", wires=num_qubits)
    #dev = qml.device("default.mixed", wires=num_qubits)
    #dev = qml.device("lightning.qubit", wires=num_qubits)

    layers = 50
    weight_shapes = {"weights": (int(layers*3), num_qubits)}



    # Step 5: Training Loop
    # ---------------------

    depolarising_rates = np.logspace(-5,np.log10(3/4),10)
    print(i, flush=True)
    p_depolarising = depolarising_rates[i]
    print(f'depolarising noise rate: {p_depolarising}', flush=True)

    damping_rates = np.logspace(-5,np.log10(3/4),10)
    print(i_, flush=True)
    p_damping = damping_rates[i_]
    print(f'damping noise rate: {p_damping}', flush=True)

    results = {}
    #model = HybridModel(dev=dev, num_qubits=num_qubits, weight_shapes=weight_shapes, noise_model='depolarising', p_depolarising=p_depolarising)
    #model = HybridModel(dev=dev, num_qubits=num_qubits, weight_shapes=weight_shapes, noise_model='thermal', t1=0.001,t2=0.005,tg=0.003)
    #model = HybridModel(dev=dev, num_qubits=num_qubits, weight_shapes=weight_shapes, noise_model='pauli noise',
    #           pX_single=0.001, pY_single=0.001, pZ_single=0.001, pX_double=0.01, pY_double=0.01, pZ_double=0.01).to(device)
    #model = HybridModel(dev=dev, device=device, num_qubits=num_qubits, weight_shapes=weight_shapes, noise_model='depolarising',
    #                    p_depolarising=p_depolarising).to(device)

    #model = HybridModel(dev=dev, device=device, num_qubits=num_qubits, weight_shapes=weight_shapes, 
    #                    noise_model='gaussian over rotation', 
    #                    sigma=sigma,
    #                    mean=mean,
    #                    num_channels=9).to(device)
    model = HybridModel(dev=dev, device=device, num_qubits=num_qubits, weight_shapes=weight_shapes, 
                        noise_model='depol and damping',
                        p_depolarising = p_depolarising,
                        p_damping = p_damping).to(device)
                        
    #criterion = FullCriterion()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=1/np.e, patience=50)

    accuracies = {}
    training_samples = [0]
    loss_values = []
    gradients_list = []
    nan_counts = []


    trained_images = 0
    last_image_index = -1
    finished_loop_counter = 0

    while trained_images < epoch_size:
        train_subset = Subset(train_dataset, indices=range(last_image_index+1, last_image_index+1 + epoch_size - trained_images))
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)#, pin_memory=True)#, pin_memory_device=device)
        for batch_idx, (data, target) in enumerate(train_loader):
            #if batch_idx == 150:# or batch_idx == 200:
            #    inp=input('job Continue? (y/n): ')
            #    if inp=='y':
            #        pass
            #print(scheduler.get_last_lr())
                
            model.train()
            # Flatten images: (batch_size, 1, 4, 4) -> (batch_size, 16)
            data = data.view(data.size(0), -1)#.to(device)
            target_new = torch.zeros(len(data),10).to(device)
            for j in range(len(target)):
                target_new[j][target[j]] = 1

            optimizer.zero_grad()

            #output = model(data)
            #loss = criterion(output, target_new)
            #loss.backward()
            loss, nan_count = process_batch(model, data, target_new, batch_size, batch_idx, minibatch_size, criterion)
            trained_images += len(data) - nan_count
            nan_counts.append(nan_count)
            
            optimizer.step()

            #gradients_linear_weights = optimizer.param_groups[0]['params'][0].grad.data.cpu().numpy()
            #gradients_linear_offsets = optimizer.param_groups[0]['params'][1].grad.data.cpu().numpy()
            #gradients_quantum = optimizer.param_groups[0]['params'][2].grad.data.cpu().numpy()
            #gradients_list.append({'linear weights': gradients_linear_weights,
            #                        'linear offsets': gradients_linear_offsets,
            #                        'quantum': gradients_quantum}
            #                        )
            gradients_quantum = optimizer.param_groups[0]['params'][0].grad.data.cpu().numpy()
            gradients_list.append({'quantum': gradients_quantum})

            #loss_values.append(loss.item())
            loss_values.append(loss)
            training_samples.append(training_samples[-1]+len(data))
            last_image_index = training_samples[-1] - 1
            #if batch_idx % 5 == 0:
            #print(f"Epoch {epoch}, Batch {batch_idx}: Loss {loss.item():.4f}, LR {scheduler.get_last_lr()[0]}, | GPU Mem: {torch.cuda.memory_allocated()/1e6:.2f}MB")
            #print(f"Epoch {epoch}, Batch {batch_idx}: Loss Rolling Average (Past 20 batches) {np.mean(loss_values[-20:]):.4f}, LR {scheduler.get_last_lr()[0]}, | GPU Mem: {torch.cuda.memory_allocated()/1e6:.2f}MB")
            print(f"Finished Loop {finished_loop_counter}, last image index {last_image_index}, trained images {trained_images}: Loss {loss:.4f}", flush=True)
            
            #for param in model.parameters():
            #    print(param)
                
            # Step 6: Evaluation
            # ------------------
            if batch_idx % 4 == 0:
                model.eval()
                correct = 0
                total = 0
                last_test_idx = -1
                with torch.no_grad():
                    while total < test_size:
                        test_subset = Subset(test_dataset, indices=range(last_test_idx+1, last_test_idx+1 + test_size - total))
                        test_loader = DataLoader(test_subset, batch_size=minibatch_size, shuffle=False)#, pin_memory=True)#, pin_memory_device=device)
                        for idx, (data_, target_) in enumerate(test_loader):
                            last_test_idx += 1
                            print(f'test index: {last_test_idx}, successful: {total}', flush=True)
                            data_ = data_.view(data_.size(0), -1)#.to(device)
                            target_ = target_.to(device)

                            output = model(data_)
                            if torch.any(torch.isnan(output)):
                                print('test nan encountered', flush=True)
                                del output
                                continue
                                
                            
                            _, predicted = torch.max(output.data, 1)
                            total += target_.size(0)
                            correct += (predicted.to(device) == target_).sum().item()
                            print('test prcoessed', flush=True)
                #if batch_idx % 5 == 0:
                print(f'Test Accuracy: {100 * correct / total:.2f}%\n', flush=True)
                accuracies[trained_images] = correct/total
                torch.cuda.empty_cache()  # Clean up GPU memory

        finished_loop_counter += 1
        
                #scheduler.step(loss)
        #scheduler.step(total_loss/epoch_size*batch_size)
        #print(total_loss/epoch_size*batch_size)
        
    #parameters = model.state_dict()
    #
    #with open(f'noisy QNN test/QVC50_8q_encoded_50batch_15000epoch_0005lr_depol{i}_param_states.pkl', 'wb') as file:
    #    pickle.dump(parameters, file, protocol=pickle.HIGHEST_PROTOCOL)
    #with open(f'noisy QNN test/QVC50_8q_encoded_50batch_15000epoch_0005lr_depol{i}_optimizer_states.pkl', 'wb') as file:
    #    pickle.dump(optimizer, file, protocol=pickle.HIGHEST_PROTOCOL)

    print(f'last image index: {last_image_index}', flush=True)
    results['nan_counts'] = nan_counts
    results['training samples'] = training_samples[1:]
    results['accuracies'] = accuracies
    results['loss values'] = loss_values
    results['gradients'] = gradients_list


    #state = {'state_dict': model.state_dict(), 
    #         'optimizer': optimizer.state_dict(),
    #         'last_image_index': last_image_index,}

    #torch.save(state, f'noisy_QNN_test/QVC50_10q_encoded_50batch_15000epoch_0005lr_depol{i}_param_states_part1.pth.tar')

    with open(f'noisy_QNN_test/QVC50_10q_encoded_50batch_15000epoch_0005lr_depol_and_damping{i_}.pkl', 'wb') as file:
        pickle.dump(results, file, protocol=pickle.HIGHEST_PROTOCOL)
    
    return


def depol_and_generalised_damping_job(p_damping : float, gamma : float, p_depolarising : float = 1.46789129e-03) -> None:
    """
    function to run the depolarising and damping noise job with quantum circuits.
    Note classical layer is not used in this job.
    Args:
        p_damping: damping noise rate
        gamma: damping noise rate
        i: index for the depolarising noise rate
    """

    # Check CUDA availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)
    # Step 1: Data Preprocessing
    # --------------------------
    # Resize images to 4x4, convert to tensors, and scale pixel values to [0, π]
    transform = transforms.Compose([
        transforms.Resize((32, 32)),          # Downsample to 16x16 (256 pixels)
        transforms.ToTensor(),              # Convert to tensor
        #transforms.Lambda(lambda x: x * np.pi),  # Scale pixel values to [0, π]
        transforms.Lambda(lambda x: x.flatten()),  # Flatten to 16D vector
        transforms.Lambda(lambda x: x / torch.norm(x)), # Normalize to unit length
        transforms.Lambda(lambda x: x.to(device)),
    ])

    # Load MNIST dataset
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    # Use smaller subsets for faster training (optional)
    epoch_size = 15000
    batch_size = 50
    test_size = 250
    minibatch_size = 1



    num_qubits = 10  # Using 9 qubits if using amplitude+phase encoding

    #try:
    #    # Try to use GPU simulator
    #    dev = qml.device("lightning.gpu", wires=num_qubits)
    #    print("Using lightning.gpu quantum simulator", flush=True)
    #except:
    # Fallback to CPU with GPU classical components
    #dev = qml.device("default.qubit", wires=num_qubits)
    #print("Using default.qubit simulator - install lightning.gpu for GPU acceleration", flush=True)
    dev = qml.device("default.mixed", wires=num_qubits)
    print("Using default.mixed simulator - install lightning.gpu for GPU acceleration", flush=True)
    #dev = qml.device("default.qubit", wires=num_qubits)
    #dev = qml.device("default.mixed", wires=num_qubits)
    #dev = qml.device("lightning.qubit", wires=num_qubits)

    layers = 75
    weight_shapes = {"weights": (int(layers*3), num_qubits)}



    # Step 5: Training Loop
    # ---------------------
    print(f'depolarising noise rate: {p_depolarising}', flush=True)

    #damping_rates = np.logspace(-5,np.log10(3/4),10)
    #print(i_, flush=True)
    #p_damping = damping_rates[i_]
    print(f'damping probability (p_): {p_damping}', flush=True)
    print(f'damping strength (gamma): {gamma}', flush=True)

    results = {}
    
    model = HybridModel(dev=dev, device=device, num_qubits=num_qubits, weight_shapes=weight_shapes, 
                        noise_model='depol and generalised damping',
                        p_depolarising = p_depolarising,
                        p_damping = p_damping,
                        gamma = gamma).to(device)
                        
    #criterion = FullCriterion()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=1/np.e, patience=50)

    accuracies = {}
    training_samples = [0]
    loss_values = []
    gradients_list = []
    nan_counts = []


    trained_images = 0
    last_image_index = -1
    finished_loop_counter = 0

    while trained_images < epoch_size:
        train_subset = Subset(train_dataset, indices=range(last_image_index+1, last_image_index+1 + epoch_size - trained_images))
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)#, pin_memory=True)#, pin_memory_device=device)
        for batch_idx, (data, target) in enumerate(train_loader):
            #if batch_idx == 150:# or batch_idx == 200:
            #    inp=input('job Continue? (y/n): ')
            #    if inp=='y':
            #        pass
            #print(scheduler.get_last_lr())
                
            model.train()
            # Flatten images: (batch_size, 1, 4, 4) -> (batch_size, 16)
            data = data.view(data.size(0), -1)#.to(device)
            target_new = torch.zeros(len(data),10).to(device)
            for j in range(len(target)):
                target_new[j][target[j]] = 1

            optimizer.zero_grad()

            #output = model(data)
            #loss = criterion(output, target_new)
            #loss.backward()
            loss, nan_count = process_batch(model, data, target_new, batch_size, batch_idx, minibatch_size, criterion)
            trained_images += len(data) - nan_count
            nan_counts.append(nan_count)
            
            optimizer.step()

            #gradients_linear_weights = optimizer.param_groups[0]['params'][0].grad.data.cpu().numpy()
            #gradients_linear_offsets = optimizer.param_groups[0]['params'][1].grad.data.cpu().numpy()
            #gradients_quantum = optimizer.param_groups[0]['params'][2].grad.data.cpu().numpy()
            #gradients_list.append({'linear weights': gradients_linear_weights,
            #                        'linear offsets': gradients_linear_offsets,
            #                        'quantum': gradients_quantum}
            #                        )
            gradients_quantum = optimizer.param_groups[0]['params'][0].grad.data.cpu().numpy()
            gradients_list.append({'quantum': gradients_quantum})

            #loss_values.append(loss.item())
            loss_values.append(loss)
            training_samples.append(training_samples[-1]+len(data))
            last_image_index = training_samples[-1] - 1
            #if batch_idx % 5 == 0:
            #print(f"Epoch {epoch}, Batch {batch_idx}: Loss {loss.item():.4f}, LR {scheduler.get_last_lr()[0]}, | GPU Mem: {torch.cuda.memory_allocated()/1e6:.2f}MB")
            #print(f"Epoch {epoch}, Batch {batch_idx}: Loss Rolling Average (Past 20 batches) {np.mean(loss_values[-20:]):.4f}, LR {scheduler.get_last_lr()[0]}, | GPU Mem: {torch.cuda.memory_allocated()/1e6:.2f}MB")
            print(f"Finished Loop {finished_loop_counter}, last image index {last_image_index}, trained images {trained_images}: Loss {loss:.4f}", flush=True)
            
            #for param in model.parameters():
            #    print(param)
                
            # Step 6: Evaluation
            # ------------------
            if batch_idx % 4 == 0:
                model.eval()
                correct = 0
                total = 0
                last_test_idx = -1
                with torch.no_grad():
                    while total < test_size:
                        test_subset = Subset(test_dataset, indices=range(last_test_idx+1, last_test_idx+1 + test_size - total))
                        test_loader = DataLoader(test_subset, batch_size=minibatch_size, shuffle=False)#, pin_memory=True)#, pin_memory_device=device)
                        for idx, (data_, target_) in enumerate(test_loader):
                            last_test_idx += 1
                            print(f'test index: {last_test_idx}, successful: {total}', flush=True)
                            data_ = data_.view(data_.size(0), -1)#.to(device)
                            target_ = target_.to(device)

                            output = model(data_)
                            if torch.any(torch.isnan(output)):
                                print('test nan encountered', flush=True)
                                del output
                                continue
                                
                            
                            _, predicted = torch.max(output.data, 1)
                            total += target_.size(0)
                            correct += (predicted.to(device) == target_).sum().item()
                            print('test prcoessed', flush=True)
                #if batch_idx % 5 == 0:
                print(f'Test Accuracy: {100 * correct / total:.2f}%\n', flush=True)
                accuracies[trained_images] = correct/total
                torch.cuda.empty_cache()  # Clean up GPU memory

        finished_loop_counter += 1
        
                #scheduler.step(loss)
        #scheduler.step(total_loss/epoch_size*batch_size)
        #print(total_loss/epoch_size*batch_size)
        
    #parameters = model.state_dict()
    #
    #with open(f'noisy QNN test/QVC50_8q_encoded_50batch_15000epoch_0005lr_depol{i}_param_states.pkl', 'wb') as file:
    #    pickle.dump(parameters, file, protocol=pickle.HIGHEST_PROTOCOL)
    #with open(f'noisy QNN test/QVC50_8q_encoded_50batch_15000epoch_0005lr_depol{i}_optimizer_states.pkl', 'wb') as file:
    #    pickle.dump(optimizer, file, protocol=pickle.HIGHEST_PROTOCOL)

    print(f'last image index: {last_image_index}', flush=True)
    results['nan_counts'] = nan_counts
    results['training samples'] = training_samples[1:]
    results['accuracies'] = accuracies
    results['loss values'] = loss_values
    results['gradients'] = gradients_list


    #state = {'state_dict': model.state_dict(), 
    #         'optimizer': optimizer.state_dict(),
    #         'last_image_index': last_image_index,}

    #torch.save(state, f'noisy_QNN_test/QVC50_10q_encoded_50batch_15000epoch_0005lr_depol{i}_param_states_part1.pth.tar')

    with open(f'noisy_QNN_test/QVC75_10q_encoded_50batch_15000epoch_0005lr_depol_and_generalised_damping{p_depolarising:.2e}_{gamma:.2e}_{p_damping:.2e}.pkl', 'wb') as file:
        pickle.dump(results, file, protocol=pickle.HIGHEST_PROTOCOL)
    
    return