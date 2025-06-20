
def depolarising_job(i: int) -> None:
    """    
    function to run the depolarising noise job with quantum circuits.
    """
    
    # Import required libraries
    import pennylane as qml
    from pennylane import numpy as np
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader, Subset
    from functools import partial
    from qml_modules import HybridModel
    from util_funcs import process_batch, load_checkpoint
    import pickle


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

    results_total = {}
    depolarising_rates = np.logspace(-5,np.log10(3/4),10)

    mean = 0
    #standard_deviations = np.linspace(np.pi/50,np.pi/4,6)
    #standard_deviations = np.logspace(-5,np.log10(np.pi/4),6)

    #i = 0
    print(i, flush=True)

    p_depolarising = depolarising_rates[i]
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

            gradients_linear_weights = optimizer.param_groups[0]['params'][0].grad.data.cpu().numpy()
            gradients_linear_offsets = optimizer.param_groups[0]['params'][1].grad.data.cpu().numpy()
            gradients_quantum = optimizer.param_groups[0]['params'][2].grad.data.cpu().numpy()
            # if no classical layer
            #gradients_quantum = optimizer.param_groups[0]['params'][0].grad.data.cpu().numpy()
            gradients_list.append({'linear weights': gradients_linear_weights,
                                    'linear offsets': gradients_linear_offsets,
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
                                print(f'test nan encountered', flush=True)
                                del output
                                continue
                                
                            
                            _, predicted = torch.max(output.data, 1)
                            total += target_.size(0)
                            correct += (predicted.to(device) == target_).sum().item()
                            print(f'test prcoessed', flush=True)
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

    with open(f'noisy_QNN_test/QVC50_10q_encoded_50batch_15000epoch_0005lr_depol{i}.pkl', 'wb') as file:
        pickle.dump(results, file, protocol=pickle.HIGHEST_PROTOCOL)
        
    return



def depolarising_and_damping_job(i: int) -> None:
    """
    function to run the depolarising and damping noise job with quantum circuits.
    """
    # Import required libraries
    import pennylane as qml
    from pennylane import numpy as np
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader, Subset
    from qml_modules import HybridModel
    from util_funcs import process_batch
    import pickle



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

    depolarising_rates = np.logspace(-5,np.log10(3/4),10)

    i = 0
    print(i, flush=True)

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

            gradients_linear_weights = optimizer.param_groups[0]['params'][0].grad.data.cpu().numpy()
            gradients_linear_offsets = optimizer.param_groups[0]['params'][1].grad.data.cpu().numpy()
            gradients_quantum = optimizer.param_groups[0]['params'][2].grad.data.cpu().numpy()
            # if no classical layer
            #gradients_quantum = optimizer.param_groups[0]['params'][0].grad.data.cpu().numpy()
            gradients_list.append({'linear weights': gradients_linear_weights,
                                    'linear offsets': gradients_linear_offsets,
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
                                print(f'test nan encountered', flush=True)
                                del output
                                continue
                                
                            
                            _, predicted = torch.max(output.data, 1)
                            total += target_.size(0)
                            correct += (predicted.to(device) == target_).sum().item()
                            print(f'test prcoessed', flush=True)
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

    with open(f'noisy_QNN_test/QVC50_10q_encoded_50batch_15000epoch_0005lr_extended_depol{i}.pkl', 'wb') as file:
        pickle.dump(results, file, protocol=pickle.HIGHEST_PROTOCOL)
        
    return


def gaussian_job(sigma_index: int, mu:float = 0) -> None:
    # Import required libraries
    import pennylane as qml
    from pennylane import numpy as np
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader, Subset
    from functools import partial
    from qml_modules import HybridModel
    from util_funcs import process_batch, load_checkpoint
    import pickle



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

    standard_deviations = np.logspace(-5,np.log10(np.pi/4),6)

    sigma_index = 0
    print(sigma_index, flush=True)

    sigma = standard_deviations[sigma_index]
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

            gradients_linear_weights = optimizer.param_groups[0]['params'][0].grad.data.cpu().numpy()
            gradients_linear_offsets = optimizer.param_groups[0]['params'][1].grad.data.cpu().numpy()
            gradients_quantum = optimizer.param_groups[0]['params'][2].grad.data.cpu().numpy()
            # if no classical layer
            #gradients_quantum = optimizer.param_groups[0]['params'][0].grad.data.cpu().numpy()
            gradients_list.append({'linear weights': gradients_linear_weights,
                                    'linear offsets': gradients_linear_offsets,
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
                                print(f'test nan encountered', flush=True)
                                del output
                                continue
                                
                            
                            _, predicted = torch.max(output.data, 1)
                            total += target_.size(0)
                            correct += (predicted.to(device) == target_).sum().item()
                            print(f'test prcoessed', flush=True)
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

    with open(f'noisy_QNN_test/QVC50_10q_encoded_50batch_15000epoch_0005lr_gaussian_pi2mean{sigma_index}.pkl', 'wb') as file:
        pickle.dump(results, file, protocol=pickle.HIGHEST_PROTOCOL)
    
    return
