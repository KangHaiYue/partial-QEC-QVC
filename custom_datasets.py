import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
import numpy as np
#from nunmpy.random import rand
import os
import itertools
import pickle
import matplotlib.pyplot as plt
#from sklearn.datasets import make_regression
#from typing import Sequence

def cartesian_product_transpose_pp(arrays):
    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty((la, *map(len, arrays)), dtype=dtype)
    idx = slice(None), *itertools.repeat(None, la)
    for i, a in enumerate(arrays):
        arr[i, ...] = a[idx[:la-i]]
    return arr.reshape(la, -1).T


class CustomWaveDataset(Dataset):
        def __init__(self, input_dim: int = 2, 
                     spectrum_size: int = 5, 
                     num_samples_train: int = 10000, 
                     num_samples_test: int = 500,
                     noise_std: float = 0.1,
                     transform = None,
                     root: str = './fft_data'
                     ):
            
            self.input_dim = input_dim
            self.spectrum_size = spectrum_size
            self.num_samples_train = num_samples_train
            self.num_samples_test = num_samples_test
            self.noise_std = noise_std
            self.transform = transform
            self.root = root
            self.data_file = os.path.join(root, f"CustomWaveDataset_{input_dim}_{spectrum_size}_{num_samples_train}_{num_samples_test}_{noise_std}.npz")
            
            self.num_samples = num_samples_train + num_samples_test
            if os.path.exists(self.data_file):
                data = np.load(self.data_file)
                self.x = data['x']
                self.y = data['y']
                print(f"Data loaded from {self.data_file}")
            else:
                # Generate and save data
                x1 = np.random.rand(self.num_samples_train, self.input_dim) * 2 * np.pi
                
                # Create homogeneous grid from 0 to 2π in each dimension
                grid_points_per_dim = int(np.ceil(self.num_samples_test**(1/self.input_dim)))
                homogeneous_generators = [np.linspace(0, 2*np.pi, grid_points_per_dim, endpoint=False) for _ in range(self.input_dim)]
                x2_grid = cartesian_product_transpose_pp(homogeneous_generators)
                
                # Take only the required number of test samples
                x2 = x2_grid[:self.num_samples_test]
                
                # Concatenate random and homogeneous samples
                x = np.concatenate([x1, x2], axis=0)
                
                if input_dim < 2:
                    full_spectrum_generators = [np.linspace(0, self.spectrum_size-1, self.spectrum_size) for i in range(self.input_dim)]
                    full_spectrum = cartesian_product_transpose_pp(full_spectrum_generators)
                    omega_dot_x = np.einsum('ij,kj->ik', full_spectrum, x)
                
                    fourier_coeff_an = np.random.rand(len(full_spectrum))
                    fourier_coeff_bn = np.random.rand(len(full_spectrum))
                    
                    y = np.sum(np.stack((fourier_coeff_bn,)*self.num_samples).T *np.sin(omega_dot_x)
                                    + np.stack((fourier_coeff_an,)*self.num_samples).T *np.cos(omega_dot_x), axis=0).reshape((self.num_samples,1)) + np.random.rand(self.num_samples,1) * noise_std
                    
                    c = np.concatenate((np.array([fourier_coeff_an[0]]), 0.5*(fourier_coeff_an[1:] - 1j*fourier_coeff_bn[1:])))
                    fourier_coeff_cn = 2*(c)/(np.max(y) - np.min(y))
                    c[0] -= 2 * np.min(y) / (np.max(y) - np.min(y)) + 1
                    
                    y = 2 * (y - np.min(y)) / (np.max(y) - np.min(y)) - 1 # Normalize to [-1, 1]
                    self.x = x
                    self.y = y
                
                else:
                    full_spectrum_generators = [np.linspace(-self.spectrum_size+1, self.spectrum_size-1, 2*self.spectrum_size-1) for i in range(self.input_dim)]
                    full_spectrum = cartesian_product_transpose_pp(full_spectrum_generators)
                    omega_dot_x = np.einsum('ij,kj->ik', full_spectrum, x)
                    
                    freq_tuples = [tuple(f) for f in full_spectrum]
                    freq_to_idx = {f: i for i, f in enumerate(freq_tuples)}
                    
                    # Generate random Fourier coefficients with Hermitian symmetry
                    c = np.zeros(len(full_spectrum), dtype=complex)
                    for i, f in enumerate(freq_tuples[:len(freq_tuples)//2 + 1]):
                        neg_f = tuple(-np.array(f))
                        real = np.random.randn()
                        imag = np.random.randn()
                        c[i] = real + 1j * imag
                        if neg_f in freq_to_idx and neg_f != f:
                            c[freq_to_idx[neg_f]] = real - 1j * imag
                    c[len(freq_tuples)//2] = np.random.randn()
                    
                    # Generate y values
                    y = np.sum(np.stack((c,)*self.num_samples).T *np.exp(1j*omega_dot_x), axis=0).real.reshape((self.num_samples,1)) + np.random.randn(self.num_samples,1) * noise_std
                    # Normalize coefficients
                    c = 2*c/(np.max(y) - np.min(y))
                    c[len(freq_tuples)//2] -= 2 * np.min(y) / (np.max(y) - np.min(y)) + 1
                    # Normalize y to [-1, 1]
                    y = 2 * (y - np.min(y)) / (np.max(y) - np.min(y)) - 1 # Normalize to [-1, 1]
                    self.x = x
                    self.y = y
                    # Reshape c to multi-dimensional array
                    c = c.reshape([spectrum_size*2-1]*input_dim)
                    
                with open(f'{root}/original_fourier_coeff_{input_dim}_{spectrum_size}_{num_samples_train}_{num_samples_test}_{noise_std}.pkl', 'wb') as file:
                    pickle.dump(c, file, protocol=pickle.HIGHEST_PROTOCOL)
                    
                np.savez(self.data_file, x=x, y=y)
                print(f"Data generated and saved to {self.data_file}")
            
        def __len__(self):
            return len(self.x)

        def __getitem__(self, idx):
            x = self.x[idx]
            y = self.y[idx]
            if self.transform:
                x = self.transform(x)
                y = self.transform(y)
            return x, y

class CustomFFTInputs(Dataset):
        def __init__(self, input_dim: int = 2, 
                     fft_density: int = 5,
                     sampling_overhead_multiplier: int = 1,
                     transform = None,
                     root: str = './fft_data'
                     ):

            self.input_dim = input_dim
            self.fft_density = fft_density
            self.sampling_overhead_multiplier = sampling_overhead_multiplier
            self.transform = transform
            self.root = root
            self.data_file = os.path.join(root, f"CustomFFTInputs_{input_dim}_{fft_density}.npz")
            #print(f'{self.data_file}')
            if os.path.exists(self.data_file):
                data = np.load(self.data_file)
                self.x = data['arr_0'] if 'arr_0' in data else data['x']
                print(f"FFT input loaded from {self.data_file}")
            else:
                homogeneous_inputs_generators = [np.linspace(0, 2*np.pi*self.sampling_overhead_multiplier, self.fft_density*self.sampling_overhead_multiplier) 
                                                 for i in range(self.input_dim)]
                x = cartesian_product_transpose_pp(homogeneous_inputs_generators)
                self.x = x
                np.savez(self.data_file, x=x)
                print(f"FFT input data generated and saved to {self.data_file}")

        def __len__(self):
            return len(self.x)

        def __getitem__(self, idx):
            x = self.x[idx]
            if self.transform:
                x = self.transform(x)
            return x
        
        
#class CustomLinearDataset(Dataset):
#        def __init__(self, dim=10, num_samples=10000, noise_std=0.1):
#            
#            self.x, self.y = make_regression(
#                n_samples=num_samples, 
#                n_features=dim, 
#                n_informative=dim,
#                n_targets=1,
#                noise=noise_std
#                )
#
#        def __len__(self):
#            return len(self.x)
#
#        def __getitem__(self, idx):
#            return self.x[idx], self.y[idx]




if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Lambda(lambda x: torch.from_numpy(x).float()),
        transforms.Lambda(lambda x: x.to(device))
    ])
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = CustomWaveDataset(input_dim=1, spectrum_size=10, num_samples=10000, noise_std=0.01,
                                transform=transform, root='./fft_data')
    subset = Subset(dataset, indices=range(1000,2000))
    dataloader = DataLoader(subset, batch_size=1000, shuffle=True)

    for batch_idx, (data, target) in enumerate(dataloader):
        #print(f"Batch {batch_idx}:")
        #rint("Data:", data)
        #print("Target:", target)
        #if batch_idx == 1:  # Just show first 3 batches
        #    break
        plt.scatter(torch.flatten(data).numpy(), torch.flatten(target).numpy())
        plt.show()
    #full_spectrum_generators = [np.linspace(0, 2, 3) for i in range(3)]
    #print(cartesian_product_transpose_pp(full_spectrum_generators))