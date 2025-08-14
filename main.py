import numpy as np
from util_funcs import calc_amplitude_damping_params
from jobs import depolarising_job, two_qubit_noise_job, gaussian_job, depol_and_damping_job, depol_and_generalised_damping_job
from jobs import depolarising_smoothedPQC_job, depolarising_smoothedPQC_cpu_mpi_parallel_job, depolarising_smoothedPQC_cpu_parallel_job, depolarising_smoothedPQC_torch_parallel_job
import torch.multiprocessing as mp
import os
import argparse

def main(*args, noise_model: str) -> None:
    """
    Main function to run the specified noise model job.
    Args:
        noise_model (str): The type of noise model to use.
        strength (int): The strength of the noise model.
    """
    if noise_model == "depolarising":
        print(f"Using depolarizing noise model with strength {args}")
        depolarising_job(*args)
    
    elif noise_model == "two qubit noise":
        print(f"Using two qubit noise model with single-qubit depolarising channel strength {args}")
        two_qubit_noise_job(*args)
    
    elif noise_model == "gaussian":
        print(f"Using Gaussian noise model with strength {args}")
        gaussian_job(*args)
    
    elif noise_model == "gaussian non zero mean":
        print(f"Using non zero mean Gaussian noise model with strength {args}")
        gaussian_job(*args, mu=np.pi/2)
    
    elif noise_model == "depol and damping":
        print(f"Using depolarizing and damping noise model with damping strength {args}")
        depol_and_damping_job(*args)
    
    elif noise_model == "depol and generalised damping":
        print(f"Using depolarizing and damping noise model with generalised damping strengths p_damping:{args[0]}, gamma:{args[1]}")
        depol_and_generalised_damping_job(*args)
    
    elif noise_model == "smoothedPQC depolarising":
        print(f"Using test noise model with strength {args}")
        depolarising_smoothedPQC_job(*args)
    
    elif noise_model == "smoothedPQC depolarising cpu parallel":
        print(f"Using smoothedPQC depolarising noise model with strength {args}")
        depolarising_smoothedPQC_cpu_parallel_job(*args)
    
    elif noise_model == "smoothedPQC depolarising cpu mpi parallel":
        print(f"Using smoothedPQC depolarising noise model with strength {args}")
        depolarising_smoothedPQC_cpu_mpi_parallel_job(*args)
    
    elif noise_model == "smoothedPQC depolarising torch parallel":
        print(f"Using smoothedPQC depolarising noise model with strength {args}")
        # if use torch.multiprocessing
        size, p_depol = args
        mp.spawn(
            depolarising_smoothedPQC_torch_parallel_job, args=(size, p_depol), nprocs=size, join=True
        )
        
        #if use torchrun
        #size, p_depol = args
        #parser = argparse.ArgumentParser()
        #parser.add_argument("--local-rank", "--local_rank", type=int)
        #args = parser.parse_args()
        #rank = args.local_rank
        #depolarising_smoothedPQC_torch_parallel_job(rank, size, p_depol)
        
    else:
        raise ValueError(f"Unknown noise model: {noise_model}")

if __name__ == "__main__":
    #import torch
    #print(torch.cuda.is_available())
    p_damping, gamma, Gammda_dt = calc_amplitude_damping_params(decay_rate = 0.5, 
                                                     relative_T = 1.0, 
                                                     dt = 3.2e-9*1.5*np.log2(1/1e-4))
    print(f"p_damping: {p_damping}, gamma: {gamma}, Gamma*dt: {Gammda_dt}", flush=True)
    # Example usage
    main(p_damping, gamma, 4, noise_model="depol and generalised damping")
    
    
    
    ######
    size = 2
    p_depol = 0.01
    main(size, p_depol, noise_model='smoothedPQC depolarising torch parallel')
