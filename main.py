import numpy as np
from jobs import depolarising_job, two_qubit_noise_job, gaussian_job, depol_and_damping_job

def main(noise_model: str, strength: int) -> None:
    """
    Main function to run the specified noise model job.
    Args:
        noise_model (str): The type of noise model to use.
        strength (int): The strength of the noise model.
    """
    if noise_model == "depolarising":
        print(f"Using depolarizing noise model with strength {strength}")
        depolarising_job(strength)
    
    elif noise_model == "two qubit noise":
        print(f"Using two qubit noise model with single-qubit depolarising channel strength {strength}")
        two_qubit_noise_job(strength)
    
    elif noise_model == "gaussian":
        print(f"Using Gaussian noise model with strength {strength}")
        gaussian_job(strength)
    
    elif noise_model == "gaussian non zero mean":
        print(f"Using non zero mean Gaussian noise model with strength {strength}")
        gaussian_job(strength, mu=np.pi/2)
    
    elif noise_model == "depol and damping":
        print(f"Using depolarizing and damping noise model with damping strength {strength}")
        depol_and_damping_job(i_ = strength)
    else:
        raise ValueError(f"Unknown noise model: {noise_model}")
    
if __name__ == "__main__":
    #import torch
    #print(torch.cuda.is_available())
    main('depolarising', 1)