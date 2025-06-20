import numpy as np
from jobs import depolarising_job, depolarising_and_damping_job, gaussian_job

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
    
    elif noise_model == "depol and damping":
        print(f"Using depolarizing and damping noise model with strength {strength}")
        depolarising_and_damping_job(strength)
        
    elif noise_model == "gaussian":
        print(f"Using Gaussian noise model with strength {strength}")
        gaussian_job(strength)
    
    elif noise_model == "gaussian non zero mean":
        print(f"Using non zero mean Gaussian noise model with strength {strength}")
        gaussian_job(strength, mu=np.pi/2)
        
    else:
        raise ValueError(f"Unknown noise model: {noise_model}")
    
if __name__ == "__main__":
    #import torch
    #print(torch.cuda.is_available())
    main('depolarising', 1)