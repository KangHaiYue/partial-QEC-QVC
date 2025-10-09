import numpy as np
from typing import Sequence


class SuperOpTools:
    '''
    The following 3 staticmethods are copied from PyQuil's source codes,
    to be used to convert between superoperator and Kraus operator representations.
    forest-benchmarking.operator_tools.superoperator_transformations.py
    @Forest Benchmarking: QCVV using PyQuil, 2019, 10.5281/zenodo.3455847, https://doi.org/10.5281/ze    nodo.3455847
    '''
    @staticmethod
    def kraus2superop(kraus_list: Sequence) -> np.ndarray:
        """Convert a list of Kraus operators to a superoperator"""
        # Get the number of qubits from the first Kraus operator
        dimension = kraus_list[0].shape[0]
        
        # Initialize the superoperator as a zero matrix
        superop = np.zeros((dimension**2, dimension**2), dtype=complex)
        
        # Loop over each Kraus operator and add its contribution to the superoperator
        for kraus in kraus_list:
            superop += np.kron(kraus.conj(), kraus)
        
        return superop
    
    @staticmethod
    def superop2kraus(superop: np.ndarray, tol: float = 1e-10) -> list:
        """Convert a superoperator to a list of Kraus operators"""
        # Get the number of qubits from the superoperator shape
        dim_original_kraus = int(np.sqrt(superop.shape[0]))
        
        # superop to choi
        choi_matrix = np.reshape(superop, [dim_original_kraus] * 4).swapaxes(0, 3).reshape([dim_original_kraus ** 2, dim_original_kraus ** 2])
        
        #choi to kraus
        eigvals, eigvecs = np.linalg.eigh(choi_matrix)
        
        return [np.lib.scimath.sqrt(eigval) * SuperOpTools.unvec(np.array([evec]).T) 
                for eigval, evec in zip(eigvals, eigvecs.T) if abs(eigval) > tol]
    
    @staticmethod
    def superop2choi(superop: np.ndarray) -> np.ndarray:
        """
        Convert a superoperator to its Choi matrix form.
        :param superop: The superoperator as a square numpy array.
        :return: The Choi matrix as a numpy array.
        """
        dim = int(np.sqrt(superop.shape[0]))
        # Reshape and swap axes to get the Choi matrix
        choi = np.reshape(superop, [dim, dim, dim, dim]).swapaxes(0, 3).reshape([dim**2, dim**2])
        return choi
    
    @staticmethod
    def unvec(vector) -> np.ndarray:
        """
        Take a column vector and turn it into a matrix.

        By default, the unvec'ed matrix is assumed to be square. Specifying shape = [N, M] will
        produce a N by M matrix where N is the number of rows and M is the number of columns.

        Consider::

            |A>> := vec(A) = (a, c, b, d)^T

        `unvec(|A>>)` should return::

            A = [[a, b]
                [c, d]]

        :param vector: A (N*M) by 1 numpy array.
        :param shape: The shape of the output matrix; by default, the matrix is assumed to be square.
        :return: Returns a N by M matrix.
        """
        vector = np.asarray(vector)
        
        dim = int(np.sqrt(vector.size))
        shape = dim, dim
        matrix = vector.reshape(*shape).T
        return matrix



if __name__ == "__main__":
    ''' test usage of SuperOpTools '''
    import pennylane as qml
    ZZ = np.array([[1, 0, 0, 0],
                   [0, -1, 0, 0],    
                   [0, 0, -1, 0],
                   [0, 0, 0, 1]])
    CZ = qml.CZ.compute_matrix()
    
    #ZZ_superop = SuperOpTools.kraus2superop([CZ])
    
    #depol_single_kraus = qml.DepolarizingChannel.compute_kraus_matrices(0.1)
    #depol_two_kraus =  [qml.math.kron(depol_single_kraus[i], 
    #                                  depol_single_kraus[j]) 
    #                    for i in range(4) 
    #                    for j in range(4)]
    
    #depol_two_superop = SuperOpTools.kraus2superop(depol_two_kraus)
    #print(ZZ_superop@depol_two_superop - depol_two_superop@ZZ_superop)
    
    
    
    def generate_haar_random_unitary(n_qubits, seed=None):
        """
        Generate a Haar random unitary matrix using QR decomposition.
        This is the mathematically rigorous way to sample from the Haar measure.
        """
        if seed is not None:
            np.random.seed(seed)
            
        dim = 2**n_qubits
        # Generate random complex matrix with Gaussian entries
        A = np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim)
        
        # QR decomposition
        Q, R = np.linalg.qr(A)
        
        # Adjust phases to ensure proper Haar distribution
        # This step is crucial for true Haar randomness
        phases = np.diag(R) / np.abs(np.diag(R))
        U = Q @ np.diag(phases)
        
        return U

    def haar_random_state_generator(n_qubits=2, depol_prob=0.1, seed=None):
        """
        Generate an n-qubit TRUE Haar random state, apply noise sequence, and calculate fidelity.
        
        Args:
            n_qubits: Number of qubits
            depol_prob: Depolarizing channel probability
            seed: Random seed for reproducibility
            
        Returns:
            tuple: (original_state, noisy_state, fidelity)
        """
        dev = qml.device("default.mixed", wires=n_qubits)
        
        # Generate TRUE Haar random unitary matrix
        haar_unitary = generate_haar_random_unitary(n_qubits, seed)
        
        @qml.qnode(dev)
        def generate_original_state():
            # Apply the Haar random unitary to |0...0⟩ to get Haar random state
            qml.QubitUnitary(haar_unitary, wires=list(range(n_qubits)))
            return qml.state()
        
        @qml.qnode(dev)
        def generate_noisy_state():
            # Apply the same Haar random unitary to get the same initial state
            qml.QubitUnitary(haar_unitary, wires=list(range(n_qubits)))
            
            # Apply noise sequence: CZ gates and depolarizing channels
            if n_qubits > 1:
                # Apply CZ gates between adjacent qubits
                for i in range(n_qubits - 1):
                    qml.CZ(wires=[i, i + 1])
            
            # Apply depolarizing channel to each qubit
            for i in range(n_qubits):
                qml.DepolarizingChannel(depol_prob, wires=[i])
            
            if n_qubits > 1:
                # Apply CZ gates again
                for i in range(n_qubits - 1):
                    qml.CZ(wires=[i, i + 1])
            
            # Apply depolarizing channel again
            for i in range(n_qubits):
                qml.DepolarizingChannel(depol_prob, wires=[i])
            
            return qml.state()
        
        # Generate states
        original_state = generate_original_state()
        noisy_state = generate_noisy_state()
        
        # Calculate fidelity between density matrices
        # Convert pure state to density matrix if needed
        if original_state.ndim == 1:
            rho_original = np.outer(original_state, np.conj(original_state))
        else:
            rho_original = original_state
            
        # Fidelity between two density matrices: F = Tr(rho1 * rho2) when one is pure
        fidelity = np.real(np.trace(rho_original @ noisy_state))
        
        return original_state, noisy_state, fidelity
    
    # Test the function and verify Haar randomness
    n_qubits = 3
    depol_prob = 0.1
    
    original, noisy, fid = haar_random_state_generator(n_qubits, depol_prob, seed=42)
    print(f"Number of qubits: {n_qubits}")
    print(f"Depolarizing probability: {depol_prob}")
    print(f"Fidelity between original and noisy state: {fid:.6f}")
    
    if original.ndim == 1:
        print(f"Original state norm: {np.linalg.norm(original):.6f}")
    else:
        print(f"Original state trace: {np.real(np.trace(original)):.6f}")
        
    if noisy.ndim == 1:
        print(f"Noisy state norm: {np.linalg.norm(noisy):.6f}")
    else:
        print(f"Noisy state trace: {np.real(np.trace(noisy)):.6f}")
    
    # Calculate average fidelity over 100 instances of 2-qubit Haar random states
    print("\n--- Average Fidelity Analysis ---")
    
    num_instances = 1000
    n_qubits_test = 2
    depol_prob_test = 1e-10  # Keep noise constant at 10^-10
    
    fidelities = []
    print(f"Calculating fidelity for {num_instances} random Haar states ({n_qubits_test} qubits, noise={depol_prob_test})...")
    
    for i in range(num_instances):
        _, _, fidelity = haar_random_state_generator(n_qubits_test, depol_prob_test, seed=i)
        fidelities.append(fidelity)
        
        # Progress indicator
        if (i + 1) % 20 == 0:
            print(f"Progress: {i + 1}/{num_instances}")
    
    fidelities = np.array(fidelities)
    
    # Statistics
    mean_fidelity = np.mean(fidelities)
    std_fidelity = np.std(fidelities)
    min_fidelity = np.min(fidelities)
    max_fidelity = np.max(fidelities)
    
    print(f"\nResults for {num_instances} instances (noise = {depol_prob_test}):")
    print(f"Average fidelity: {mean_fidelity:.6e} ± {std_fidelity:.6e}")
    print(f"Min fidelity: {min_fidelity:.6e}")
    print(f"Max fidelity: {max_fidelity:.6e}")
    print(f"Fidelity range: {max_fidelity - min_fidelity:.6e}")
    print(f"Loss of fidelity: {1 - mean_fidelity:.6e}")
    
    # Show first 10 individual fidelities to see the pattern
    print(f"\nFirst 10 individual fidelities:")
    for i in range(min(10, len(fidelities))):
        print(f"  State {i+1}: {fidelities[i]:.8e}")
    
    # Verify Haar randomness by checking statistical properties
    print("\n--- Verifying Haar Randomness ---")
    
    # Test 1: Generate multiple states and check if they're uniformly distributed
    num_samples = 1000
    overlaps = []
    
    for i in range(num_samples):
        state1, _, _ = haar_random_state_generator(2, 0.0, seed=i)  # No noise, 2 qubits for speed
        state2, _, _ = haar_random_state_generator(2, 0.0, seed=i+num_samples)
        
        # Convert to state vectors if they're density matrices
        if state1.ndim > 1:
            # Extract the state vector from density matrix (assuming it's pure)
            eigenvals, eigenvecs = np.linalg.eigh(state1)
            state1 = eigenvecs[:, np.argmax(eigenvals)]
        if state2.ndim > 1:
            eigenvals, eigenvecs = np.linalg.eigh(state2)
            state2 = eigenvecs[:, np.argmax(eigenvals)]
        
        overlap = np.abs(np.vdot(state1, state2))**2
        overlaps.append(overlap)
    
    mean_overlap = np.mean(overlaps)
    expected_overlap = 1 / (2**2)  # For 2 qubits: 1/2^n
    
    print(f"Mean overlap between random states: {mean_overlap:.6f}")
    print(f"Expected overlap for Haar random states: {expected_overlap:.6f}")
    print(f"Difference: {abs(mean_overlap - expected_overlap):.6f}")
    
    if abs(mean_overlap - expected_overlap) < 0.05:
        print("✓ States appear to be Haar random!")
    else:
        print("✗ States may not be properly Haar random")
    
    # Test 2: Check unitary properties
    U = generate_haar_random_unitary(2, seed=42)
    print(f"\nUnitary matrix properties:")
    print(f"U†U - I norm: {np.linalg.norm(U.conj().T @ U - np.eye(4)):.2e}")
    print(f"Det(U): {np.linalg.det(U):.6f} (should be close to 1 in magnitude)")
    print(f"|Det(U)|: {np.abs(np.linalg.det(U)):.6f} (should be 1)")
    
    if np.linalg.norm(U.conj().T @ U - np.eye(4)) < 1e-10 and np.abs(np.abs(np.linalg.det(U)) - 1) < 1e-10:
        print("✓ Generated matrix is a proper unitary!")
    else:
        print("✗ Generated matrix is not properly unitary")
    
