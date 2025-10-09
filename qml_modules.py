# Import required libraries
import pennylane as qml
from pennylane import numpy as np
import torch
import torch.nn as nn
#from functools import partial, wraps
from superop import SuperOpTools
from scipy.linalg import expm, logm
import scipy.stats
from typing import Union
#import pickle


class post_process:
    def __call__(self, expval):
        c1 = expval
        c2 = 1 - expval**2
        c3 = 2*expval**3 -2*expval
        c4 = 1 - expval**2 - 3*expval**3 + 3*expval

        return c1 - c2**2/(c3**2-c2*c4)*(np.sqrt(3*c3**2-2*c2*c4)-c3)


class CustomSigmoid(nn.Module):
    """A custom sigmoid activation function with adjustable dilation and vertical translation."""
    def __init__(self, dilation=2, vert_translation=-1):
        super().__init__()
        self.dilation = dilation
        self.vert_translation = vert_translation
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        return self.sigmoid(x)*self.dilation + self.vert_translation
        
        
class ShotNoise(nn.Module):
    """A stochastic layer to simulate shot noise in quantum circuits."""
    def __init__(self, num_shots, device):
        super().__init__()
        self.num_shots = num_shots
        self.device = device
        
    def forward(self, x):
        #x = x[:,::2] - x[:,1::2]
        p_positive_eigvals = (x+1)/2
        variance = 4*p_positive_eigvals*(1-p_positive_eigvals)/self.num_shots
        expval_with_shotnoise = x + variance**0.5*torch.randn(x.shape[0], x.shape[1]).to(self.device)
        
        return expval_with_shotnoise.clamp(min=-1,max=1)

class ClampLinear(nn.Module):
    """A linear layer with clamping to specified bounds."""
    def __init__(self, input_features, output_features, lower_bound, upper_bound):
        super().__init__()
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.Linear = nn.Linear(input_features, output_features)
        
    def forward(self, x):
        return self.Linear(x).clamp(min=self.lower_bound, max=self.upper_bound)


        

# Quantum neural network
class HybridModel(nn.Module):
    
    """
    A hybrid quantum-classical neural network model.
    This model uses a quantum circuit for feature extraction and a classical neural network for classification.
    The quantum circuit is implemented using PennyLane, and the classical part is a simple feedforward neural network.
    The quantum circuit uses amplitude encoding and trainable parameters for quantum gates.
    The model can be configured with various noise models to simulate realistic quantum operations.
    Args:
        dev: The PennyLane device to set the mode of circuit execution.
        device: The PyTorch device (CPU or GPU) for the classical simulation.
        num_qubits: The number of qubits in the quantum circuit.
        weight_shapes: A dictionary defining the shapes of the trainable weights in the quantum circuit.
        noise_model: A string specifying the type of noise model to apply to the quantum circuit.
        **kwargs: Additional keyword arguments for configuring the quantum circuit and noise model.
    Attributes:
        dev: The PennyLane device for the quantum circuit.
        device: The PyTorch device for the classical part.
        num_qubits: The number of qubits in the quantum circuit.
        shot_noise: An instance of ShotNoise to simulate shot noise in the quantum circuit.
        classical: A linear layer for the classical part of the model.
        quantum_circuit: A PennyLane quantum node representing the quantum circuit.
        qnode_test: A test quantum node for debugging purposes.
        quantum: A PennyLane TorchLayer that wraps the quantum circuit for integration with PyTorch.
        lanczos_expand: A post-processing layer to expand the quantum output using the Lanczos method.
        clamped_classical: A linear layer with clamping to specified bounds.
        sigmoid_activation: A custom sigmoid activation function with adjustable dilation and vertical translation.
    """
    
    def __init__(self, 
                 dev, 
                 device: torch.device, 
                 num_qubits: int, 
                 weight_shapes: dict[str, tuple], 
                 noise_model: Union[str, None] = None, 
                 **kwargs):
        super().__init__()
        
        self.dev = dev
        self.device = device
        self.num_qubits = num_qubits
        self.shot_noise = ShotNoise(num_shots=10000, device=device)
        self.classical = nn.Linear(num_qubits, 10)#.to(device)
        #self.clamped_classical = ClampLinear(num_qubits, 10, -1e8, 1e8)
        #self.sigmoid_activation = CustomSigmoid()
        
        
        @qml.qnode(dev)#, interface='torch')#,diff_method="adjoint")
        def quantum_circuit(inputs, weights):
            """Quantum circuit with amplitude encoding"""
            # Amplitude encoding (critical change)
            #qml.AmplitudeEmbedding(
            #    features=inputs,
            #    wires=range(self.num_qubits),
            #    normalize=True  # Already normalized in preprocessing
            #)
            
            #coeffs_real = inputs[:,::2]
            #coeffs_imag = inputs[:,1::2]
            #coeffs = coeffs_real + 1j*coeffs_imag
            
            qml.StatePrep(
                state=inputs,
                #state=coeffs,
                wires=range(self.num_qubits),
                normalize=True
            )
            
            #bases = torch.Tensor([[int(x) for x in bin(i)[2:]] for i in range(2**num_qubits)])
            #bases.repeat(inputs.shape[0], len(bases), num_qubits)
            #qml.Superposition(coeffs=coeffs,
            #                  bases=bases,
            #                  wires=range(self.num_qubits),
            #                  work_wire=num_qubits)
            
            
            
            #always-on Z rotation parameters:
            #bases = qml.ops.qubit.special_unitary.pauli_basis_strings(self.num_qubits)
            #
            #coefficients = np.zeros(4**self.num_qubits)
            #
            #for idx, basis in enumerate(bases):
            #    Z_count = basis.count('Z')
            #    I_count = basis.count('I')
            #    if I_count + Z_count == self.num_qubits:
            #        if Z_count == 1:
            #            coefficients[idx] += kwargs['Z']
            
            

            # Trainable layers (batch-agnostic)
            #for layer in range(int(weights.shape[0]/3)):
            #    for qubit in range(self.num_qubits):
            #        qml.RZ(weights[layer*3, qubit], wires=qubit)
            #        qml.RY(weights[layer*3+1, qubit], wires=qubit)
            #        qml.RZ(weights[layer*3+2, qubit], wires=qubit)
            #        
            #    # Entanglement remains the same across batches
            #    for qubit in range(0, self.num_qubits, 2):
            #        qml.CZ(wires=[qubit, (qubit+1)%self.num_qubits])
            #    
            #    for qubit in range(1, self.num_qubits, 2):
            #        qml.CZ(wires=[qubit, (qubit+1)%self.num_qubits])
            
            @qml.for_loop(0, self.num_qubits, 2)
            def entangling_gate_even_qubits(qubit):
                qml.CZ(wires=[qubit, (qubit+1)%self.num_qubits])

            @qml.for_loop(1, self.num_qubits, 2)
            def entangling_gate_odd_qubits(qubit):
                qml.CZ(wires=[qubit, (qubit+1)%self.num_qubits])

            
            num_layers = int(weights.shape[0]/3)
            for layer in range(num_layers):
                @qml.for_loop(0, self.num_qubits)
                def single_qubit_rotation(qubit):
                    qml.RZ(weights[layer*3, qubit], wires=qubit)
                    qml.RY(weights[layer*3+1, qubit], wires=qubit)
                    qml.RZ(weights[layer*3+2, qubit], wires=qubit)
                
                single_qubit_rotation()
                entangling_gate_even_qubits()
                entangling_gate_odd_qubits()
                
            # Return measurements for all batches
            #print(qml.state())
            return [qml.expval(qml.PauliZ(wires=q)) for q in range(self.num_qubits)]

            #return [qml.probs(op=qml.PauliZ(wires=q)) for q in range(self.num_qubits)]
            
            #return [qml.expval(qml.Hermitian(ZZ, wires=[q,q])) for q in range(num_qubits)]
        
        
        @qml.qnode(dev)#, interface='torch')#,diff_method="adjoint")
        def repeated_encoding_circuit(inputs, weights):
            """Quantum circuit with amplitude encoding"""
            # Amplitude encoding (critical change)
            #qml.AmplitudeEmbedding(
            #    features=inputs,
            #    wires=range(self.num_qubits),
            #    normalize=True  # Already normalized in preprocessing
            #)
            
            #coeffs_real = inputs[:,::2]
            #coeffs_imag = inputs[:,1::2]
            #coeffs = coeffs_real + 1j*coeffs_imag
            
            if self.num_qubits >= 2:
                @qml.for_loop(0, self.num_qubits, 2)
                def entangling_gate_even_qubits(qubit):
                    qml.CZ(wires=[qubit, (qubit+1)%self.num_qubits])
            else:
                def entangling_gate_even_qubits(qubit):
                    pass
            
            if self.num_qubits > 2:
                @qml.for_loop(1, self.num_qubits, 2)
                def entangling_gate_odd_qubits(qubit):
                    qml.CZ(wires=[qubit, (qubit+1)%self.num_qubits])
            else:
                def entangling_gate_odd_qubits(qubit):
                    pass
            
            
            @qml.for_loop(0, self.num_qubits)
            def single_qubit_rotation_1(qubit):
                qml.RZ(weights[0, qubit], wires=qubit)
                qml.RY(weights[1, qubit], wires=qubit)
                qml.RZ(weights[2, qubit], wires=qubit)
            
            @qml.for_loop(0, self.num_qubits)
            def single_qubit_rotation_2(qubit):
                qml.RZ(weights[3, qubit], wires=qubit)
                qml.RY(weights[4, qubit], wires=qubit)
                qml.RZ(weights[5, qubit], wires=qubit)
                
            single_qubit_rotation_1()
            if self.num_qubits >= 2:
                entangling_gate_even_qubits()
            if self.num_qubits > 2:
                entangling_gate_odd_qubits()
            single_qubit_rotation_2()
            if self.num_qubits >= 2:
                entangling_gate_even_qubits()
            if self.num_qubits > 2:
                entangling_gate_odd_qubits()
                
            num_layers = int(weights.shape[0]/3/2)
            for layer in range(1, num_layers):
                
                qml.AngleEmbedding(
                features=inputs,
                wires=range(self.num_qubits),
                rotation='X',
                )
                
                @qml.for_loop(0, self.num_qubits)
                def single_qubit_rotation_1(qubit):
                    qml.RZ(weights[2*layer*3, qubit], wires=qubit)
                    qml.RY(weights[2*layer*3+1, qubit], wires=qubit)
                    qml.RZ(weights[2*layer*3+2, qubit], wires=qubit)
                
                @qml.for_loop(0, self.num_qubits)
                def single_qubit_rotation_2(qubit):
                    qml.RZ(weights[2*layer*3+3, qubit], wires=qubit)
                    qml.RY(weights[2*layer*3+4, qubit], wires=qubit)
                    qml.RZ(weights[2*layer*3+5, qubit], wires=qubit)
                    
                single_qubit_rotation_1()
                if self.num_qubits >= 2:
                    entangling_gate_even_qubits()
                if self.num_qubits > 2:
                    entangling_gate_odd_qubits()
                single_qubit_rotation_2()
                if self.num_qubits >= 2:
                    entangling_gate_even_qubits()
                if self.num_qubits > 2:
                    entangling_gate_odd_qubits()
            
            # Return measurements for all batches
            #print(qml.state())
            return [qml.expval(qml.PauliZ(wires=0))]
        
        @qml.qnode(dev)
        def angle_embedding_vqc(inputs, weights):
            """Quantum circuit with amplitude encoding"""
            
            @qml.for_loop(0, self.num_qubits)
            def parllel_angle_embedding(qubit):
                qml.AngleEmbedding(
                    features=inputs,
                    wires=[qubit],
                    rotation='X',
                    )
            
            @qml.for_loop(0, self.num_qubits, 2)
            def entangling_gate_even_qubits(qubit):
                qml.CZ(wires=[qubit, (qubit+1)%self.num_qubits])

            @qml.for_loop(1, self.num_qubits, 2)
            def entangling_gate_odd_qubits(qubit):
                qml.CZ(wires=[qubit, (qubit+1)%self.num_qubits])

            
            num_layers = int(weights.shape[0]/3)
            for layer in range(num_layers):
                @qml.for_loop(0, self.num_qubits)
                def single_qubit_rotation(qubit):
                    qml.RZ(weights[layer*3, qubit], wires=qubit)
                    qml.RY(weights[layer*3+1, qubit], wires=qubit)
                    qml.RZ(weights[layer*3+2, qubit], wires=qubit)
                
                single_qubit_rotation()
                entangling_gate_even_qubits()
                entangling_gate_odd_qubits()
                
                if layer == 49:
                    parllel_angle_embedding()

            return [qml.expval(qml.PauliZ(wires=0))]
        
        @qml.qnode(dev)
        def repeated_amplitude_embedding_vqc(inputs, weights):
            """Quantum circuit with amplitude encoding"""
                        
            @qml.for_loop(0, self.num_qubits, 2)
            def entangling_gate_even_qubits(qubit):
                qml.CZ(wires=[qubit, (qubit+1)%self.num_qubits])

            @qml.for_loop(1, self.num_qubits, 2)
            def entangling_gate_odd_qubits(qubit):
                qml.CZ(wires=[qubit, (qubit+1)%self.num_qubits])

            
            num_layers = int(weights.shape[0]/3)
            for layer in range(num_layers):
                @qml.for_loop(0, self.num_qubits)
                def single_qubit_rotation(qubit):
                    qml.RZ(weights[layer*3, qubit], wires=qubit)
                    qml.RY(weights[layer*3+1, qubit], wires=qubit)
                    qml.RZ(weights[layer*3+2, qubit], wires=qubit)
                
                single_qubit_rotation()
                entangling_gate_even_qubits()
                entangling_gate_odd_qubits()
                
                if layer % 10 == 0:
                    qml.StatePrep(
                        state=inputs,
                        wires=range(self.num_qubits),
                        normalize=True
                        )

            return [qml.expval(qml.PauliZ(wires=0))]
        
        
        for var_name, var_value in kwargs.items():
            setattr(self, var_name, var_value)
            
        if noise_model == 'depolarising':
            noise_model = self.gen_depolarising_noise(p_depolarising = kwargs['p_depolarising'])
            #noise_model = self.gen_pauli_noises(p_depolarising = kwargs['p_depolarising'])
            
        elif noise_model == 'extended depolarising on two qubit gates':
            noise_model = self.gen_depolarising_noise(single_qubit_only=False,
                                                      p_depolarising = kwargs['p_depolarising'],
                                                      p_depolarising_two_qubit = kwargs['p_depolarising_two_qubit'],
                                                      phi_IX = kwargs['phi_IX'],
                                                      phi_IY = kwargs['phi_IY'],
                                                      phi_IZ = kwargs['phi_IZ'],
                                                      phi_XI = kwargs['phi_XI'],
                                                      phi_XX = kwargs['phi_XX'],
                                                      phi_XY = kwargs['phi_XY'],
                                                      phi_XZ = kwargs['phi_XZ'],
                                                      phi_YI = kwargs['phi_YI'],
                                                      phi_YX = kwargs['phi_YX'],
                                                      phi_YY = kwargs['phi_YY'],
                                                      phi_YZ = kwargs['phi_YZ'],
                                                      phi_ZI = kwargs['phi_ZI'],
                                                      phi_ZX = kwargs['phi_ZX'],
                                                      phi_ZY = kwargs['phi_ZY'],
                                                      phi_ZZ = kwargs['phi_ZZ'])
        
        elif noise_model == 'depol and damping':
            noise_model = self.gen_depolarising_noise(single_qubit_only='depol and damping',
                                                      p_depolarising = kwargs['p_depolarising'],
                                                      p_damping = kwargs['p_damping'])
        
        elif noise_model == 'depol and generalised damping':
            noise_model = self.gen_depolarising_noise(single_qubit_only='depol and generalised damping',
                                                      p_depolarising = kwargs['p_depolarising'],
                                                      gamma = kwargs['gamma'],
                                                      p_damping = kwargs['p_damping'])
        elif noise_model == 'pauli noises':
            noise_model = self.gen_pauli_noises(pX_single = kwargs['pX_single'],
                                                pY_single = kwargs['pY_single'],
                                                pZ_single = kwargs['pZ_single'],
                                                pX_double = kwargs['pX_double'],
                                                pY_double = kwargs['pX_double'],
                                                pZ_double = kwargs['pX_double'])
        
        elif noise_model == 'gaussian over rotation':
            noise_model = self.gen_gaussian_rotation_noise(sigma=kwargs['sigma'],
                                                           mean=kwargs['mean'],
                                                           num_channels=kwargs['num_channels'])
        
        elif noise_model == 'amplitude damping':
            noise_model = self.gen_amplitude_damping_noise(p_damping=kwargs['p_damping'])
        
        elif noise_model == 'generalised damping':
            noise_model = self.gen_generalised_damping(gamma=kwargs['gamma'], p_damping=kwargs['p_damping'])
        
        elif noise_model == 'thermal':
            noise_model = self.gen_thermal_noise(p_e=kwargs['pe'], t1=kwargs['t1'], t2=kwargs['t2'], tg=kwargs['tg'])
            
        elif noise_model == 'always on':
            noise_model = self.gen_always_on_noise(Z=kwargs['Z'])
                
        
        if noise_model is not None:
            if 'repeated_encoding' in kwargs:
                repeated_encoding_circuit = qml.add_noise(repeated_encoding_circuit, noise_model=noise_model)
            elif 'single_parallel_repeated_encoding' in kwargs:
                angle_embedding_vqc = qml.add_noise(angle_embedding_vqc, noise_model=noise_model)
            elif 'repeated_amplitude_embedding' in kwargs:
                repeated_amplitude_embedding_vqc = qml.add_noise(repeated_amplitude_embedding_vqc, noise_model=noise_model)
            else:
                quantum_circuit = qml.add_noise(quantum_circuit, noise_model=noise_model)
        
        if 'repeated_encoding' in kwargs:
            print('using repeated encoding circuit')
            self.qnode_test = repeated_encoding_circuit
            self.quantum = qml.qnn.TorchLayer(repeated_encoding_circuit, weight_shapes)
        elif 'single_parallel_repeated_encoding' in kwargs:
            print('using single qubit parallel angle embedding circuit')
            self.qnode_test = angle_embedding_vqc
            self.quantum = qml.qnn.TorchLayer(angle_embedding_vqc, weight_shapes)
        elif 'repeated_amplitude_embedding' in kwargs:
            print('using repeated amplitude embedding circuit')
            self.qnode_test = repeated_amplitude_embedding_vqc
            self.quantum = qml.qnn.TorchLayer(repeated_amplitude_embedding_vqc, weight_shapes)
        else:
            print('using normal circuit')
            self.qnode_test = quantum_circuit
            self.quantum = qml.qnn.TorchLayer(quantum_circuit, weight_shapes)
        
        #self.lanczos_expand = post_process()
        #self.classical = nn.Linear(num_qubits, 10)#.to(device)  # 8 quantum outputs -> 10 classes
    
    @property
    def device(self):
        return self._device
    
    @device.setter
    def device(self, new_value):
        if not isinstance(new_value, torch.device):
            raise ValueError("device must be a torch.device")
        self._device = new_value
        
        
    def forward(self, x):
        #print('entering forward function', flush=True)
        #print(f'input image vec: {x} with norm {torch.linalg.norm(x)}', flush=True)
        x = self.quantum(x)
        #print(f'after quantum layer: {x}', flush=True)
        
        x = self.shot_noise(x)
        #print(f'after shot noise layer: {x}', flush=True)
        
        #x = self.lanczos_expand(x)
        #return nn.functional.softmax(x, dim=1)
        
        x = self.classical(x)
        #print(f'after classical layer: {x}', flush=True)
        #print('exiting forward function', flush=True)
        
        return x
        #return self.classical(x)#.clamp(min=-1e8,max=1e8)
        #return self.clamped_classical(x)
        
        
        #return self.quantum(x)
        #return self.sigmoid_activation(x)
    
    def quantum_forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        return self.quantum(x).item()
    
    
    def gen_depolarising_noise(self, single_qubit_only: bool|str = True, **kwargs) -> qml.NoiseModel:
        
        @qml.BooleanFn
        def single_qubit_ops_condition(op):
            #return isinstance(op, qml.RX) or isinstance(op, qml.RY) or isinstance(op, qml.RZ)# or isinstance(op, qml.CZ)
            return isinstance(op, qml.RY)
        
        depol_err_single = qml.noise.partial_wires(qml.DepolarizingChannel, kwargs['p_depolarising'])
        
        def depol_and_damping(op, **params):
            depolarising_channel_kraus_reps = qml.DepolarizingChannel.compute_kraus_matrices(params['p_depolarising'])
            
            if single_qubit_only == 'depol and damping':
                amplitude_damping_channel_kraus_reps = qml.AmplitudeDamping.compute_kraus_matrices(params['p_damping'])
            
            elif single_qubit_only == 'depol and generalised damping':
                amplitude_damping_channel_kraus_reps = qml.GeneralizedAmplitudeDamping.compute_kraus_matrices(gamma=params['gamma'], p=params['p_damping'])
            
            depolarising_channel_superop_reps = SuperOpTools.kraus2superop(depolarising_channel_kraus_reps)
            amplitude_damping_channel_superop_reps = SuperOpTools.kraus2superop(amplitude_damping_channel_kraus_reps)
            
            net_channel_superop = expm(logm(depolarising_channel_superop_reps) + logm(amplitude_damping_channel_superop_reps))
            kraus_list = SuperOpTools.superop2kraus(net_channel_superop)
            qml.QubitChannel(kraus_list, wires=op.wires)
            
        
        @qml.BooleanFn
        def two_qubit_ops_condition(op):
            return isinstance(op, qml.CZ)
        
        def two_qubit_noise(op, **params):
            #ZZ_channel = np.sqrt(params['p_ZZ'])*qml.IsingZZ.compute_matrix(params['phi_ZZ'])
            #theta = [params['phi_IX'], params['phi_IY'], params['phi_IZ'], 
            #         params['phi_XI'], params['phi_XX'], params['phi_XY'],params['phi_XZ'],
            #         params['phi_YI'], params['phi_YX'], params['phi_YY'],params['phi_YZ'], 
            #         params['phi_ZI'], params['phi_ZX'], params['phi_ZY'],params['phi_ZZ']]
            #unitary_channel = np.sqrt(params['p_2q_unitary'])*qml.SpecialUnitary.compute_matrix(theta,
            #                                                                                    num_wires=2)
            #
            #depolarising_channel_kraus_matrices = qml.DepolarizingChannel.compute_kraus_matrices(params['p_depolarising'])
            #depolarising_channel_kraus_matrices_two_qubit = [np.sqrt(1-params['p_2q_unitary'])*qml.math.dot(depolarising_channel_kraus_matrices[i],
            #                                                                                                depolarising_channel_kraus_matrices[j]) 
            #                                                 for i in range(4) 
            #                                                 for j in range(4)]
            #Kraus_list = [unitary_channel] + depolarising_channel_kraus_matrices_two_qubit
            #
            #qml.QubitChannel(Kraus_list, wires=op.wires)
            
            
            
            ####################################################################### use either the top or the bottom codes to impplement
            #theta = [params['phi_IX'], params['phi_IY'], params['phi_IZ'], 
            #         params['phi_XI'], params['phi_XX'], params['phi_XY'], params['phi_XZ'],
            #         params['phi_YI'], params['phi_YX'], params['phi_YY'], params['phi_YZ'], 
            #         params['phi_ZI'], params['phi_ZX'], params['phi_ZY'], params['phi_ZZ']]
            
            #unitary_channel_adjoint_rep = qml.SpecialUnitary.compute_matrix(theta, num_wires=2)
            #unitary_channel_superop_rep = SuperOpTools.kraus2superop([unitary_channel_adjoint_rep])
            #unitary_channel_superop_generator = logm(unitary_channel_superop_rep)
            
            
            #depolarising_channel_adjoint_reps = qml.DepolarizingChannel.compute_kraus_matrices(params['p_depolarising_two_qubit'])
            #depolarising_channel_adjoint_reps_two_qubit = [qml.math.dot(depolarising_channel_adjoint_reps[i], 
            #                                                            depolarising_channel_adjoint_reps[j]) 
            #                                                 for i in range(4) 
            #                                                 for j in range(4)]
            #depolarising_channel_superop_reps = SuperOpTools.kraus2superop(depolarising_channel_adjoint_reps_two_qubit)
            #depolarising_channel_superop_generator = logm(depolarising_channel_superop_reps)
            
            #net_channel_superop = expm(unitary_channel_superop_generator + depolarising_channel_superop_generator)
            
            #Kraus_list = SuperOpTools.superop2kraus(net_channel_superop)            
            #qml.QubitChannel(Kraus_list, wires=op.wires)
            
            
            #######################################################################
            theta = [params['phi_IX'], params['phi_IY'], params['phi_IZ'], 
                     params['phi_XI'], params['phi_XX'], params['phi_XY'], params['phi_XZ'],
                     params['phi_YI'], params['phi_YX'], params['phi_YY'], params['phi_YZ'], 
                     params['phi_ZI'], params['phi_ZX'], params['phi_ZY'], params['phi_ZZ']]
            
            unitary_channel_adjoint_rep = [qml.SpecialUnitary.compute_matrix(theta, num_wires=2)]
            
            #print(op.wires)
            crosstalk_nearest_neighbour_wires_1 = sorted([(op.wires[0]-1)%self.num_qubits, op.wires[0]])
            #print(crosstalk_nearest_neighbour_wires_1)
            crosstalk_nearest_neighbour_wires_2 = sorted([op.wires[1], (op.wires[1]+1)%self.num_qubits])
            qml.QubitChannel(unitary_channel_adjoint_rep, wires=crosstalk_nearest_neighbour_wires_1)
            qml.QubitChannel(unitary_channel_adjoint_rep, wires=crosstalk_nearest_neighbour_wires_2)
            
            depolarising_channel_adjoint_reps = qml.DepolarizingChannel.compute_kraus_matrices(params['p_depolarising_two_qubit'])
            depolarising_channel_adjoint_reps_two_qubit = [qml.math.kron(depolarising_channel_adjoint_reps[i], 
                                                                         depolarising_channel_adjoint_reps[j]) 
                                                           for i in range(4) 
                                                           for j in range(4)]
            
            #print(depolarising_channel_adjoint_reps_two_qubit)
            qml.QubitChannel(depolarising_channel_adjoint_reps_two_qubit, wires=op.wires)
            
        #metadata = dict(t1=0.02, t2=0.03, tg=0.001)  # times unit: sec
        
        if single_qubit_only:
            noise_model = qml.NoiseModel(
                {single_qubit_ops_condition: depol_err_single,}
                )
        
        elif single_qubit_only == 'depol and damping' or single_qubit_only == 'depol and generalised damping':
            metadata = kwargs.copy()
            noise_model = qml.NoiseModel(
                {single_qubit_ops_condition: depol_and_damping,}, **metadata
                )
        
        else:
            metadata = kwargs.copy()
            if metadata['p_depolarising'] == 0:
                noise_model = qml.NoiseModel(
                    {two_qubit_ops_condition: two_qubit_noise}, **metadata
                    )
            else:
                noise_model = qml.NoiseModel(
                    {single_qubit_ops_condition: depol_err_single,
                    two_qubit_ops_condition: two_qubit_noise}, **metadata
                    )
        
        return noise_model
    
    def gen_pauli_noises(self, **kwargs) -> qml.NoiseModel:
        
        @qml.BooleanFn
        def single_qubit_ops_condition(op):
            #return isinstance(op, qml.RX) or isinstance(op, qml.RY) or isinstance(op, qml.RZ)# or isinstance(op, qml.CZ)
            return isinstance(op, qml.RY)
        
        #two_qubit_ops_condition = qml.noise.op_eq(qml.CZ)
        
        #E0 = np.sqrt(1-kwargs['pX_single']-kwargs['pY_single']-kwargs['pZ_single'])*qml.Identity.compute_matrix()
        #E1 = np.sqrt(kwargs['pX_single'])*qml.PauliX.compute_matrix()
        #E2 = np.sqrt(kwargs['py_single'])*qml.PauliX.compute_matrix()
        #E3 = np.sqrt(kwargs['pz_single'])*qml.PauliX.compute_matrix()
        
        E0 = np.sqrt(1-kwargs['p_depolarising'])*qml.Identity.compute_matrix()
        E1 = np.sqrt(kwargs['p_depolarising']/3)*qml.PauliX.compute_matrix()
        E2 = np.sqrt(kwargs['p_depolarising']/3)*qml.PauliY.compute_matrix()
        E3 = np.sqrt(kwargs['p_depolarising']/3)*qml.PauliZ.compute_matrix()
        kraus_list = [E0,E1,E2,E3]
        pauli_err_single = qml.noise.partial_wires(qml.QubitChannel, kraus_list)
        
        
        #K0 = np.sqrt(1-kwargs['pX_double']-kwargs['pY_double']-kwargs['pZ_double'])*qml.Identity.compute_matrix()
        #K1 = np.sqrt(kwargs['pX_double'])*qml.PauliX.compute_matrix()
        #K2 = np.sqrt(kwargs['pY_double'])*qml.PauliX.compute_matrix()
        #K3 = np.sqrt(kwargs['pZ_double'])*qml.PauliX.compute_matrix()
        
        #kraus_list = [K0,K1,K2,K3]
        #pauli_err_double = qml.noise.partial_wires(qml.QubitChannel, kraus_list)
        
        
        noise_model = qml.NoiseModel(
            {single_qubit_ops_condition: pauli_err_single,
             #two_qubit_ops_condition: pauli_err_double}}
            })
        
        return noise_model
    
    def gen_gaussian_rotation_noise(self, **kwargs) -> qml.NoiseModel:
        
        @qml.BooleanFn
        def single_qubit_ops_condition(op):
            #return isinstance(op, qml.Rot)
            return isinstance(op, qml.RX) or isinstance(op, qml.RY) or isinstance(op, qml.RZ)# or isinstance(op, qml.CZ)
        
        def gaussian_rotation(op, **distribution_params):
            sigma = distribution_params['sigma']
            mean = distribution_params['mean']
            num_channels = distribution_params['num_channels']
            if num_channels % 2 == 0:
                raise ValueError("num_meshes must be odd")
            
            cdf_bounds = np.linspace(-np.pi, np.pi, num_channels+1)
            cdf_values = [scipy.stats.norm(0, sigma).cdf(cdf_bounds[1:][i]) - scipy.stats.norm(0, sigma).cdf(cdf_bounds[:-1][i]) 
                          for i in range(len(cdf_bounds)-2)]
            cdf_values_normalised = cdf_values/np.sum(cdf_values)
            
            rotation_angles = mean + (cdf_bounds[1:] + cdf_bounds[:-1])/2
            
            if isinstance(op, qml.RX):
                kraus_ops_list = [np.sqrt(cdf_values_normalised[i])*qml.RZ.compute_matrix(rotation_angles[i]) 
                                  for i in range(len(cdf_values_normalised))]
                
                #kraus_ops_list = [np.sqrt(cdf_values_normalised[i])*qml.SpecialUnitary.compute_matrix([op.parameters[0]/2,
                #                                                                                       0,
                #                                                                                       rotation_angles[i]/2
                #                                                                                       ], 
                #                                                                                      num_wires=1) 
                #                  for i in range(len(cdf_values_normalised))]
                
                
            elif isinstance(op, qml.RY):
                kraus_ops_list = [np.sqrt(cdf_values_normalised[i])*qml.RZ.compute_matrix(rotation_angles[i]) 
                                  for i in range(len(cdf_values_normalised))]
            elif isinstance(op, qml.RZ):
                kraus_ops_list = [np.sqrt(cdf_values_normalised[i])*qml.RZ.compute_matrix(rotation_angles[i]) 
                                  for i in range(len(cdf_values_normalised))]
            #elif isinstance(op, qml.Rot):
            #    kraus_ops_list = [np.sqrt(cdf_values_normalised[i])*qml.RZ.compute_matrix(rotation_angles[i]) 
            #                      for i in range(len(cdf_values_normalised))]
            
            
            #qml.noise.partial_wires(qml.QubitChannel, kraus_ops_list, wires=op.wires)
            qml.QubitChannel(kraus_ops_list, wires=op.wires)


        metadata = kwargs.copy()
        noise_model = qml.NoiseModel(
            {single_qubit_ops_condition: gaussian_rotation}, **metadata
            )
        
        return noise_model
        
        
    def gen_amplitude_damping_noise(self, **kwargs) -> qml.NoiseModel:
            
            condition = qml.noise.wires_in(list(range(self.num_qubits)))
            
            
            def amplitude_damping(op, **decay_rates):
                for wire in range(self.num_qubits):
                    qml.AmplitudeDamping(
                        decay_rates['p_damping'], 
                        wire
                        )
            
            metadata = kwargs.copy()
            noise_model = qml.NoiseModel(
                {condition: amplitude_damping}, **metadata
                )
            
            return noise_model
    
    def gen_generalised_damping(self, **kwargs) -> qml.NoiseModel:
            
            condition = qml.noise.wires_in(list(range(self.num_qubits)))
            
            
            def generalised_damping(op, **decay_rates):
                for wire in range(self.num_qubits):
                    qml.GeneralizedAmplitudeDamping(
                        gamma=decay_rates['gamma'],
                        p=decay_rates['p_damping'], 
                        wires=wire
                        )
            
            metadata = kwargs.copy()
            noise_model = qml.NoiseModel(
                {condition: generalised_damping}, **metadata
                )
            
            return noise_model
            
    #def gen_thermal_noise(self, t1, t2, tg):
    #    
    #    condition = qml.noise.wires_in(list(range(self.num_qubits)))
    #    
    #    def thermal_noise(op):
    #        for wire in range(self.num_qubits):
    #            qml.noise.partial_wires(qml.ThermalRelaxationError, 0.1, t1, t2, tg, wire)
    #    
    #    noise_model = qml.NoiseModel(
    #        {condition: thermal_noise}
    #        )
    #    
    #    return noise_model
    
    def gen_thermal_noise(self, **kwargs) -> qml.NoiseModel:
        
        condition = qml.noise.wires_in(list(range(self.num_qubits)))
        
        def thermal_noise(op, **decay_rates):
            for wire in range(self.num_qubits):
                #qml.noise.partial_wires(qml.ThermalRelaxationError, 
                #                        0.1, 
                #                        decay_rates['t1'], 
                #                        decay_rates['t2'], 
                #                        decay_rates['tg'],
                #                        wire)
                qml.ThermalRelaxationError(
                    decay_rates['p_e'], 
                    decay_rates['t1'], 
                    decay_rates['t2'], 
                    decay_rates['tg'],
                    wires=wire
                    )
                
        metadata = kwargs.copy()
        noise_model = qml.NoiseModel(
            {condition: thermal_noise}, **metadata
            )
        
        return noise_model
    
    def gen_always_on_noise(self, **kwargs) -> qml.NoiseModel:
        
        condition = qml.noise.op_eq(qml.CZ)
        
        def always_on_Z_and_ZZ(op, **noise_params):
            #qml.apply(op)
            
            #bases = qml.ops.qubit.special_unitary.pauli_basis_strings(self.num_qubits)
            #
            #coefficients = np.zeros(4**self.num_qubits)
            #
            #for idx, basis in enumerate(bases):
            #    Z_count = basis.count('Z')
            #    I_count = basis.count('I')
            #    if I_count + Z_count == self.num_qubits:
            #        if Z_count == 1:
            #            coefficients[idx] += noise_params['Z']
            #
            #qml.SpecialUnitary(coefficients, wires=list(range(self.num_qubits)))
            
            
            D0 = np.sqrt(1-noise_params['p_phase'])*np.eye(2)
            D1 = np.sqrt(noise_params['p_phase'])*np.array([[1,0],[0,-1]])
            A0 = np.array([[1, 0], [0, np.sqrt(1-noise_params['p_damping'])]])
            A1 = np.sqrt(noise_params['p_damping'])*np.array([[0,1],[0,0]])
            U = expm(-1j*noise_params['Z']*qml.PauliZ(wires=0).compute_matrix())
            
            scale_factor = np.trace(D0.conj().T@D0 + D1.conj().T@D1 + A0.conj().T@A0 + A1.conj().T@A1 + U.conj().T@U)/2
            
            D0 = D0/np.sqrt(scale_factor)
            D1 = D1/np.sqrt(scale_factor)
            A0 = A0/np.sqrt(scale_factor)
            A1 = A1/np.sqrt(scale_factor)
            U = U/np.sqrt(scale_factor)
            kraus_ops_list = [D0,D1,A0,A1,U]
            for qubit in range(self.num_qubits):
                qml.QubitChannel(kraus_ops_list, wires=qubit)
            
            
            
            
        metadata = kwargs.copy()
        
        noise_model = qml.NoiseModel(
            {condition: always_on_Z_and_ZZ}, **metadata
            )
        
        return noise_model


#class Trainer(HybridModel):
#    def __init__(self, ):
#        super().__init__()
#        self.optimizer = optim.Adam(model.parameters(), lr=0.001)