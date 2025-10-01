# Import required libraries
from os import ftruncate
import pennylane as qml
from pennylane import numpy as np
from qml_essentials.model import Model
from qml_essentials.ansaetze import Circuit, Gates
#from qml_essentials.ansaetze import PulseInformation as pinfo
from qml_essentials.coefficients import FourierTree
from functools import partial
from typing import Union, Optional
#from util_funcs import is_commuting
#from copy import deepcopy


num_qubits = 2
dev = qml.device("default.qubit", wires=num_qubits)

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
    
    
    
    @qml.for_loop(0, num_qubits, 2)
    def entangling_gate_even_qubits(qubit):
        qml.CZ(wires=[qubit, (qubit+1)%num_qubits], id='constant Clifford')

    @qml.for_loop(1, num_qubits, 2)
    def entangling_gate_odd_qubits(qubit):
        qml.CZ(wires=[qubit, (qubit+1)%num_qubits], id='constant Clifford')
    
    
    @qml.for_loop(0, num_qubits)
    def single_qubit_rotation(qubit):
        qml.RZ(weights[0, qubit, 0], wires=qubit, id='variational')
        qml.RY(weights[0, qubit, 1], wires=qubit, id='variational')
        qml.RZ(weights[0, qubit, 2], wires=qubit, id='variational')
        
        qml.RZ(weights[0, qubit, 3], wires=qubit, id='variational')
        qml.RY(weights[0, qubit, 4], wires=qubit, id='variational')
        qml.RZ(weights[0, qubit, 5], wires=qubit, id='variational')
        
    single_qubit_rotation()
    if num_qubits >= 2:
        entangling_gate_even_qubits()
    if num_qubits > 2:
        entangling_gate_odd_qubits()
        
    num_layers = int(weights.shape[0]/3/2)
    for layer in range(1, num_layers):
        
        #sqml.AngleEmbedding(
        #sfeatures=inputs,
        #swires=range(num_qubits),
        #srotation='X',
        #)
        @qml.for_loop(0, num_qubits)
        def AngleEmbedding(qubit):
            qml.RX(inputs[qubit], wires=qubit, id='encoding')
        AngleEmbedding()
        
        @qml.for_loop(0, num_qubits)
        def single_qubit_rotation(qubit):
            qml.RZ(weights[layer, qubit, 2*layer*3], wires=qubit, id='variational')
            qml.RY(weights[layer, qubit, 2*layer*3+1], wires=qubit, id='variational')
            qml.RZ(weights[layer, qubit, 2*layer*3+2], wires=qubit, id='variational')
            
            qml.RZ(weights[layer, qubit, 2*layer*3+3], wires=qubit, id='variational')
            qml.RY(weights[layer, qubit, 2*layer*3+4], wires=qubit, id='variational')
            qml.RZ(weights[layer, qubit, 2*layer*3+5], wires=qubit, id='variational')
        single_qubit_rotation()
        
        if num_qubits >= 2:
            entangling_gate_even_qubits()
        if num_qubits > 2:
            entangling_gate_odd_qubits()
    
    # Return measurements for all batches
    #print(qml.state())
    return [qml.expval(qml.PauliZ(wires=0))]


class VariationalLayer(Circuit):
    @staticmethod
    def n_params_per_layer(n_qubits: int) -> int:
        return n_qubits * 6  # 6 parameters per qubit per layer

    #@staticmethod
    #def n_pulse_params_per_layer(n_qubits: int) -> int:
    #    n_params_RY = pinfo.num_params("RY")
    #    n_params_RZ = pinfo.num_params("RZ")
    #    n_params_CZ = pinfo.num_params("CZ")
#
    #    n_pulse_params = (n_params_RY + n_params_RZ) * n_qubits
    #    n_pulse_params += n_params_CZ * (n_qubits - 1)
#
    #    return n_pulse_params

    @staticmethod
    def get_control_indices(n_qubits: int) -> Optional[np.ndarray]:
        #return [n_qubits*6, n_qubits*6+1, 1]  # This means: start at 6, stop at 7, step 1
        return None  # No control parameters in this ansatz
        

    @staticmethod
    def build(weights: np.ndarray, n_qubits: int, **kwargs):
        
        @qml.for_loop(0, n_qubits)
        def single_qubit_rotation(qubit):
            qml.RZ(weights[0, qubit], wires=qubit)#, **kwargs)
            qml.RY(weights[1, qubit], wires=qubit)#, **kwargs)
            qml.RZ(weights[2, qubit], wires=qubit)#, **kwargs)

            qml.RZ(weights[3, qubit], wires=qubit)#, **kwargs)
            qml.RY(weights[4, qubit], wires=qubit)#, **kwargs)
            qml.RZ(weights[5, qubit], wires=qubit)#, **kwargs)
            
        single_qubit_rotation()
        if n_qubits >= 2:
            @qml.for_loop(0, n_qubits, 2)
            def entangling_gate_even_qubits(qubit):
                qml.CZ(wires=[qubit, (qubit+1)%n_qubits])#, **kwargs)
            entangling_gate_even_qubits()
        if n_qubits > 2:
            @qml.for_loop(1, n_qubits, 2)
            def entangling_gate_odd_qubits(qubit):
                qml.CZ(wires=[qubit, (qubit+1)%n_qubits])#, **kwargs)
            entangling_gate_odd_qubits()
            
def EncodingLayer(inputs: np.ndarray, n_qubits: int = 2, **kwargs):
    #print("EncodingLayer inputs shape:", np.shape(inputs))
    
    qml.RX(inputs[0], wires=0)
    #@qml.for_loop(0, n_qubits)
    #def AngleEmbedding(qubit):
    #    qml.RX(inputs[qubit], wires=qubit)#, **kwargs)
    #AngleEmbedding()



if __name__ == "__main__":
    # Example usage
    n_qubits = 1
    n_layers = 3
    Measuring_qubit_idx = 0
    encoding = partial(EncodingLayer, n_qubits=n_qubits)
    
    model = Model(
        n_qubits = n_qubits,
        n_layers = n_layers,
        circuit_type = VariationalLayer,
        data_reupload = True,  # Re-upload data at each layer
        encoding = encoding,
        output_qubit = Measuring_qubit_idx, 
        )

    #model.initialize_params(rng=np.random.default_rng(1001))
    custom_weights = np.random.randn(n_layers+1, 6, n_qubits)
    model.params = custom_weights
    print("Model parameters shape:", model.params.shape)
    fourier_tree = FourierTree(model)
    an_coeffs, an_freqs = fourier_tree.get_spectrum(force_mean=True)
    print("Analytic Fourier Coefficients:", an_coeffs)
    print("Analytic Fourier Frequencies:", an_freqs)

#class AnalyticFourierCoefficients:
#    def __init__(self, circuit: type[qml.qnode], num_qubits, layers):
#        self.circuit = circuit
#        self.num_qubits = num_qubits
#        self.layers = layers
#    
#    def gen_tape(self, params):
#        inputs = np.random.rand(self.num_qubits)
#        params = np.random.rand(self.layers, 6, num_qubits)
#        self.quantum_tape = qml.workflow.construct_tape(self.circuit)(inputs, params)
#        
#        #operation_names = []
#        #operation_qubits = []
#        operation_ids = []
#        
#        
#        for operation in self.quantum_tape.operations:
#            #if operation.name[0] == 'R':
#            #    operation_names.append(operation.name[1:])
#            #else:
#            #    operation_names.append(operation.name)
#            #operation_qubits.append(operation.wires.labels) 
#            operation_ids.append(operation.id)
#                
#        #self.operation_names = operation_names
#        #self.operation_qubits = operation_qubits
#        self.operation_ids = operation_ids
#        
#        param_labels = []
#        encoding_idx = 0
#        variational_idx = 0
#        for i in range(len(self.operation_ids)):
#            if self.operation_ids[i] == 'variational':
#                param_labels.append(self.operation_ids[i] + str(variational_idx))
#                variational_idx += 1
#                #self.operation_basis
#            elif self.operation_ids[i] == 'encoding':
#                param_labels.append(self.operation_ids[i] + str(encoding_idx))
#                encoding_idx += 1
#            elif self.operation_ids[i] == 'constant Clifford':
#                param_labels.append(id)
#        self.param_labels = param_labels
#        
#        observable_basis = []
#        observable_qubits = []
#        for observable in self.quantum_tape.observables:
#            observable_basis.append('Z'*len(observable.wires.labels))
#            observable_qubits.append(observable.wires.labels)
#        self.observable_basis = observable_basis
#        self.observable_qubits = observable_qubits
#    
#    @staticmethod
#    def sorted_ascending(lst):
#        return all(lst[i] <= lst[i+1] for i in range(len(lst) - 1))
#    
#    def sorted_descending(lst):
#        return all(lst[i] >= lst[i+1] for i in range(len(lst) - 1))
#    
#    def push_Cliffords_right(self):
#        
#        while self.sorted_ascending(self.operation_ids) and self.sorted_descending(self.operation_ids):
#            
#            #new_operation_names = deepcopy(self.operation_names)
#            #new_operation_qubits = deepcopy(self.operation_qubits)
#            new_operation_ids = deepcopy(self.operation_ids)
#            new_param_labels = deepcopy(self.param_labels)
#            new_quantum_tape = deepcopy(self.quantum_tape)
#            
#            for i in range(len(self.operation_names)-1):
#                if self.operation_ids[i] == 'constant Clifford' and self.operation_ids[i+1] != 'constant Clifford':
#                    if qml.is_commuting(self.quantum_tape.circuit[i], self.quantum_tape.circuit[i+1]):
#                        #new_operation_names[i+1] = self.operation_names[i]
#                        #new_operation_names[i] = self.operation_names[i+1]
#                        
#                        #new_operation_qubits[i+1] = self.new_operation_qubits[i]
#                        #new_operation_qubits[i] = self.new_operation_qubits[i+1]
#                        
#                        new_operation_ids[i+1] = self.new_operation_ids[i]
#                        new_operation_ids[i] = self.new_operation_ids[i+1]
#                        self.operation_ids = new_operation_ids
#                        
#                        new_param_labels[i+1] = self.new_param_labels[i]
#                        new_param_labels[i] = self.new_param_labels[i+1]
#                        self.param_labels = new_param_labels
#                        
#                        new_ops = []
#                        for j in range(len(self.quantum_tape)):
#                            if j == i+1:
#                                new_ops.insert(-2, self.quantum_tape[j])
#                            else:
#                                new_ops.append(self.quantum_tape[j])
#                        
#                        self.quantum_tape = qml.tape.QuantumTape(ops=new_ops)
#                        
#                    else:
#                        clifford_gate_name = self.quantum_tape.circuit[i].name
#                        clifford_gate_control_target_basis = 'Z' + clifford_gate_name[1:]
#                        clifford_gate_qubits = self.quantum_tape.circuit[i].wires.labels
#                        
#                        pauli_gate_name = self.quantum_tape.circuit[i+1].name[1:]
#                        pauli_gate_qubits = self.quantum_tape.circuit[i+1].wires.labels
#                        
#                    
#    
#    def IS_COMMUTING(self, A, B):
#        A_name, A_qubits = A
#        B_name, B_qubits = B
#        common_qubtis = set(A_qubits) & set(B_qubits)
#        common_qubits_A_indices = [i for i, q in enumerate(list(A_qubits)) if q in common_qubtis]
#        common_qubits_B_indices = [i for i, q in enumerate(list(B_qubits)) if q in common_qubtis]
#        A_truncated = ''
#        for i in common_qubits_A_indices:
#            A_truncated += A_name[i]
#        B_truncated = ''
#        for i in common_qubits_B_indices:
#            B_truncated += B_name[i]
#            
#        for i in range(len(A_truncated)):
#            if A_truncated[i] == 'C':
#                A_truncated[i] = 'Z'
#                
#        for i in range(len(B_truncated)):
#            if B_truncated[i] == 'C':
#                B_truncated[i] = 'Z'
#        
#        
#        
#        
#        for i in range(len(A_name)):
#            if A_name[i] == 'C':
#                A_name[i] = 'Z'
#        
#        for i in range(len(B_name)):
#            if B_name[i] == 'C':
#                B_name[i] = 'Z'
#        
#        
#        
#        
#        
#        
#        A_x = 0
#        A_z = 0
#        for i in range(len(A_truncated)):
#            if A_truncated[i] == 'I':
#                A_x += 0
#                A_z += 0
#            elif A_truncated[i] == 'X':
#                A_x += 2**(len(A_truncated)-i-1)
#                A_z += 0
#            elif A_truncated[i] == 'Z':
#                A_x += 0
#                A_z += 2**(len(A_truncated)-i-1)
#            elif A_truncated[i] == 'Y':
#                A_x += 2**(len(A_truncated)-i-1)
#                A_z += 2**(len(A_truncated)-i-1)
#            else:
#                raise Exception(f'{A_truncated} contains {A_truncated[i]} which is an invalid Pauli string')
#        A_truncated_num = (A_x, A_z)
#        
#        B_x = 0
#        B_z = 0
#        for i in range(len(B_truncated)):
#            if B_truncated[i] == 'I':
#                B_x += 0
#                B_z += 0
#            elif B_truncated[i] == 'X':
#                B_x += 2**(len(B_truncated)-i-1)
#                B_z += 0
#            elif B_truncated[i] == 'Z':
#                B_x += 0
#                B_z += 2**(len(B_truncated)-i-1)
#            elif B_truncated[i] == 'Y':
#                B_x += 2**(len(B_truncated)-i-1)
#                B_z += 2**(len(B_truncated)-i-1)
#            else:
#                raise Exception(f'{B_truncated} contains {B_truncated[i]} which is an invalid Pauli string')
#        B_truncated_num = (B_x, B_z)
#        
#        return is_commuting(A_truncated_num, B_truncated_num)
#
#        