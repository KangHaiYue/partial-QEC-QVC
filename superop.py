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
    def superop2kraus(superop, tol: float = 1e-10) -> list:
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
    print('You are running superop.py directly, which is not recommended.')