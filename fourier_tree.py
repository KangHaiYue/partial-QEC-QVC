class TreeNode:
    def __init__(self):
        self.data = None
        self.coefficient = None
        self.leftFactor = None
        self.rightFactor = None
        self.SinCosFactor = None
        self.leftChild = None
        self.rightChild = None
        
def binary_graph_expansion(O, pauli_generators):
    if isinstance(O, str):
        num_qubits = len(O)
        O_x = 0
        O_z = 0
        for i in range(len(O)):
            if O[i] == 'I':
                O_x += 0
                O_z += 0
            elif O[i] == 'X':
                O_x += 2**(num_qubits-i-1)
                O_z += 0
            elif O[i] == 'Z':
                O_x += 0
                O_z += 2**(num_qubits-i-1)
            elif O[i] == 'Y':
                O_x += 2**(num_qubits-i-1)
                O_z += 2**(num_qubits-i-1)
            else:
                raise Exception(f'{O} contains {O[i]} which is an invalid Pauli string')
        O_num = (O_x, O_z)
    
    pauli_generators_num = []
    for pauli in pauli_generators:
        if isinstance(pauli, str):
            num_qubits = len(pauli)
            pauli_x = 0
            pauli_z = 0
            for i in range(len(pauli)):
                if pauli[i] == 'I':
                    pauli_x += 0
                    pauli_z += 0
                elif pauli[i] == 'X':
                    pauli_x += 2**(num_qubits-i-1)
                    pauli_z += 0
                elif pauli[i] == 'Z':
                    pauli_x += 0
                    pauli_z += 2**(num_qubits-i-1)
                elif pauli[i] == 'Y':
                    pauli_x += 2**(num_qubits-i-1)
                    pauli_z += 2**(num_qubits-i-1)
                else:
                    raise Exception(f'{pauli} contains {pauli[i]} which is an invalid Pauli string')
            pauli_num = (pauli_x, pauli_z)
        else:
            pauli_num = pauli
        
        pauli_generators_num.append(pauli_num)
    

    root = TreeNode()
    root.data = (pauli_generators_num, O_num)
    root.coefficient = 1
    root.CosSinFactor = ['', '']  # [num of cos, num of sin]
    nodes_to_process = [(pauli_generators_num, O_num),]
    tree = [root]
    num_nodes_processed = 0
    while nodes_to_process:
        new_nodes_to_process = []
        num_nodes_to_process = len(nodes_to_process)
        for node in tree[num_nodes_processed : num_nodes_processed+num_nodes_to_process]:
            #current_node = TreeNode()
            if len(node.data[0]) > 1:
                new_nodes_to_process.append((node.data[0][:-1], node.data[1]))
            
            node.leftChild = TreeNode()
            node.leftChild.data = (node.data[0][:-1], node.data[1])
            node.leftChild.coefficient = node.coefficient
            tree.append(node.leftChild)
            
            if not is_commuting(node.data[0][-1], node.data[1]):
                if len(node.data[0]) > 1:
                    new_nodes_to_process.append((node.data[0][:-1], node.data[0][-1]^node.data[1]))
                
                node.rightChild = TreeNode()
                node.rightChild.data = (node.data[0][:-1], node.data[0][-1]^node.data[1])
                node.rightChild.coefficient = node.coefficient * 1j
                node.rightChild.CosSinFactor = [node.CosSinFactor[0]+'0', node.CosSinFactor[1]+'1']
                tree.append(node.rightChild)
                
                node.leftChild.CosSinFactor = [node.CosSinFactor[0]+'1', node.CosSinFactor[1]+'0']
                
            else:
                node.leftChild.CosSinFactor = [node.CosSinFactor[0]+'0', node.CosSinFactor[1]+'0']
            
            num_nodes_processed += 1
            
        nodes_to_process += new_nodes_to_process
        nodes_to_process = nodes_to_process[num_nodes_to_process : ]
    
    tree_leaves = [(node.data, node.SinCosFactor, node.coefficient) 
                   for node in tree if node.leftChild is None and node.rightChild is None]
    
    
    return tree_leaves


def binary_graph_expansion_raw(O, pauli_generators):
    if isinstance(O, str):
        num_qubits = len(O)
        O_x = 0
        O_z = 0
        for i in range(len(O)):
            if O[i] == 'I':
                O_x += 0
                O_z += 0
            elif O[i] == 'X':
                O_x += 2**(num_qubits-i-1)
                O_z += 0
            elif O[i] == 'Z':
                O_x += 0
                O_z += 2**(num_qubits-i-1)
            elif O[i] == 'Y':
                O_x += 2**(num_qubits-i-1)
                O_z += 2**(num_qubits-i-1)
            else:
                raise Exception(f'{O} contains {O[i]} which is an invalid Pauli string')
        O_num = (O_x, O_z)
    
    pauli_generators_num = []
    for pauli in pauli_generators:
        if isinstance(pauli, str):
            num_qubits = len(pauli)
            pauli_x = 0
            pauli_z = 0
            for i in range(len(pauli)):
                if pauli[i] == 'I':
                    pauli_x += 0
                    pauli_z += 0
                elif pauli[i] == 'X':
                    pauli_x += 2**(num_qubits-i-1)
                    pauli_z += 0
                elif pauli[i] == 'Z':
                    pauli_x += 0
                    pauli_z += 2**(num_qubits-i-1)
                elif pauli[i] == 'Y':
                    pauli_x += 2**(num_qubits-i-1)
                    pauli_z += 2**(num_qubits-i-1)
                else:
                    raise Exception(f'{pauli} contains {pauli[i]} which is an invalid Pauli string')
            pauli_num = (pauli_x, pauli_z)
        else:
            pauli_num = pauli
        
        pauli_generators_num.append(pauli_num)
    

    #root = TreeNode()
    #root.data = (pauli_generators_num, O_num)
    #root.coefficient = 1
    #root.CosSinFactor = ['', '']  # [num of cos, num of sin]
    nodes_to_process = [(pauli_generators_num, O_num),]
    tree = [[pauli_generators_num, O_num, 1, ['', '']],]
    num_nodes_processed = 0
    while nodes_to_process:
        new_nodes_to_process = []
        num_nodes_to_process = len(nodes_to_process)
        for node in tree[num_nodes_processed : num_nodes_processed+num_nodes_to_process]:
            #current_node = TreeNode()
            if len(node[0]) > 1:
                new_nodes_to_process.append((node[0][:-1], node[1]))
            
            #node.leftChild = TreeNode()
            #node.leftChild.data = (node.data[0][:-1], node.data[1])
            #node.leftChild.coefficient = node.coefficient
            
            branch_left = [node[0][:-1], node[1], node[2], node[3]]
            tree.append(branch_left)
            
            if not is_commuting(node[0][-1], node[1]):
                if len(node[0]) > 1:
                    new_nodes_to_process.append((node[0][:-1], node[0][-1]^node[1]))
                
                #node.leftChild.CosSinFactor = [node.CosSinFactor[0]+'1', node.CosSinFactor[1]+'0']
                tree[-1][3] = [node.CosSinFactor[0]+'1', node.CosSinFactor[1]+'0']
                
                #node.rightChild = TreeNode()
                #node.rightChild.data = (node[0][:-1], node[0][-1]^node[1])
                #node.rightChild.coefficient = node.coefficient * 1j
                #node.rightChild.CosSinFactor = [node.CosSinFactor[0]+'0', node.CosSinFactor[1]+'1']
                branch_right = [node[0][:-1], node[0][-1]^node[1], node[2]*1j, [node[3][0]+'0', node[3][1]+'1']]
                tree.append(branch_right)
                
            else:
                tree[-1][3] = [node.CosSinFactor[0]+'0', node.CosSinFactor[1]+'0']
            
            num_nodes_processed += 1
            
        nodes_to_process += new_nodes_to_process
        nodes_to_process = nodes_to_process[num_nodes_to_process : ]
    
    tree_leaves = [node for node in tree if len(node[0]) <= 1]
    
    
    return tree_leaves


def Pauli_product(A, B):
    X_A, Z_A = A
    X_B, Z_B = B
    
    X_C = X_A^X_B
    Z_C = Z_A^Z_B
    
    induced_coefficient = (1j)**( hamming_weight(X_A&(~Z_A)&(~X_B)&Z_B)#XZ
                         +hamming_weight(X_A&Z_A&X_B&(~Z_B))#YX
                         +hamming_weight((~X_A)&Z_A&X_B&Z_B)#ZY
                         -hamming_weight((~X_A)&Z_A&X_B&(~Z_B))#ZX
                         -hamming_weight(X_A&(~Z_A)&X_B&Z_B)#XY
                         -hamming_weight(X_A&Z_A&(~X_B)&Z_B))#YZ
    
    return X_C, Z_C, induced_coefficient

def is_commuting(A, B):
    #X_A, Z_A = A
    #X_B, Z_B = B
    
    X_AB, Z_AB, coeff_AB = Pauli_product(A,B)
    X_BA, Z_BA, coeff_BA = Pauli_product(B,A)
    
    if X_AB == X_BA and Z_AB == Z_BA and coeff_AB == coeff_BA:
        return True
    else:
        return False

# ref: http://graphics.stanford.edu/~seander/bithacks.html#CountBitsSetTable
POPCOUNT_TABLE16 = [0, 1] * 2**15
for index in range(2, len(POPCOUNT_TABLE16)):  # 0 and 1 are trivial
    POPCOUNT_TABLE16[index] += POPCOUNT_TABLE16[index >> 1]

def hamming_weight(n):
    """return the Hamming weight of an integer (check how many '1's for an integer after converted to binary)

    Args:
        n (int): any integer

    Returns:
        int: number of ones of the integer n after converted to binary
    """
    c = 0
    while n:
        c += POPCOUNT_TABLE16[n & 0xffff]
        n >>= 16
    return c