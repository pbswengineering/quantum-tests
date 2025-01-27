import pennylane as qml
from pennylane import numpy as np

# Define the device
dev = qml.device("default.qubit", wires=1)

# Define the QNode with known operations
@qml.qnode(dev)
def single_qubit_circuit():
    qml.RX(np.pi / 4, wires=0)
    qml.RY(np.pi / 6, wires=0)
    return qml.expval(qml.PauliZ(0))

# Execute the circuit and print the result
result = single_qubit_circuit()
print(f"Expected output of the circuit: {result:0.5f}")