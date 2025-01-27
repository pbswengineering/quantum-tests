import matplotlib.pyplot as plt
import pennylane as qml
import numpy as np
#from qiskit_aer import AerSimulator

from forwarding_device import DeviceWrapper, create_forwarding_device

SHOTS = 4096
ITERATIONS = 100

# Example usage
if __name__ == "__main__":
    # Create backend devices and wrap them with labels and trust levels
    #backends = [
    #    DeviceWrapper(label='Backend A', device=qml.device('qiskit.aer', wires=2, backend=AerSimulator(), shots=SHOTS), confidentiality_trust_level=20),
    #    DeviceWrapper(label='Backend B', device=qml.device('qiskit.aer', wires=2, backend=AerSimulator(), shots=SHOTS), confidentiality_trust_level=50),
    #    DeviceWrapper(label='Backend C', device=qml.device('qiskit.aer', wires=2, backend=AerSimulator(), shots=SHOTS), confidentiality_trust_level=90),
    #]

    backends = [
        DeviceWrapper(label='Backend A', device=qml.device('default.mixed', wires=3), confidentiality_trust_level=20),
        DeviceWrapper(label='Backend B', device=qml.device('default.mixed', wires=3), confidentiality_trust_level=50),
        DeviceWrapper(label='Backend C', device=qml.device('default.mixed', wires=3), confidentiality_trust_level=90),
    ]
    
    # Create the forwarding device with round-robin execution
    forwarding_device = create_forwarding_device(backends, shots=SHOTS)

    @qml.cut_circuit
    @qml.qnode(forwarding_device)
    def test_circuit():
        qml.RX(0.531, wires=0)
        qml.RY(0.9, wires=1)
        qml.RX(0.3, wires=2)
        qml.CZ(wires=(0, 1))
        qml.RY(-0.4, wires=0)
        qml.WireCut(wires=1)
        qml.CZ(wires=[1, 2])
        qml.RY(-0.4, wires=1)
        qml.RY(-0.4, wires=0)
        return qml.expval(qml.pauli.string_to_pauli_word("ZZZ"))

    result = test_circuit()
    print("RESULT =", result)