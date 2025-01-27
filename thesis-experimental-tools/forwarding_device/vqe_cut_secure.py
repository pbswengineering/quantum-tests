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
        DeviceWrapper(label='Backend A', device=qml.device('default.mixed', wires=2), confidentiality_trust_level=20),
        #DeviceWrapper(label='Backend B', device=qml.device('default.mixed', wires=2), confidentiality_trust_level=50),
        #DeviceWrapper(label='Backend C', device=qml.device('default.mixed', wires=2), confidentiality_trust_level=90),
    ]
    
    # Create the forwarding device with round-robin execution
    forwarding_device = create_forwarding_device(backends, shots=SHOTS)

    # Define the Hamiltonian for the H2 molecule
    H = (
        -0.24274280513140462 * qml.PauliZ(0)
        + 0.17771287465139946 * qml.PauliZ(1)
        + 0.1777128746513994 * qml.PauliZ(0) @ qml.PauliZ(1)
        + 0.12293305056183798 * qml.PauliX(0) @ qml.PauliX(1)
    )

    # Define the ansatz (variational form)
    # This is the cost function to optimize (minimize the energy)
    @qml.cut_circuit
    @qml.qnode(forwarding_device)
    def ansatz(params):
        print("Ansatz", params)
        qml.RX(params[0], wires=0)
        qml.WireCut(1)
        qml.RY(params[1], wires=1)
        qml.CNOT(wires=[0, 1])
        return qml.expval(H)

    # Initialize parameters for the ansatz
    #params = np.random.random(2)
    #params = np.array([0.99156913, 0.61790925])
    params = qml.numpy.tensor([0.99156913, 0.61790925], requires_grad=True)

    #fig, ax = qml.draw_mpl(ansatz)(params)
    #plt.show()

    # Perform optimization to find the minimum energy (ground state)
    energies = []
    opt = qml.GradientDescentOptimizer(stepsize=0.4)
    for n in range(ITERATIONS):
        params = opt.step(ansatz, params)
        energy = ansatz(params)
        energies.append(energy)
        print(f"Iteration {n+1}: Energy = {energy}")

    # Final energy (ground state energy approximation)
    print("Ground state energy:", energy)

    # Save the energy convergence (hopefully!) plot to a file
    plt.plot(energies)
    plt.title('Energy value VS iteration')
    plt.xlabel('Iteration')
    plt.ylabel('Energy')
    plt.savefig('vqe_cut_secure.png')