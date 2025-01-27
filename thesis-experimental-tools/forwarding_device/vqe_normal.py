import matplotlib.pyplot as plt
import pennylane as qml
from pennylane import numpy as np
from qiskit_aer import AerSimulator

ITERATIONS = 100

# Define the Hamiltonian for the H2 molecule
H = (
    -0.24274280513140462 * qml.PauliZ(0)
    + 0.17771287465139946 * qml.PauliZ(1)
    + 0.1777128746513994 * qml.PauliZ(0) @ qml.PauliZ(1)
    + 0.12293305056183798 * qml.PauliX(0) @ qml.PauliX(1)
)

# Define the quantum device
dev = qml.device("default.mixed", wires=2)
#dev = qml.device("qiskit.aer", wires=2, backend=AerSimulator(), shots=4096)


# Define the ansatz (variational form)
# This is the cost function to optimize (minimize the energy)
@qml.qnode(dev)
def ansatz(params):
    print("Ansatz", params)
    qml.RX(params[0], wires=0)
    qml.RY(params[1], wires=1)
    qml.CNOT(wires=[0, 1])
    return qml.expval(H)

# Initialize parameters for the ansatz
#params = np.random.random(2)
params = np.array([0.99156913, 0.61790925])

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
plt.savefig('vqe_normal.png')