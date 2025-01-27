from datetime import datetime, timedelta
import warnings

import pennylane as qml
import numpy as np
from qiskit_aer import AerSimulator
import matplotlib.pyplot as plt

test_count = 100
n_qubits = 15
shots = 1000
bins = 30
bin_range = None

dev = qml.device("qiskit.aer", wires=n_qubits, backend=AerSimulator(), shots=shots)


warnings.simplefilter(action="ignore", category=FutureWarning)


@qml.cut_circuit
@qml.qnode(device=dev)
def circuit():
    qml.Hadamard(wires=0)
    qml.RX(np.random.uniform(0, np.pi), wires=1)
    qml.RY(np.random.uniform(0, np.pi), wires=2)
    qml.RZ(np.random.uniform(0, np.pi), wires=3)
    qml.CNOT(wires=[0, 1])
    qml.CZ(wires=[1, 2])
    qml.SWAP(wires=[2, 3])
    qml.RX(np.random.uniform(0, np.pi), wires=3)
    qml.CNOT(wires=[3, 4])
    qml.WireCut(wires=4)
    qml.Hadamard(wires=5)
    qml.RX(np.random.uniform(0, np.pi), wires=6)
    qml.RY(np.random.uniform(0, np.pi), wires=7)
    qml.RZ(np.random.uniform(0, np.pi), wires=8)
    qml.CNOT(wires=[5, 6])
    qml.CZ(wires=[6, 7])
    qml.CNOT(wires=[7, 8])
    qml.Toffoli(wires=[5, 6, 7])
    qml.SWAP(wires=[8, 9])
    qml.WireCut(wires=9)
    qml.Hadamard(wires=10)
    qml.RX(np.random.uniform(0, np.pi), wires=11)
    qml.RY(np.random.uniform(0, np.pi), wires=12)
    qml.RZ(np.random.uniform(0, np.pi), wires=13)
    qml.CNOT(wires=[10, 11])
    qml.CZ(wires=[11, 12])
    qml.CNOT(wires=[12, 13])
    qml.Toffoli(wires=[10, 11, 12])
    qml.CNOT(wires=[13, 14])
    return qml.expval(qml.pauli.string_to_pauli_word("Z" * n_qubits))


results = []
start_execs = datetime.now()
for i in range(test_count):
    res = circuit()
    results.append(res)
    if (i + 1) % (test_count // 10) == 0:
        elapsed = int((datetime.now() - start_execs).total_seconds())
        total = elapsed * test_count // i
        remaining = timedelta(seconds=total - elapsed)
        print(f"{i+1} / {test_count} ({remaining} left)...")

counts, bin_edges = np.histogram(results, bins=bins, range=bin_range, density=True)
bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

plt.bar(
    bin_centers,
    counts,
    width=(bin_edges[1] - bin_edges[0]),
    color="skyblue",
    edgecolor="black",
    alpha=0.7,
)
plt.title(f"Histogram of Expectation Values (shots={shots})")
if bin_range:
    plt.xlim(*bin_range)
plt.xlabel("Expectation Value")
plt.ylabel("Probability Density")
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.show()
