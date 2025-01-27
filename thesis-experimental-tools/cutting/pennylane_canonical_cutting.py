import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt

dev = qml.device("default.qubit", wires=3)

@qml.cut_circuit
@qml.qnode(dev)
def circuit(params):
	qml.RY(params[0], wires=0)
	qml.RX(params[1], wires=1)
	qml.RZ(params[0], wires=2)
	qml.WireCut(wires=1)
	qml.CNOT([0, 1])
	qml.WireCut(wires=2)
	qml.CNOT([1, 2])
	return qml.expval(qml.pauli.string_to_pauli_word("ZZZ"))

params = np.array([0.16, 0.01])
fig, ax = qml.draw_mpl(circuit)(params)
plt.savefig("pennylane-cutting.pdf", format="pdf", bbox_inches="tight")
result = circuit(params)
print(result)