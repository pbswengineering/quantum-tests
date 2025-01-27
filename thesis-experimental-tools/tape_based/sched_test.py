from datetime import datetime

import pennylane as qml
from pennylane import numpy as np
from qiskit_aer import AerSimulator

from qscheduler import QScheduler


def check_error(reference_res, other_res, label):
    error = reference_res - other_res
    error_perc = abs(error * 100 / reference_res)
    print(f"ERROR ({label}) = {error:0.5f} ({error_perc:0.2f}%)")
    return error_perc


start_time = datetime.now()

sched = QScheduler()

sched.add_device("DEV1", qml.device("qiskit.aer", wires=2, backend=AerSimulator(), shots=4096), 10)
sched.add_device("DEV2", qml.device("qiskit.aer", wires=2, backend=AerSimulator(), shots=4096), 5)
sched.add_device("DEV3", qml.device("qiskit.aer", wires=2, backend=AerSimulator(), shots=4096), 7)
# sched.add_device(qml.device("qiskit.aer", wires=2, backend=AerSimulator(), method="automatic", shots=1024), 10)
# sched.add_device(qml.device("qiskit.aer", wires=2, backend=UnitarySimulator(), shots=1024), 10)

# first_dev = qml.device("qiskit.basicsim", wires=2, shots=1024)
# wires = first_dev.wires
# sched.add_device(first_dev, 10)
# sched.add_device(qml.device("qiskit.basicsim", wires=2, shots=1024), 2)

ops = [
    qml.RX(0.531, wires=0),  # Angle encoding
    qml.RY(0.9, wires=1),
    qml.RX(0.3, wires=2),
    qml.CZ(wires=(0, 1)),
    qml.RY(-0.4, wires=0),
    qml.CZ(wires=[1, 2]),
    qml.RY(-0.4, wires=1),
    qml.RY(-0.4, wires=0),
]
measurements = [qml.expval(qml.pauli.string_to_pauli_word("ZZZ"))]
baseline_tape = qml.tape.QuantumTape(ops, measurements)

ops_with_cutting = [
    qml.RX(0.531, wires=0),  # Angle encoding
    qml.RY(0.9, wires=1),
    qml.RX(0.3, wires=2),
    qml.CZ(wires=(0, 1)),
    qml.RY(-0.4, wires=0),
    qml.WireCut(wires=1),
    qml.CZ(wires=[1, 2]),
    qml.RY(-0.4, wires=1),
    qml.RY(-0.4, wires=0),
]
measurements = [qml.expval(qml.pauli.string_to_pauli_word("ZZZ"))]
original_tape = qml.tape.QuantumTape(ops_with_cutting, measurements)
def circuit_fn():
    return original_tape
sched.set_real_circuit_fn(circuit_fn)
qml.drawer.tape_mpl(original_tape)

dev = qml.device("lightning.qubit", wires=3)
baseline = qml.execute([baseline_tape], dev)[0]
print("RESULT (baseline) =", baseline)

singledev_errors = []
multidev_normal_errors = []
multidev_secure_errors = []
for n in range(10):
    singledev = sched.exec_with_singledev_cut()
    print("RESULT (with single-device cutting) =", singledev)
    multidev_normal = sched.exec_with_normal_multidev_cut()
    print("RESULT (with normal multiple-device cutting) =", multidev_normal)
    multidev_secure = sched.exec_with_secure_multidev_cut()
    if not multidev_secure:
        print("Secure execution skipped")
        continue
    print("RESULT (with secure multiple-device cutting) =", multidev_secure)
    singledev_errors.append(check_error(baseline, singledev, "baseline VS singledev"))
    multidev_normal_errors.append(check_error(baseline, multidev_normal, "baseline VS multidev normal"))
    multidev_secure_errors.append(check_error(baseline, multidev_secure, "baseline VS multidev secure"))

print("-" * 30)
print(
    f"Single-device: MIN {min(singledev_errors):0.2f}%, MAX {max(singledev_errors):0.2f}%, AVG {(sum(singledev_errors) / len(singledev_errors)):0.2f}%"
)
print(
    f"Multiple-device, normal: MIN {min(multidev_normal_errors):0.2f}%, MAX {max(multidev_normal_errors):0.2f}%, AVG {(sum(multidev_normal_errors) / len(multidev_normal_errors)):0.2f}%"
)
print(
    f"Multiple-device, secure: MIN {min(multidev_secure_errors):0.2f}%, MAX {max(multidev_secure_errors):0.2f}%, AVG {(sum(multidev_secure_errors) / len(multidev_secure_errors)):0.2f}%"
)
print("-" * 30)

end_time = datetime.now()
print(f"Duration: {end_time - start_time}")
