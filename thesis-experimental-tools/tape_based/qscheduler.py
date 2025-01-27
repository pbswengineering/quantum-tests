import random
from typing import Callable, Dict, List, Optional, Tuple, Union

import networkx.classes.multidigraph
import numpy as np
import pennylane as qml

class QScheduler:
    """Executes quantum circuits by applying circuit cutting (both normal and secure cutting)."""

    devices: List[Dict[str, Union[qml.Device, int, str]]]
    real_circuit_fn: Optional[Callable]  # It must return a qml.tape.tape.QuantumTape

    def __init__(self):
        self.devices = []
        self.real_circuit_fn = None

    def add_device(self, label: str, device: qml.Device, trust_confidentiality: int, trust_integrity: int = 10) -> bool:
        """Add a quantum processor (it may be simulated or physical) with its trust levels (1..10, inclusive)."""
        assert device is not None, "Please specify a device"
        assert 1 <= trust_confidentiality <= 10, "Confidentiality trust should be 1..10, inclusive"
        assert 1 <= trust_integrity <= 10, "Integrity trust should be 1..10, inclusive"
        self.devices.append(
            {
                "device": device,
                "trust_confidentiality": trust_confidentiality,
                "trust_integrity": trust_integrity,
                "label": label,
            }
        )
        return True

    def set_real_circuit_fn(self, circuit_fn: Callable):
        """Set the real quantum circuit that must be executed."""
        assert circuit_fn is not None, "Please specify a circuit"
        self.real_circuit_fn = circuit_fn

    def __cut_tape(self, original_tape: qml.tape.tape.QuantumTape, wires: qml.wires.Wires
                   ) -> Tuple[List[qml.tape.qscript.QuantumScript], networkx.classes.multidigraph.MultiDiGraph]:
        graph = qml.qcut.tape_to_graph(original_tape)
        qml.qcut.replace_wire_cut_nodes(graph)
        fragments, communication_graph = qml.qcut.fragment_graph(graph)
        fragment_tapes = [qml.qcut.graph_to_tape(f) for f in fragments]
        fragment_tapes = [qml.map_wires(t, dict(zip(t.wires, wires)))[0][0] for t in fragment_tapes]
        return fragment_tapes, communication_graph

    def __expand_sub_circuit(self, fragment_tapes: List[qml.tape.qscript.QuantumScript]
                             ) -> Tuple[List[qml.tape.qscript.QuantumScript], List[List[qml.qcut.utils.PrepareNode]],
                                        List[List[qml.qcut.utils.MeasureNode]]]:
        expanded = [qml.qcut.expand_fragment_tape(t) for t in fragment_tapes]
        configurations = []
        prepare_nodes = []
        measure_nodes = []
        for t, p, m in expanded:
            configurations.append(t)
            prepare_nodes.append(p)
            measure_nodes.append(m)
        sub_tapes = tuple(tape for c in configurations for tape in c)
        return sub_tapes, prepare_nodes, measure_nodes

    def __check_integrity(self, device: Dict[str, Union[qml.Device, int, str]]):
        @qml.qnode(device["device"])
        def test_circuit():
            qml.RX(np.pi / 4, wires=0)
            qml.RY(np.pi / 6, wires=0)
            return qml.expval(qml.PauliZ(0))
        expected_outcome = 0.61237
        result = test_circuit()
        error = abs(expected_outcome - result)
        error_perc = error / expected_outcome
        device['trust_integrity'] = max(int(device['trust_integrity'] * error_perc), 0)

    def exec_with_singledev_cut(self, *args, **kwargs):
        """Execute a quantum circuit with a standard cutting technique using the first device."""
        tape = self.real_circuit_fn(*args, **kwargs)
        wires = self.devices[0]["device"].wires
        fragment_tapes, communication_graph = self.__cut_tape(tape, wires)
        sub_tapes, prepare_nodes, measure_nodes = self.__expand_sub_circuit(fragment_tapes)
        results = []
        for t in sub_tapes:
            dev = self.devices[0]
            res = qml.execute([t], dev["device"], gradient_fn=None)[0]
            results.append(res)
        result_with_cut = qml.qcut.qcut_processing_fn(
            results,
            communication_graph,
            prepare_nodes,
            measure_nodes,
        )
        return result_with_cut

    def exec_with_normal_multidev_cut(self, *args, **kwargs):
        """Execute a quantum circuit with a standard multiple-device cutting technique."""
        tape = self.real_circuit_fn(*args, **kwargs)
        wires = self.devices[0]["device"].wires
        fragment_tapes, communication_graph = self.__cut_tape(tape, wires)
        sub_tapes, prepare_nodes, measure_nodes = self.__expand_sub_circuit(fragment_tapes)
        results = []
        for i, t in enumerate(sub_tapes):
            dev = self.devices[i % len(self.devices)]
            res = qml.execute([t], dev["device"], gradient_fn=None)[0]
            results.append(res)
        result_with_cut = qml.qcut.qcut_processing_fn(
            results,
            communication_graph,
            prepare_nodes,
            measure_nodes,
        )
        return result_with_cut

    def exec_with_secure_multidev_cut(self, *args, **kwargs):
        """Execute a quantum circuit with a multiple-device, trust-based cutting technique."""

        def combine_results(results, trusts):
            """Combine results from multiple devices based on their integrity trust levels."""
            total_weight = sum(2**trust for trust in trusts)
            if isinstance(results[0], tuple):
                combined_result = [0] * len(results[0])
                for result, trust in zip(results, trusts):
                    for i in range(len(result)):
                        combined_result[i] += result[i] * (2**trust)
                combined_result = tuple(r / total_weight for r in combined_result)
            else:
                combined_result = sum(result * (2**trust) for result, trust in zip(results, trusts)) / total_weight
            return combined_result

        # Check integrity of all devices before execution
        for device in self.devices:
            self.__check_integrity(device)

        # Filter devices with non-zero confidentiality trust
        eligible_devices = [dev for dev in self.devices if dev['trust_confidentiality'] > 0]

        # If less than 3 devices are eligible, skip execution
        if len(eligible_devices) < 3:
            print("Insufficient devices with positive confidentiality trust. Skipping execution.")
            return None

        # Cut the tape into fragments
        tape = self.real_circuit_fn(*args, **kwargs)
        wires = eligible_devices[0]["device"].wires
        fragment_tapes, communication_graph = self.__cut_tape(tape, wires)
        sub_tapes, prepare_nodes, measure_nodes = self.__expand_sub_circuit(fragment_tapes)

        results = []

        for t in sub_tapes:
            # Allocate devices proportionally to their trust levels
            chosen_devices = random.choices(
                eligible_devices,
                weights=[device['trust_confidentiality'] + device['trust_integrity'] for device in eligible_devices],
                k=2
            )

            # Execute the same sub-tape on both chosen devices
            fragment_results = []
            for dev in chosen_devices:
                res = qml.execute([t], dev["device"], gradient_fn=None)[0]
                fragment_results.append(res)

            # Combine results using their integrity trust
            fragment_trusts = [dev['trust_integrity'] for dev in chosen_devices]
            combined_result = combine_results(fragment_results, fragment_trusts)
            results.append(combined_result)

        result_with_weighted_cut = qml.qcut.qcut_processing_fn(
            results, communication_graph, prepare_nodes, measure_nodes
        )
        return result_with_weighted_cut
