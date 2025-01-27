import numpy as np
import pennylane as qml

def convert(quantum_script, device):
    # Extract the operations and observables from the QuantumScript
    operations = quantum_script.operations
    observables = quantum_script.observables

    # Create a function to define the QNode
    @qml.qnode(device)
    def qnode_func(*params):
        # Apply each operation in the QuantumScript
        for op, param_set in zip(operations, params):
            op.params = param_set
            qml.apply(op)
        return [qml.expval(obs) for obs in observables]

    return qnode_func


# Device wrapper class that includes label, device, and trust levels
class DeviceWrapper:
    def __init__(self, label, device, confidentiality_trust_level, integrity_trust_level=50):
        if not device:
            raise ValueError("Please specify a device")
        if 0 > confidentiality_trust_level or 100 < confidentiality_trust_level:
            raise ValueError("Confidentiality trust level must be between 1-100, inclusive")
        if 0 > integrity_trust_level or 100 < integrity_trust_level:
            raise ValueError("Integrity trust level must be between 1-100, inclusive")
        self.label = label
        self.device = device
        self.confidentiality_trust_level = confidentiality_trust_level
        self.integrity_trust_level = integrity_trust_level

    def __repr__(self):
        return f"DeviceWrapper(label={self.label}, confidentiality_trust_level={self.confidentiality_trust_level}, integrity_trust_level={self.integrity_trust_level})"

import pennylane as qml
import numpy as np

def create_forwarding_device(wrapped_devices, shots=None):
    if not wrapped_devices:
        raise ValueError("A list of wrapped devices must be provided")
    
    class ForwardingDevice(qml.QubitDevice):
        """A device that forwards quantum execution to another device and handles wire cutting."""

        name = "Forwarding device"
        short_name = "forwarding.device"
        pennylane_requires = ">=0.24"
        version = "0.1.0"
        author = "Paolo Bernardi"

        def __init__(self):
            super().__init__(wires=wrapped_devices[0].device.wires, shots=shots)
            self.wrapped_devices = wrapped_devices

            # Trust level probabilities
            self.total_trust_levels = np.array([d.confidentiality_trust_level + d.integrity_trust_level for d in wrapped_devices])
            self.probabilities = self.total_trust_levels / self.total_trust_levels.sum()
            self.execution_counts = np.zeros(len(wrapped_devices), dtype=int)

        @property
        def current_device(self):
            """Get the current device to execute on, selected probabilistically."""
            return self.wrapped_devices[self._select_device_based_on_trust()].device
        
        @property
        def current_device_label(self):
            """Get the label of the current wrapped device."""
            return self.wrapped_devices[self._select_device_based_on_trust()].label

        @property
        def operations(self):
            return self.current_device.operations

        @property
        def observables(self):
            return self.current_device.observables

        @classmethod
        def capabilities(cls):
            current_capabilities = wrapped_devices[0].device.capabilities()
            return current_capabilities

        def apply(self, operations, **kwargs):
            """Apply quantum operations to the current backend device."""
            self.current_device.reset()

            # Apply the operations to the current device
            for operation in operations:
                self.current_device.apply([operation])

        def expval(self, observable, **kwargs):
            """Calculate the expectation value."""
            if isinstance(observable, qml.ops.Prod):
                observables = [op for op in observable]
                expval_sum = sum(self.current_device.expval(op) for op in observables)
                return expval_sum
            else:
                return self.current_device.expval(observable)

        def execute(self, circuit):
            """Execute the quantum circuit on the selected backend device."""
            n = convert(circuit, self.current_device)

            # Handle cut circuits by executing each sub-circuit
            if hasattr(circuit, 'cutting') and circuit.cutting:
                # Extract sub-circuits from the cut circuit
                subcircuits = qml.cut_circuit_decompose(circuit)

                # Execute each sub-circuit on the appropriate device
                sub_results = []
                for sub_circuit in subcircuits:
                    sub_qnode = convert(sub_circuit, self.current_device)
                    sub_result = sub_qnode(*sub_circuit.get_parameters())
                    sub_results.append(sub_result)

                # Combine the results from the sub-circuits
                result = qml.cut_circuit_combine(sub_results)
            else:
                result = n(*circuit.get_parameters())
            
            # Update execution count for the selected device
            selected_device_index = self._select_device_based_on_trust()
            self.execution_counts[selected_device_index] += 1
            
            return result

        def _select_device_based_on_trust(self):
            """Select a device based on the weighted trust levels."""
            return np.random.choice(len(self.wrapped_devices), p=self.probabilities)

        def reset(self):
            super().reset()
            self.current_device.reset()

    return ForwardingDevice()

