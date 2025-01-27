from qiskit_ibm_runtime import QiskitRuntimeService
 
# Save an IBM Quantum account and set it as your default account.
QiskitRuntimeService.save_account(
    channel="ibm_quantum",
    token="<TOKEN>",
    set_as_default=True,
    # Use `overwrite=True` if you're updating your token.
    overwrite=True,
)