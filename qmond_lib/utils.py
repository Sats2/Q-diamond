import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector
from typing import Dict, Tuple, Iterable, Union

def probs_dict_to_subset_dict(probs_dict: Dict[str, float]) -> Dict[Tuple[int, ...], float]:
    """
    Convert Qiskit's probabilities_dict (bitstring keys) to a subset→value dict.

    Example: {"00": 0.5, "11": 0.5}  →  {(): 0.5, (0,1): 0.5}
    """
    out = {}
    for bitstring, p in probs_dict.items():
        n = len(bitstring)
        # Reverse order: bitstring[-1] = qubit 0, bitstring[-2] = qubit 1, etc.
        subset = tuple(i for i, bit in enumerate(reversed(bitstring)) if bit == "1")
        out[subset] = p
    return out

Subset = Tuple[int, ...]
AmpDict = Dict[Subset, complex]

def statevector_to_subset_dict(state: Union["Statevector", np.ndarray, Iterable[complex]],
                               eps: float = 0.0) -> AmpDict:
    """
    Convert a Qiskit Statevector (or 1D complex iterable) into a dict that maps
    'subset of qubit indices that are 1' -> complex amplitude.

    - Bit convention: bit j corresponds to qubit j (qubit 0 is least significant).
    - If eps > 0, amplitudes with |a| <= eps are dropped.
    """
    # Extract flat complex array
    if isinstance(state, Statevector):
        vec = np.asarray(state.data, dtype=complex)
        n = int(round(np.log2(vec.size)))
    else:
        vec = np.asarray(state, dtype=complex).ravel()
        n = int(round(np.log2(vec.size)))

    if vec.size != (1 << n):
        raise ValueError("Input length is not a power of two; expected a qubit statevector.")

    out: AmpDict = {}
    for i, a in enumerate(vec):
        if abs(a) <= eps:
            continue
        # subset = indices of qubits that are 1 in the basis state |b_{n-1} ... b_1 b_0>
        subset = tuple(j for j in range(n) if (i >> j) & 1)
        out[subset] = a
    return out