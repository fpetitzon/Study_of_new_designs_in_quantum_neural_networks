

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import RealAmplitudes, ZFeatureMap, ZZFeatureMap
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Statevector
from qiskit.circuit.library.n_local import EfficientSU2
from qiskit.circuit.library import NLocal


def ZzFeatureMapRealAmplitudeCircuit(n):
    feature_map = ZZFeatureMap(n, reps=2)
    var_form = RealAmplitudes(n, reps=1)
    circuit = feature_map.combine(var_form)
    return circuit


def CombinedQnn(n):
    circuit = QuantumCircuit(n)
    p1 = ParameterVector("x", n)
    p2 = ParameterVector("alpha", 1)
    for idx, i in enumerate(p1):
        circuit.ry(i, idx)
        circuit.cx(idx, (idx + 1) % n)
        circuit.rx(p2.params[0] * i, (idx + 1) % n)
        circuit.cx(idx, (idx + 1) % n)

    circuit.barrier()

    p3 = ParameterVector("beta", n)
    p4 = ParameterVector("gamma", n)
    p5 = ParameterVector("delta", n)
    for idx, i in enumerate(p3):
        circuit.ry(i, idx)
        circuit.cx(idx, (idx + 1) % n)
        circuit.rx(p4.params[0], idx)
        circuit.cx(idx, (idx + 1) % n)
        circuit.ry(p5.params[idx], idx)

    circuit.barrier()

    p6 = ParameterVector("eta", 1)
    for idx, i in enumerate(p1):
        circuit.ry(i, idx)
        circuit.cx(idx, (idx + 1) % n)
        circuit.rx(p6.params[0] * i, idx)
        circuit.cx(idx, (idx + 1) % n)

    return circuit


class CircuitFactory:
    @staticmethod
    def provide(type, nqbit):
        return provide_circuit(type, nqbit)


def provide_circuit(type, n):
    if type == 'CombinedQnn':
        return CombinedQnn(n)
    elif type == 'ZzFeatureMapRealAmplitudeCircuit':
        return ZzFeatureMapRealAmplitudeCircuit(n)
    else:
        raise ValueError(type)
