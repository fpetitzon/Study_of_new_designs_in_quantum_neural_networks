import numpy as np
from qiskit import QuantumCircuit, execute
from qiskit.circuit.library import RealAmplitudes, ZFeatureMap, ZZFeatureMap
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Statevector
from qiskit.providers.aer import AerError
from qiskit.providers.aer import AerSimulator
from qiskit.providers.aer import QasmSimulator
from qiskit.providers.aer.noise import NoiseModel
from qiskit.test.mock import FakeManila
import torch as T

from qiskit import Aer, transpile, IBMQ

processor = Aer.backends(name = 'qasm_simulator', method="statevector", )[0]

API_TOKEN = '210fee10a48b26b8d09eacf5d8dc6c16b61380e6df4062640eb87aa59165cfd2e223e4775c8b2c17971e9cb44e0b1f835ed3544b27cf2c33406d08cc337baa83'
IBMQ.save_account(API_TOKEN)
provider = IBMQ.load_account()

RUN_ON_QPU = True
RUN_ON_GPU = False

QPU_INSTANCE_NAME = 'ibmq_manila'

if RUN_ON_GPU:
    try:
        processor.set_options(devices='GPU')
        print('GPU Acceleration Enabled')
    except AerError as e:
        print("error = ",e)
elif RUN_ON_QPU:
    try:
        backend = provider.get_backend(QPU_INSTANCE_NAME)
        noise_model = NoiseModel.from_backend(backend)

        coupling_map = backend.configuration().coupling_map
        basis_gates = noise_model.basis_gates
        processor = backend
        print('QPU Enabled')
    except:
        print('QPU Error')
else:
    try:
        backend = provider.get_backend(QPU_INSTANCE_NAME)
        noise_model = NoiseModel.from_backend(backend)

        coupling_map = backend.configuration().coupling_map
        basis_gates = noise_model.basis_gates
        processor = backend
        print('QPU Enabled')
    except:
        print('QPU Error')



class ClassicalNet(T.nn.Module):

    def __init__(self, n, extra_layer=True, neurons=4, binary_classification=True):
        super(ClassicalNet, self).__init__()
        self.extra_layer = extra_layer
        self.hid1 = T.nn.Linear(n, neurons)
        if self.extra_layer is True:
            self.hid2 = T.nn.Linear(neurons, neurons)
        self.output = T.nn.Linear(neurons, 1)
        self.binary_classification = binary_classification

        T.nn.init.xavier_uniform_(self.hid1.weight)
        T.nn.init.zeros_(self.hid1.bias)

        if self.extra_layer is True:
            T.nn.init.xavier_uniform_(self.hid2.weight)
            T.nn.init.zeros_(self.hid2.bias)
        T.nn.init.xavier_uniform_(self.output.weight)
        T.nn.init.zeros_(self.output.bias)

        self.double()

    def get_num_params(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        return sum([np.prod(p.size()) for p in model_parameters])

    def forward(self, x):
        if self.binary_classification is True:
            z = T.tanh(self.hid1(x))
            if self.extra_layer is True:
                z = T.tanh(self.hid2(z))
            z = self.output(z)
            z = T.sigmoid(z)
        else:
            z = T.nn.functional.leaky_relu(self.hid1(x))
            if self.extra_layer is True:
                z = T.nn.functional.leaky_relu(self.hid2(z))
            z = self.output(z)
            z = T.nn.functional.leaky_relu(z)

        return z


def parity(counts, class_labels=['0', '1']):
    shots = sum(counts.values())
    result = {class_labels[0]: 0,
              class_labels[1]: 0}
    for key, item in counts.items():
        label = assign_label_by_parity(key, class_labels)
        result[label] += counts[key] / shots
    return result


def assign_label_by_parity(bit_string, class_labels):
    hamming_weight = sum([int(k) for k in list(bit_string)])
    is_odd_parity = hamming_weight & 1
    if is_odd_parity:
        return class_labels[1]
    else:
        return class_labels[0]


def probabilities_per_qubit(counts):
    shots = sum(counts.values())
    nqbit = len(list(counts.items())[0][0])
    result = np.zeros(nqbit)
    for key, item in counts.items():
        for i in range(nqbit):
            if key[i] is '0':
                result[i] += counts[key] / shots
    return result


def measure_one_qubit(counts, measuring_qubit=0):
    shots = sum(counts.values())
    result = np.zeros(1)
    for key, item in counts.items():
        if key[measuring_qubit] is '0':
            result[0] += counts[key] / shots
    return result


""" def run_circuit(circuit, backend, shots=20):
    compiled_circuit = transpile(circuit, backend)
    job = backend.run(compiled_circuit, shots=shots)
    result = job.result()
    # counts = result.get_counts(compiled_circuit)
    counts = result.get_counts()
    return counts """

def run_circuit(circuit, backend, shots=20):
    compiled_circuit = transpile(circuit, backend)
    result = execute(compiled_circuit, Aer.get_backend('qasm_simulator'),
                 coupling_map=coupling_map,
                 basis_gates=basis_gates,
                 noise_model=noise_model).result()
    counts = result.get_counts(0)
    return counts


class Qnn:
    def __init__(self, circuit, post_process="parity"):
        # super().__init__()
        self.circuit = circuit
        self.circuit_length = len(self.circuit._qubits)
        self.sv = Statevector.from_label('0' * self.circuit_length)
        self.post_process = post_process  # might be better to pass on function though?
        self.count_of_runs = 0
        self.params = np.random.rand(len(self.circuit.parameters) - self.circuit_length) * 2 * np.pi

    def get_data_dict(self, params, x):
        # x are the features from the dataset
        # params are the params of the variational circuit
        if T.is_tensor(params):
            params = params.detach().numpy()
        if T.is_tensor(x):
            x = x.detach().numpy()

        param_dict = {}
        for idx, param in enumerate([x for x in self.circuit.parameters if str(x.name).split('[')[0] != "x"]):
            param_dict.update({param: params[idx]})
        for idx, param in enumerate([x for x in self.circuit.parameters if str(x.name).split('[')[0] == "x"]):
            param_dict.update({param: x[idx]})
        return param_dict

    def get_count_of_runs(self):
        return self.count_of_runs

    def cost_function(self, input, target, loss_obj, params=None):
        output = T.tensor(self.predict(input, params))
        loss = loss_obj(output, target)
        return loss.item()

    def predict(self, x_list, params=None):  # predicts via statevector evolve

        if params is None:
            params = self.params

        qc_list = []
        for x in x_list:
            circ_ = self.circuit.assign_parameters(self.get_data_dict(params, x))
            qc = self.sv.evolve(circ_)
            qc_list += [qc]
            self.count_of_runs += 1

        results = []
        for qc in qc_list:
            counts = qc.probabilities_dict()
            if self.post_process is "parity":
                result = parity(counts)
            elif self.post_process is "probabilities_per_qubit":
                result = probabilities_per_qubit(counts)
            elif self.post_process is "measure_one_qubit":
                result = measure_one_qubit(counts)
            else:
                raise Exception("not implemented postprocessing method")

            results += [result]

        # TO DO, fix this / move to parity function or move whole post_process into one method.
        if self.post_process is "parity":
            # small rework to tensor shape
            result = np.zeros(len(results))
            for index in range(len(results)):
                result[index] = results[index]['0']
            return result

        return results

    def predict_with_backend(self, x_list, params=None, backend=processor, shots=20):  # _with_backend
        # predicts via backend, for example qasm_backend. Generally slower.

        if params is None:
            params = self.params

        qc_list = []
        for x in x_list:
            circ_ = self.circuit.assign_parameters(self.get_data_dict(params, x))
            circ_.measure_all()
            counts = run_circuit(circ_, backend, shots)
            qc_list += [counts]
            self.count_of_runs += 1

        results = []
        for counts in qc_list:

            if self.post_process is "parity":
                result = parity(counts)
            elif self.post_process is "probabilities_per_qubit":
                result = probabilities_per_qubit(counts)
            else:
                raise Exception("not implemented postprocessing method")

            results += [result]

        # TO DO, fix this / move to parity function or move whole post_process into one method.
        if self.post_process is "parity":
            # small rework to tensor shape
            result = np.zeros(len(results))
            for index in range(len(results)):
                result[index] = results[index]['0']
            return result

        return results

class QnnTorchConnector(T.nn.Module):
    def __init__(self, qnn, shift=np.pi / 32.0):
        super(QnnTorchConnector, self).__init__()
        self.shift = shift
        self.qnn = qnn
        self.circuit = qnn.circuit
        self.circuit_length = len(self.circuit._qubits)
        custom_weight = T.rand(len(self.circuit.parameters) - self.circuit_length) * 2 * np.pi
        self.weight = T.nn.Parameter(custom_weight, requires_grad=True)

    def forward(self, input):
        result = QnnCircuitFunction.apply(input, self.qnn, self.weight, self.shift)
        return result

    def get_params(self):
        return self.weight.detach().numpy()

    def get_num_params(self):
        return len(self.weight.detach().numpy())


class HybridClassificationNet(T.nn.Module):
    def __init__(self, qnn, cnn, binary_classification=True):
        super(HybridClassificationNet, self).__init__()
        self.qnn = qnn
        self.cnn = cnn  # T.nn.Linear(len(self.qnn.circuit._qubits), 1)
        self.double()
        self.binary_classification = binary_classification

    def forward(self, x):
        x = self.qnn(x)
        x = self.cnn(x)
        if self.binary_classification is True:
            x = T.sigmoid(x)
        else:
            x = T.leaky_relu(x)
        return x

    def get_params(self):
        return self.qnn.weight.detach().numpy(), self.cnn.weight.detach().numpy()

    def get_num_params(self):
        return len(self.qnn.weight.detach().numpy()) + len(self.cnn.weight.detach().numpy()[0])


class QnnCircuitFunction(T.autograd.Function):
    @staticmethod
    def forward(ctx, input, qnn, weight, shift):
        ctx.shift = shift
        ctx.qnn = qnn
        ctx.weight = weight
        result = ctx.qnn.predict_with_backend(input, weight)

        predictions = T.tensor([np.array(result)])
        ctx.save_for_backward(input, predictions)
        return predictions

    @staticmethod
    def backward(ctx, grad_out):
        """ Backward pass computation """
        input, result = ctx.saved_tensors
        batch_size = input.size()[0]
        param_size = len(ctx.weight)
        gradients = []

        final_grad = np.zeros(param_size)

        for i in range(param_size):
            direction = np.zeros(param_size)
            direction[i] = 1
            shift_right = direction * ctx.shift
            shift_left = - direction * ctx.shift
            expectation_right = ctx.qnn.predict_with_backend(input, params=ctx.weight + shift_right)
            expectation_left = ctx.qnn.predict_with_backend(input, params=ctx.weight + shift_left)
            gradient = T.tensor([expectation_right]) - T.tensor([expectation_left])
            gradient *= grad_out.float()
            gradients.append(gradient)

        for j in range(param_size):
            for i in range(batch_size):
                final_grad[j] += T.sum(gradients[j][0][i]) / batch_size

        return None, None, T.tensor(final_grad), None
