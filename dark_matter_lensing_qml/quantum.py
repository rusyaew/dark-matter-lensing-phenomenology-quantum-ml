from __future__ import annotations

import numpy as np
import pennylane as qml
import torch
from torch import nn


def apply_angle_encoding(angle_features: torch.Tensor) -> None:
    # one compressed classical feature per qubit
    for qubit_index, angle in enumerate(angle_features):
        qml.RY(angle, wires=qubit_index)


def apply_variational_layer(
        layer_weights: torch.Tensor,
        *,
        n_qubits: int,
        noise_strength: float,
) -> None:
    # local trainable rotations.
    for qubit_index in range(n_qubits):
        qml.RY(layer_weights[qubit_index, 0], wires=qubit_index)
        qml.RZ(layer_weights[qubit_index, 1], wires=qubit_index)

    # ring entangling so every qubit participates in >= 1 of 2-qubit gates.
    for qubit_index in range(n_qubits - 1):
        qml.CNOT(wires=[qubit_index, qubit_index + 1])
    if n_qubits > 1:
        qml.CNOT(wires=[n_qubits - 1, 0])

    # depolarizing noise on the mixed-state simulator
    if noise_strength > 0.0:
        for qubit_index in range(n_qubits):
            qml.DepolarizingChannel(noise_strength, wires=qubit_index)


def build_variational_quantum_circuit(
        *,
        n_qubits: int,
        n_layers: int,
        noise_strength: float,
):
    """here we build a small VQC circuit baseline as in standard pennylane pattern:

    angle encoding --> trainable single-qubit rotations --> ring of CNOT entanglers
    --> Pauli-Z expectation readout.
    """

    if n_qubits <= 0:
        raise ValueError("quantum: n_qubits must be > 0")
    if n_layers <= 0:
        raise ValueError("quantum: n_layers must be > 0")
    if noise_strength < 0.0:
        raise ValueError("quantum: noise_strength must be >= 0.0")

    if noise_strength > 0.0:
        device = qml.device("default.mixed", wires=n_qubits)
        diff_method = "best"
    else:
        device = None
        diff_method = "best"
        for device_name, candidate_diff_method in (
                ("default.qubit", "backprop"),
                ("lightning.qubit", "adjoint"),
        ):
            try:
                device = qml.device(device_name, wires=n_qubits)
                diff_method = candidate_diff_method
                break
            except Exception:
                continue
        if device is None:
            raise RuntimeError("quantum: failed to initialize an available noiseless simulator")

    @qml.qnode(device, interface="torch", diff_method=diff_method)
    def circuit(angle_features: torch.Tensor, layer_weights: torch.Tensor):
        apply_angle_encoding(angle_features)
        for layer_index in range(n_layers):
            apply_variational_layer(
                layer_weights[layer_index],
                n_qubits=n_qubits,
                noise_strength=noise_strength,
            )

        # one scalar per qubit for the classical downstream heads.
        return [qml.expval(qml.PauliZ(qubit_index)) for qubit_index in range(n_qubits)]

    return circuit


class VariationalQuantumClassifier(nn.Module):
    def __init__(
            self,
            *,
            input_dim: int,
            n_lensing_classes: int,
            n_qubits: int,
            n_layers: int,
            noise_strength: float,
    ) -> None:
        super().__init__()

        self.n_qubits = n_qubits
        self.input_projection = nn.Linear(input_dim, n_qubits)
        self.quantum_weights = nn.Parameter(0.01 * torch.randn(n_layers, n_qubits, 2))
        self.output_head = nn.Linear(n_qubits, n_lensing_classes)
        self.circuit = build_variational_quantum_circuit(
            n_qubits=n_qubits,
            n_layers=n_layers,
            noise_strength=noise_strength,
        )

    def forward(self, compressed_features: torch.Tensor) -> torch.Tensor:
        projected_features = self.input_projection(compressed_features)
        angle_features = torch.tanh(projected_features) * np.pi

        quantum_outputs = []
        for sample_features in angle_features:
            sample_output = self.circuit(sample_features, self.quantum_weights)
            # float64 tensors from pennylane --> float32 torch tensors
            sample_output_tensor = torch.stack(tuple(sample_output)).to(
                device=projected_features.device,
                dtype=projected_features.dtype,
            )
            quantum_outputs.append(sample_output_tensor)

        quantum_feature_batch = torch.stack(quantum_outputs, dim=0)
        return self.output_head(quantum_feature_batch)
