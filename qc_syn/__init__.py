"""Gym environment for quantum circuit synthesis."""

from gymnasium.envs.registration import register

register(
    id="qc_syn/QuantumCircuit-v0",
    entry_point="qc_syn.envs.quantum_circuit:QuantumCircuit",
)
