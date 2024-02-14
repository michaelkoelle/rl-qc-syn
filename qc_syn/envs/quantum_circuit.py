"""This module contains the environment for the Quantum Circuit Synthesis Problem"""

from dataclasses import asdict, dataclass
from itertools import permutations
from typing import Any, Dict, List, Literal, Optional, Type

import numpy as np
import pennylane as qml
from gymnasium import Env
from gymnasium.spaces import Box, MultiDiscrete
from numpy.typing import NDArray
from pennylane.operation import Operation


@dataclass
class GateConfig:
    """Dataclass for the configuration of a gate in a quantum circuit."""

    op: Type[Operation]
    wires: NDArray[Any]


class QuantumCircuit(Env[NDArray[Any], NDArray[Any]]):
    """Environment for Quantum Circuits"""

    metadata = {"render_modes": ["human"]}
    step_penalty = -1
    gate_set = [
        qml.CNOT,
        qml.Hadamard,
        qml.T,
        qml.S,
        qml.Identity,
    ]

    def __init__(
        self,
        qubit_count: int = 3,
        gate_set: Optional[List[Type[Operation]]] = None,
        gate_count: Optional[int] = None,
        max_gate_count: int = 10,
        target_state: Optional[List[float]] = None,
        epsilon: float = 0.1,
        reward_mode: (
            Literal["linear-depended"] | Literal["basic"] | Literal["dense-distance"]
        ) = "basic",
        target_gen_difficulty: Optional[
            Literal["easy"] | Literal["medium"] | Literal["hard"]
        ] = None,
    ) -> None:
        super().__init__()

        if gate_set is not None:
            self.gate_set = gate_set
        else:  # use clifford + T gates as default
            self.gate_set: List[Type[Operation]] = [
                qml.CNOT,
                qml.Hadamard,
                qml.T,
                qml.S,
                qml.Identity,
            ]

        self.qubit_count = qubit_count
        self.max_gate_count = max_gate_count
        self.remaining_gate_count = self.max_gate_count

        if gate_count is None:
            self.gate_count = int(self.max_gate_count // 2)
        else:
            self.gate_count = gate_count

        self.wire_permutations = self.get_wire_permutations(self.gate_set)

        """Action space divided into two integers: 1. gate ; 2. wire combination"""
        self.action_space = MultiDiscrete([len(self.gate_set), len(self.wire_permutations)])
        self.device = qml.device("default.qubit", wires=self.qubit_count, shots=None)

        """row: current state (first half real, second half imaginary ) and then target state
        (first half real, second half imaginary) Observation Space: kartesian product of 
        2**number_qubits*4 countably infinite sets"""
        self.observation_space = Box(
            low=-1.0, high=1.0, shape=(2**self.qubit_count * 4,), dtype=np.float32
        )

        self.observation = np.zeros((2**self.qubit_count * 4,), dtype=np.float32)
        self.target_state = None
        self.target_gen_difficulty = target_gen_difficulty
        self.list_of_applied_gates: List[GateConfig] = []
        self.full_circuit = qml.QNode(self.quantum_circuit_function, self.device)
        if target_state is not None:
            self.target_state = target_state
            (
                self.observation[(2**self.qubit_count * 2) : (2**self.qubit_count * 3)],
                self.observation[(2**self.qubit_count * 3) :],
            ) = (
                np.array(target_state).real,
                np.array(target_state).imag,
            )
        else:
            self.generator()

        self.epsilon = epsilon
        self.reward_mode = reward_mode

    def get_wire_permutations(self, list_of_gates: List[Type[Operation]]):
        """Determines all possible wirecombinations applicable with a given set of quantumgates"""
        wires = list(map(lambda gate: gate.num_wires, list_of_gates))
        maxwires = max(wires) if len(wires) > 0 else 0  # type: ignore
        wire_permutations = []
        for perm in list(permutations(range(self.qubit_count), maxwires)):
            wire_permutations.append(perm)
        return wire_permutations

    def generator(self):
        """A generator function enabling generation of random targets reachable through either
        a random or a fixed (given by parameter 'random_testing_difficulty') number of
        quantumgates
        """
        if self.target_gen_difficulty is None:
            self.generate_target(self.gate_count)
        else:
            if self.target_gen_difficulty == "hard":
                number_of_gates = self.np_random.integers(7, 11)
            elif self.target_gen_difficulty == "medium":
                number_of_gates = self.np_random.integers(4, 7)
            else:  # if nothing is given easy is default
                number_of_gates = self.np_random.integers(1, 4)
            self.generate_target(number_of_gates)

    def generate_target(self, gates: int):
        "Generates a random targetstate reachable through a sequence of 'gates' quantum-gates"
        self.list_of_applied_gates = []

        previous_states = []
        circuit = qml.QNode(self.quantum_circuit_function, self.device)
        previous_states.append(np.array(circuit()))
        gatecount = 0
        tries = len(self.gate_set) * 2
        while gatecount < gates:
            passed_by: bool = False
            next_action = self.action_space.sample()
            self.list_of_applied_gates.append(self.action_to_gate_config(next_action))
            circuit = qml.QNode(self.quantum_circuit_function, self.device)
            next_state = np.array(circuit())
            conj_next_state = np.conjugate(next_state)
            for state in previous_states:
                difference = 1 - (np.abs(np.dot(state, conj_next_state))) ** 2
                if difference < 0.001:
                    passed_by = True
            if tries > 0:
                if passed_by:
                    self.list_of_applied_gates.pop()
                    tries -= 1
                else:
                    previous_states.append(next_state)
                    gatecount += 1
                    tries = len(self.gate_set) * 2
            else:
                self.list_of_applied_gates = []
                previous_states = []
                circuit = qml.QNode(self.quantum_circuit_function, self.device)
                previous_states.append(np.array(circuit()))
                gatecount = 0
                tries = len(self.gate_set) * 2

        circuit = qml.QNode(self.quantum_circuit_function, self.device)
        target_state = circuit()

        (
            self.observation[2**self.qubit_count * 2 : 2**self.qubit_count * 3],
            self.observation[2**self.qubit_count * 3 :],
        ) = (
            np.array(target_state).real,
            np.array(target_state).imag,
        )

        self.list_of_applied_gates = []

    def quantum_circuit_function(self):
        """The Function representing the actually quantum circuit"""
        for gate_config in self.list_of_applied_gates:
            num_wires = gate_config.op.num_wires
            wires = gate_config.wires[:num_wires]
            gate_config.op(wires=wires)
        return qml.state()

    def action_to_gate_config(self, action: NDArray[Any]) -> GateConfig:
        "Maps integer (action) to specific gate with specific qubits as inputs"
        quantumgate = self.gate_set[int(action[0])]
        wires = self.wire_permutations[int(action[1])]
        return GateConfig(op=quantumgate, wires=wires)

    def step(self, action: NDArray[Any]):
        """Method for stepping through the environment"""

        self.remaining_gate_count -= 1
        next_gate_applied = self.action_to_gate_config(action)
        self.list_of_applied_gates.append(next_gate_applied)
        info = asdict(next_gate_applied)

        self.full_circuit = qml.QNode(self.quantum_circuit_function, self.device)

        complex_result = self.full_circuit()
        (
            self.observation[: 2**self.qubit_count],
            self.observation[2**self.qubit_count : 2**self.qubit_count * 2],
        ) = (complex_result.real, complex_result.imag)

        reward, terminated, truncated = self.calculate_reward_and_gamestatus()

        return self.observation, reward, terminated, truncated, info

    def get_absolute_state_diff(self):
        """Returns absolute difference between current state and target state"""
        real_sum, imag_sum = 0, 0
        for entry in range(2**self.qubit_count):
            real_sum += (
                self.observation[entry] * self.observation[2**self.qubit_count * 2 + entry]
                + self.observation[2**self.qubit_count + entry]
                * self.observation[2**self.qubit_count * 3 + entry]
            )
            imag_sum += (
                self.observation[entry] * self.observation[2**self.qubit_count * 3 + entry]
                - self.observation[2**self.qubit_count + entry]
                * self.observation[2**self.qubit_count * 2 + entry]
            )
        return 1 - (real_sum**2 + imag_sum**2)

    def calculate_reward_and_gamestatus(self):
        """Calculates the reward of the action and the current gamestatus (i.e., checks if game
        is terminated or truncated) Final reward: stepreward * calculation_length"""
        state_diff = self.get_absolute_state_diff()
        terminated = state_diff <= self.epsilon
        truncated = self.remaining_gate_count <= 0
        reward = self.step_penalty

        if terminated:
            reward += self.remaining_gate_count
        elif truncated:
            if self.reward_mode == "dense-distance":
                reward *= (self.max_gate_count // 2) * state_diff
        else:
            if self.reward_mode == "linear-depended":
                reward = state_diff * reward

        return reward, terminated, truncated

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        """Method reseting the environment to initial conditions"""
        super().reset(seed=seed)
        self.remaining_gate_count = self.max_gate_count
        self.observation[: 2**self.qubit_count * 2] = 0
        self.list_of_applied_gates = []
        if self.target_state is None:
            self.generator()
        return self.observation, {}

    def render(self, mode: str = "human") -> None:
        if mode == "human":
            print(qml.draw(self.full_circuit)())
