# Quantum Circuit Synthesis Environment for Reinforcement Learning

This project provides a quantum circuit synthesis environment for reinforcement learning. The environment is built on top of the Gymnasium framework.

## Installation

To install the environment, you need to have Python and pip installed on your system. If you don't have them installed, you can download them from the official Python website.

Once you have Python and pip installed, you can install the environment by running the following command in your terminal:

```sh
pip install qc_syn
```

## Usage

To create a new instance of the environment, you can use the `gym.make` function:

```python
import gymnasium as gym
import qc_syn

env = gym.make("qc_syn/QuantumCircuit-v0", qubit_count=4)
observation, info = env.reset()

for _ in range(1000):
    action = env.action_space.sample()  # agent policy that uses the observation and info
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()

env.close()
```

## Contributing

Contributions are welcome! Please feel free to submit a pull request.

## License

This project is licensed under the terms of the MIT license.
