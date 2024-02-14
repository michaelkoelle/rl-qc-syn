"""Example of using the QuantumCircuit-v0 environment. """

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
