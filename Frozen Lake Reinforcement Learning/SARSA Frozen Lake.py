"""
Reinforcement Learning using SARSA (State-Action-Reward-State-Action)
It is an on-policy, model-free reinforcement learning algorithm.

SARSA:
Q(s, a) = Q(s, a) + alpha * [reward + gamma * Q(s', a') - Q(s, a)]
"""

import time
import random
import numpy as np
import gymnasium as gym

env = gym.make("FrozenLake-v1", render_mode="human", map_name="4x4", is_slippery=False)

alpha = 0.1
gamma = 0.99
epsilon = 0.4
epsilon_decay = 0.94
min_epsilon = 0.01
episodes = 50

q_table = np.zeros((env.observation_space.n, env.action_space.n))

action_names = {
    0: "LEFT",
    1: "DOWN",
    2: "RIGHT",
    3: "UP"
}

def choose_action(q_table, state, epsilon):
    if random.uniform(0, 1) < epsilon:
        return env.action_space.sample()
    return np.argmax(q_table[state])

for episode in range(episodes):
    state, info = env.reset()
    done = False

    action = choose_action(q_table, state, epsilon)

    while not done:
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        if terminated and reward == 0:
            reward = -10
        if next_state == state and not terminated:
            reward = -10
        if terminated and reward == 1:
            reward = 10

        old_value = q_table[state, action]

        if done:
            q_table[state, action] = old_value + alpha * (reward - old_value)
        else:
            next_action = choose_action(q_table, next_state, epsilon)
            next_value = q_table[next_state, next_action]

            q_table[state, action] = old_value + alpha * (
                reward + gamma * next_value - old_value
            )

            action = next_action

        print(
            "episode:", episode + 1,
            "state:", state,
            "action:", action_names[action] if not done else action_names[action],
            "next_state:", next_state,
            "reward:", reward,
            "epsilon:", epsilon
        )

        state = next_state
        time.sleep(0.1)

    epsilon = max(min_epsilon, epsilon * epsilon_decay)

env.close()

print("\nQ-Table:")
print(q_table)

"""
Learning Rate: Determines how much the new information overrides the old Q-value. A high alpa learnms fast but can be unstable
Discount Factor: Determines the importance of future rewards. Means the agentt cares deeply about long term goals


"""