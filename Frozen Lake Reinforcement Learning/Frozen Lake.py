import time
import random
import numpy as np
import gymnasium as gym

env = gym.make("FrozenLake-v1", render_mode="human", map_name="4x4", is_slippery=False)

alpha = 0.1
gamma = 0.99
epsilon = .4
epsilon_decay = 0.940
min_epsilon = 0.01
episodes = 50

q_table = np.zeros((env.observation_space.n, env.action_space.n))

action_names = {
    0: "LEFT",
    1: "DOWN",
    2: "RIGHT",
    3: "UP"
}

for episode in range(episodes):
    state, info = env.reset()
    done = False

    while not done:
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state])

        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        if terminated and reward == 0:
            reward = -10
        if next_state == state and not terminated:
            reward = -10
        if terminated and reward == 1:
            reward = 10
        

        old_value = q_table[state, action]
        next_max = np.max(q_table[next_state])

        q_table[state, action] = old_value + alpha * (
            reward + gamma * next_max - old_value
        )

        print(
            "episode:", episode + 1,
            "state:", state,
            "action:", action_names[action],
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
Q-Learning: is a model-free, off policy reinforcement learning algorithm used to find the optimal action-selection poliocy by learrning the value Q-value of state-action-pairs

Q(s,a) <- Q(s,a) + a(r + Y * Q(S', A') - Q(S,A)

a (Alpha): is the learning rate determing how much new information affacts the old Q-values
Y (Gamma) is the discount factor which balances immediate rewards with future rewards
R (reward) is the reward received for taking action A in state S 
Epsilon: controls the exploration vs exploitation trade off: 
    Exploration -> Try random actions to discover new possibilities 
    Exploitation -> choose the best action the agent already knows

"""