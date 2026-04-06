import numpy as np
import random
import matplotlib.pyplot as plt

# Environment Setup

class TrafficEnv:
    def __init__(self):
        self.reset()

    def reset(self):
        self.queue = np.random.randint(5, 20)
        return self.queue

    def step(self, action):
        # action: 0 = keep green, 1 = switch

        if action == 0:
            # cars pass
            self.queue -= np.random.randint(1, 5)
        else:
            # switching causes delay
            self.queue += np.random.randint(1, 3)

        self.queue = max(self.queue, 0)

        reward = -self.queue  # minimize queue
        return self.queue, reward

# Q-Learning Agent

q_table = np.zeros((30, 0))  # states: queue size (0–29), actions: 2

alpha = 0.1
gamma = 0.9
epsilon = 0.2

# Training Loop

env = TrafficEnv()
rewards = []

for episode in range(200):
    state = env.reset()
    state = min(state, 29)  # FIX

    total_reward = 0

    for step in range(50):

        # Choose action
        if random.uniform(0,1) < epsilon:
            action = random.choice([0,1])
        else:
            action = np.argmax(q_table[state])

        next_state, reward = env.step(action)
        next_state = min(next_state, 29)  # FIX

        # Update Q-table
        q_table[state, action] = q_table[state, action] + alpha * (
            reward + gamma * np.max(q_table[next_state]) - q_table[state, action]
        )

        state = next_state
        total_reward += reward

    rewards.append(total_reward)
    
# Plot Learning

plt.plot(rewards)
plt.title("RL Learning Progress")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.show()