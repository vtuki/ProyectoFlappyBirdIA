import numpy as np
import random

class QAgent:
    def __init__(self):
        self.q_table = {}
        self.alpha = 0.7
        self.gamma = 0.9
        self.epsilon = 0.1  # Probabilidad de explorar

    def get_q(self, state, action):
        return self.q_table.get((state, action), 0.0)

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.choice([0, 1])  # Explorar
        else:
            q_values = [self.get_q(state, a) for a in [0, 1]]
            return int(np.argmax(q_values))  # Explotar

    def learn(self, state, action, reward, next_state):
        best_next_q = max([self.get_q(next_state, a) for a in [0, 1]])
        current_q = self.get_q(state, action)

        # Actualización Q-Learning
        new_q = current_q + self.alpha * (reward + self.gamma * best_next_q - current_q)
        self.q_table[(state, action)] = new_q
