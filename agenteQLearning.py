import numpy as np
import random
import pickle

class QAgent:
    def __init__(self):
        self.q_table = {}
        self.alpha = 0.04
        self.gamma = 0.9
        self.epsilon = 0.3  # Probabilidad de explorar
        self.initial_epsilon = self.epsilon

    def get_q(self, state, action):
        return self.q_table.get((state, action), 0.0)

    def choose_action(self, state, viewing_mode=False):
        if ((random.random() < self.epsilon) and not viewing_mode):
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
    
    def decay_epsilon(self, min_epsilon=0.01, decay_rate=0.995):
        if self.epsilon > min_epsilon:
            self.epsilon *= decay_rate
        
    def save(self, path="qtable.pkl"):
        with open(path, "wb") as f:
            pickle.dump(self.q_table, f)
        
    def load(self, path="qtable.pkl"):
        with open(path, "rb") as f:
            self.q_table = pickle.load(f)

    def saveMetaData(self, episodes, factor_disc_x, factor_disc_y, path="models/metadata.txt", boosted=False):
        metadata = {
            "alpha": self.alpha,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "episodes": episodes,
            "factor_discretizacion_x": factor_disc_x,
            "factor_discretizacion_y": factor_disc_y,
            "factores_estado": "dx, dy",
        }
        if boosted:
            metadata["boosted"] = "Si"
        else:
            metadata["boosted"] = "No"
        with open(path, "w") as f:
            for k, v in metadata.items():
                f.write(f"{k}: {v}\n")
