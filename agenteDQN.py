import numpy as np
import random
import pickle
import tensorflow as tf
from collections import deque

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size # e.g., (dist_x_pipe, dist_y_pipe)
        self.action_size = action_size # 0 (no jump), 1 (jump)
        self.memory = deque(maxlen=2000) # Replay buffer
        self.gamma = 0.95    # Discount rate
        self.epsilon = 1.0   # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()
        self.batch_size = 32

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = tf.keras.models.Sequential()
        # Input layer: State is a tuple (dist_x_pipe, dist_y_pipe), so 2 inputs
        model.add(tf.keras.layers.Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(tf.keras.layers.Dense(24, activation='relu'))
        model.add(tf.keras.layers.Dense(self.action_size, activation='linear')) # Output: Q-values for each action
        model.compile(loss='mse',
                      optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate))
        return model

    def update_target_model(self):
        # Copies weights from the main model to the target model
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def choose_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size) # Explore
        # Exploit: Predict Q-values from the model
        state_reshaped = np.reshape(state, [1, self.state_size])
        q_values = self.model.predict(state_reshaped, verbose=0)
        return np.argmax(q_values[0])

    def learn(self):
        if len(self.memory) < self.batch_size:
            return

        # Sample a minibatch from the replay buffer
        minibatch = random.sample(self.memory, self.batch_size)
        
        # Initialize arrays for states, targets, and next_states
        states = np.array([t[0] for t in minibatch])
        actions = np.array([t[1] for t in minibatch])
        rewards = np.array([t[2] for t in minibatch])
        next_states = np.array([t[3] for t in minibatch])
        dones = np.array([t[4] for t in minibatch])

        # Predict Q-values for current states
        q_values_current = self.model.predict(states, verbose=0)
        
        # Predict Q-values for next states using the target model
        q_values_next = self.target_model.predict(next_states, verbose=0)
        
        # Initialize target Q-values with current Q-values
        targets = np.copy(q_values_current)

        for i in range(self.batch_size):
            if dones[i]:
                # If episode is done, target Q-value is just the reward
                targets[i][actions[i]] = rewards[i]
            else:
                # Bellman equation: Q(s,a) = r + gamma * max_a'(Q(s',a'))
                targets[i][actions[i]] = rewards[i] + self.gamma * np.amax(q_values_next[i])

        # Train the model
        self.model.fit(states, targets, epochs=1, verbose=0)

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def save(self, path="model_dqn.h5"):
        self.model.save(path)
        with open(path.replace(".h5", "_epsilon.pkl"), "wb") as f:
            pickle.dump(self.epsilon, f)
        
    def load(self, path="model_dqn.h5"):
        self.model = tf.keras.models.load_model(path)
        self.target_model = tf.keras.models.load_model(path) # Load target model as well
        with open(path.replace(".h5", "_epsilon.pkl"), "rb") as f:
            self.epsilon = pickle.load(f)