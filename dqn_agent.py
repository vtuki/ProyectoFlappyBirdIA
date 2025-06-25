import numpy as np
import random
from collections import deque
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000) # Experiencia Replay Buffer
        self.gamma = 0.95    # Factor de descuento (qué tan importantes son las recompensas futuras)
        self.epsilon = 1.0   # Factor de exploración (probabilidad de tomar una acción aleatoria)
        self.epsilon_min = 0.01 # Mínimo de epsilon
        self.epsilon_decay = 0.998 # Tasa de decaimiento de epsilon
        self.learning_rate = 0.001 # Tasa de aprendizaje
        self.model = self._build_model()
        self.target_model = self._build_model() # Target Network
        self.update_target_model()

    def _build_model(self):
        # Neural Net para aproximar la función Q
        model = Sequential()
        model.add(Dense(64, input_dim=self.state_size, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(self.action_size, activation='linear')) # Salida lineal para valores Q
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def update_target_model(self):
        # Copia los pesos del modelo principal al modelo de destino
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size) # Exploración: acción aleatoria
        # Explotación: elige la mejor acción según el modelo
        q_values = self.model.predict(state[np.newaxis, :]) # Añadir una dimensión de batch
        return np.argmax(q_values[0])

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return

        minibatch = random.sample(self.memory, batch_size)
        
        # Preparar los datos para el entrenamiento
        states = np.array([t[0] for t in minibatch])
        actions = np.array([t[1] for t in minibatch])
        rewards = np.array([t[2] for t in minibatch])
        next_states = np.array([t[3] for t in minibatch])
        dones = np.array([t[4] for t in minibatch])

        # Calcular los valores Q de destino
        target_q = self.model.predict(states)
        target_next_q = self.target_model.predict(next_states) # Usar la target network

        for i in range(batch_size):
            if dones[i]:
                target_q[i][actions[i]] = rewards[i]
            else:
                target_q[i][actions[i]] = rewards[i] + self.gamma * np.amax(target_next_q[i])
        
        # Entrenar el modelo
        self.model.fit(states, target_q, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)
        self.update_target_model()

    def save(self, name):
        self.model.save_weights(name)