# Define la memoria para Experience Replay
from collections import deque
import random

class ReplayMemory():
    def __init__(self, maxlen, seed=None):
        # Crea una cola con un tamaño máximo especificado
        self.memory = deque([], maxlen=maxlen)

        # Semilla opcional para reproducibilidad
        if seed is not None:
            random.seed(seed)

    def append(self, transition):
        # Añade una nueva transición (estado, acción, nuevo_estado, recompensa, terminado) a la memoria
        self.memory.append(transition)

    def sample(self, sample_size):
        # Muestra aleatoriamente un número 'sample_size' de transiciones de la memoria
        return random.sample(self.memory, sample_size)

    def __len__(self):
        # Retorna el número actual de transiciones en la memoria
        return len(self.memory)