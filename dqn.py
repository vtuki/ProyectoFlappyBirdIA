import torch
from torch import nn
import torch.nn.functional as F

class DQN(nn.Module):
    # Constructor de la red DQN
    def __init__(self, state_dim, action_dim, hidden_dim=256, enable_dueling_dqn=True):
        super(DQN, self).__init__()

        self.enable_dueling_dqn = enable_dueling_dqn # Bandera para habilitar Dueling DQN

        # Primera capa completamente conectada
        self.fc1 = nn.Linear(state_dim, hidden_dim)

        if self.enable_dueling_dqn:
            # Flujo de valor (Value stream)
            self.fc_value = nn.Linear(hidden_dim, 256)
            self.value = nn.Linear(256, 1) # Salida de valor escalar

            # Flujo de ventajas (Advantages stream)
            self.fc_advantages = nn.Linear(hidden_dim, 256)
            self.advantages = nn.Linear(256, action_dim) # Salida de ventajas para cada acción

        else:
            # Si Dueling DQN está deshabilitado, es una red DQN estándar
            self.output = nn.Linear(hidden_dim, action_dim)

    # Método forward para el paso de inferencia de la red
    def forward(self, x):
        x = F.relu(self.fc1(x)) # Aplicar ReLU a la salida de la primera capa

        if self.enable_dueling_dqn:
            # Cálculo del valor (V)
            v = F.relu(self.fc_value(x))
            V = self.value(v)

            # Cálculo de las ventajas (A)
            a = F.relu(self.fc_advantages(x))
            A = self.advantages(a)

            # Calcular Q-values usando la fórmula de Dueling DQN: Q = V + (A - mean(A))
            Q = V + A - torch.mean(A, dim=1, keepdim=True)

        else:
            # Si Dueling DQN está deshabilitado, la salida directa son los Q-values
            Q = self.output(x)

        return Q

# Ejemplo de uso (solo se ejecuta si este archivo es el principal)
if __name__ == '__main__':
    state_dim = 12
    action_dim = 2
    net = DQN(state_dim, action_dim)
    state = torch.randn(10, state_dim) # Simula un lote de 10 estados
    output = net(state)
    print(output)