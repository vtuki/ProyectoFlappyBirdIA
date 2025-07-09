import pygame # Añadido para el manejo de eventos de Pygame
import sys    # Añadido para sys.exit
import numpy as np # Necesario para operaciones numéricas

import matplotlib
import matplotlib.pyplot as plt

import random
import torch
from torch import nn
import yaml
import os

# Importa las clases de otros archivos del proyecto
from experience_replay import ReplayMemory
from dqn import DQN
from flappy_env import FlappyBirdEnv # Importa tu entorno Flappy Bird

from datetime import datetime, timedelta
import argparse
import itertools

# Para imprimir fecha y hora
DATE_FORMAT = "%m-%d %H:%M:%S"

# Directorio para guardar información de las ejecuciones
RUNS_DIR = "runs"
os.makedirs(RUNS_DIR, exist_ok=True) # Crea el directorio si no existe

# 'Agg': usado para generar gráficos como imágenes y guardarlos en un archivo en lugar de renderizarlos en pantalla
matplotlib.use('Agg')

# Configura el dispositivo (CUDA para GPU si está disponible, de lo contrario CPU)
# Forzar CPU si es necesario, a veces la GPU no es más rápida debido a la sobrecarga de mover datos.
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu' # Descomenta para forzar CPU

# Agente de Deep Q-Learning
class Agent():
    def __init__(self, hyperparameter_set):
        # Cargar hiperparámetros desde el archivo YAML
        with open('hyperparameters.yml', 'r') as file:
            all_hyperparameter_sets = yaml.safe_load(file)
            hyperparameters = all_hyperparameter_sets[hyperparameter_set]
            # print(hyperparameters)

        self.hyperparameter_set = hyperparameter_set

        # Hiperparámetros (ajustables)
        self.env_id             = hyperparameters['env_id']          # ID del entorno
        self.learning_rate_a    = hyperparameters['learning_rate_a'] # Tasa de aprendizaje (alpha)
        self.discount_factor_g  = hyperparameters['discount_factor_g'] # Factor de descuento (gamma)
        self.network_sync_rate  = hyperparameters['network_sync_rate'] # Frecuencia de sincronización de la red objetivo
        self.replay_memory_size = hyperparameters['replay_memory_size'] # Tamaño de la memoria de replay
        self.mini_batch_size    = hyperparameters['mini_batch_size'] # Tamaño del mini-lote
        self.epsilon_init       = hyperparameters['epsilon_init']    # Epsilon inicial (1 = 100% acciones aleatorias)
        self.epsilon_decay      = hyperparameters['epsilon_decay']   # Tasa de decaimiento de epsilon
        self.epsilon_min        = hyperparameters['epsilon_min']     # Valor mínimo de epsilon
        self.stop_on_reward     = hyperparameters['stop_on_reward']  # Detener entrenamiento al alcanzar esta recompensa
        self.fc1_nodes          = hyperparameters['fc1_nodes']       # Nodos en la primera capa
        # Parámetros opcionales específicos del entorno, por defecto diccionario vacío
        self.env_make_params    = hyperparameters.get('env_make_params',{}) 
        self.enable_double_dqn  = hyperparameters['enable_double_dqn'] # Bandera Double DQN
        self.enable_dueling_dqn = hyperparameters['enable_dueling_dqn'] # Bandera Dueling DQN

        # Red Neuronal
        self.loss_fn = nn.MSELoss()          # Función de pérdida NN (MSE)
        self.optimizer = None                # Optimizador NN. Se inicializa más tarde.

        # Rutas para información de la ejecución
        self.LOG_FILE   = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}.log')
        self.MODEL_FILE = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}.pt')
        self.GRAPH_FILE = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}.png')

    # Método para ejecutar el agente (entrenamiento o prueba)
    def run(self, is_training=True, render=False):
        if is_training:
            start_time = datetime.now()
            last_graph_update_time = start_time

            log_message = f"{start_time.strftime(DATE_FORMAT)}: Iniciando entrenamiento..."
            print(log_message)
            with open(self.LOG_FILE, 'w') as file:
                file.write(log_message + '\n')

        # Crear instancia del entorno Flappy Bird
        # NOTA: Tu FlappyBirdEnv no es un entorno Gymnasium estándar,
        # así que lo instanciamos directamente.
        env = FlappyBirdEnv()

        # Número de posibles acciones (saltar o no saltar)
        num_actions = 2 # env.action_space.n # Tu env no tiene action_space.n directamente

        # Obtener el tamaño del espacio de observación (estado)
        num_states = env._get_state().shape[0] # Tu env usa _get_state().shape[0]

        # Lista para seguir las recompensas recolectadas por episodio.
        rewards_per_episode = []

        # Crear la red de políticas y la red objetivo.
        policy_dqn = DQN(num_states, num_actions, self.fc1_nodes, self.enable_dueling_dqn).to(device)

        if is_training:
            # Inicializar epsilon
            epsilon = self.epsilon_init

            # Inicializar la memoria de replay
            memory = ReplayMemory(self.replay_memory_size)

            # Crear la red objetivo y hacerla idéntica a la red de políticas
            target_dqn = DQN(num_states, num_actions, self.fc1_nodes, self.enable_dueling_dqn).to(device)
            target_dqn.load_state_dict(policy_dqn.state_dict())

            # Optimizador de la red de políticas (Adam)
            self.optimizer = torch.optim.Adam(policy_dqn.parameters(), lr=self.learning_rate_a)

            # Lista para seguir el decaimiento de epsilon
            epsilon_history = []

            # Contador de pasos. Usado para sincronizar política => red objetivo.
            step_count = 0

            # Mejor recompensa (para guardar el mejor modelo)
            best_reward = -9999999
        else:
            # Cargar la política aprendida
            policy_dqn.load_state_dict(torch.load(self.MODEL_FILE))

            # Cambiar el modelo a modo de evaluación (deshabilita dropout, batch norm, etc.)
            policy_dqn.eval()

        # Entrenar INDEFINIDAMENTE, detén la ejecución manualmente cuando estés satisfecho
        for episode in itertools.count():

            state = env.reset() # Inicializar entorno.
            state = torch.tensor(state, dtype=torch.float, device=device) # Convertir estado a tensor en el dispositivo

            terminated = False      # True cuando el agente alcanza la meta o falla
            truncated = False       # (Tu env no usa 'truncated', pero se mantiene por compatibilidad si lo añades)
            episode_reward = 0.0    # Acumulador de recompensas por episodio

            # Realizar acciones hasta que el episodio termine o alcance la recompensa máxima
            while(not terminated and not truncated and episode_reward < self.stop_on_reward):
                # ===> MANEJO DE EVENTOS DE PYGAME (¡CRUCIAL!) <===
                # Este bucle DEBE estar aquí para que la ventana de Pygame responda.
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        env.close()
                        sys.exit()

                # Seleccionar acción basada en la estrategia epsilon-greedy
                if is_training and random.random() < epsilon:
                    action = random.randrange(num_actions) # Seleccionar acción aleatoria
                    action = torch.tensor(action, dtype=torch.int64, device=device)
                else:
                    # Seleccionar la mejor acción
                    with torch.no_grad(): # No calcular gradientes para la inferencia
                        # state.unsqueeze(dim=0): PyTorch espera una dimensión de lote, así que la añadimos
                        q_values = policy_dqn(state.unsqueeze(dim=0)).squeeze()
                        action = q_values.argmax() # Encontrar el índice del elemento más grande (la mejor acción)

                # Ejecutar acción en el entorno
                new_state, reward, terminated, info = env.step(action.item()) # Tu env no devuelve 'truncated'

                # Acumular recompensas
                episode_reward += reward

                # Convertir nuevo estado y recompensa a tensores en el dispositivo
                new_state = torch.tensor(new_state, dtype=torch.float, device=device)
                reward = torch.tensor(reward, dtype=torch.float, device=device)

                if is_training:
                    # Guardar experiencia en la memoria
                    memory.append((state, action, new_state, reward, terminated))

                    # Incrementar contador de pasos
                    step_count += 1

                # Mover al siguiente estado
                state = new_state

            # Guardar la recompensa total del episodio
            rewards_per_episode.append(episode_reward)

            # Guardar el modelo cuando se obtiene una nueva mejor recompensa.
            if is_training:
                if episode_reward > best_reward:
                    log_message = (f"{datetime.now().strftime(DATE_FORMAT)}: Nueva mejor recompensa {episode_reward:0.1f} "
                                   f"({(episode_reward-best_reward)/abs(best_reward)*100:+.1f}%) en el episodio {episode}, guardando modelo...")
                    print(log_message)
                    with open(self.LOG_FILE, 'a') as file:
                        file.write(log_message + '\n')

                    torch.save(policy_dqn.state_dict(), self.MODEL_FILE)
                    best_reward = episode_reward


                # Actualizar gráfico cada X segundos
                current_time = datetime.now()
                if current_time - last_graph_update_time > timedelta(seconds=10):
                    self.save_graph(rewards_per_episode, epsilon_history)
                    last_graph_update_time = current_time

                # Si se ha recolectado suficiente experiencia (más que el tamaño del mini-lote)
                if len(memory) > self.mini_batch_size:
                    mini_batch = memory.sample(self.mini_batch_size)
                    self.optimize(mini_batch, policy_dqn, target_dqn)

                    # Decaer epsilon
                    epsilon = max(epsilon * self.epsilon_decay, self.epsilon_min)
                    epsilon_history.append(epsilon)

                    # Copiar la red de políticas a la red objetivo después de cierto número de pasos
                    if step_count > self.network_sync_rate:
                        target_dqn.load_state_dict(policy_dqn.state_dict())
                        step_count = 0

            # Imprimir el progreso del episodio
            print(f"Episodio {episode} | Recompensa: {episode_reward:.1f} | Epsilon: {epsilon:.4f} | Memoria: {len(memory)}")


    # Guarda los gráficos de rendimiento
    def save_graph(self, rewards_per_episode, epsilon_history):
        fig = plt.figure(1)

        # Graficar recompensas promedio (eje Y) vs episodios (eje X)
        mean_rewards = np.zeros(len(rewards_per_episode))
        for x in range(len(mean_rewards)):
            mean_rewards[x] = np.mean(rewards_per_episode[max(0, x-99):(x+1)]) # Promedio de las últimas 100 recompensas
        plt.subplot(121) # Gráfica en una cuadrícula de 1 fila x 2 columnas, en la celda 1
        plt.xlabel('Episodios')
        plt.ylabel('Recompensas Promedio')
        plt.plot(mean_rewards)

        # Graficar decaimiento de epsilon (eje Y) vs episodios (eje X)
        plt.subplot(122) # Gráfica en una cuadrícula de 1 fila x 2 columnas, en la celda 2
        plt.xlabel('Episodios')
        plt.ylabel('Decaimiento de Epsilon')
        plt.plot(epsilon_history)

        plt.subplots_adjust(wspace=1.0, hspace=1.0) # Ajustar espaciado entre subgráficas

        # Guardar gráficos
        fig.savefig(self.GRAPH_FILE)
        plt.close(fig) # Cierra la figura para liberar memoria


    # Optimizar la red de políticas
    def optimize(self, mini_batch, policy_dqn, target_dqn):
        # Transponer la lista de experiencias y separar cada elemento
        states, actions, new_states, rewards, terminations = zip(*mini_batch)

        # Apilar tensores para crear tensores de lote
        states = torch.stack(states)
        actions = torch.stack(actions)
        new_states = torch.stack(new_states)
        rewards = torch.stack(rewards)
        terminations = torch.tensor(terminations).float().to(device)

        with torch.no_grad(): # No calcular gradientes para el cálculo del Q objetivo
            if self.enable_double_dqn:
                # Double DQN: Usa la red de política para seleccionar la mejor acción en el siguiente estado,
                # y la red objetivo para estimar su valor Q.
                best_actions_from_policy = policy_dqn(new_states).argmax(dim=1)

                target_q = rewards + (1 - terminations) * self.discount_factor_g * \
                                target_dqn(new_states).gather(dim=1, index=best_actions_from_policy.unsqueeze(dim=1)).squeeze()
            else:
                # DQN estándar: Calcula los valores Q objetivo (retornos esperados)
                target_q = rewards + (1 - terminations) * self.discount_factor_g * target_dqn(new_states).max(dim=1)[0]

        # Calcular los valores Q de la política actual
        current_q = policy_dqn(states).gather(dim=1, index=actions.unsqueeze(dim=1)).squeeze()

        # Calcular la pérdida (Error Cuadrático Medio)
        loss = self.loss_fn(current_q, target_q)

        # Optimizar el modelo (retropropagación)
        self.optimizer.zero_grad()  # Limpiar gradientes
        loss.backward()             # Calcular gradientes
        self.optimizer.step()       # Actualizar los parámetros de la red (pesos y sesgos)

# Punto de entrada del script
if __name__ == '__main__':
    # Parsear argumentos de línea de comandos
    parser = argparse.ArgumentParser(description='Entrenar o probar modelo.')
    parser.add_argument('hyperparameters', help='Nombre del conjunto de hiperparámetros del archivo hyperparameters.yml (e.g., flappybird1)')
    parser.add_argument('--train', help='Modo de entrenamiento', action='store_true')
    args = parser.parse_args()

    dql = Agent(hyperparameter_set=args.hyperparameters)

    if args.train:
        dql.run(is_training=True)
    else:
        dql.run(is_training=False, render=True)