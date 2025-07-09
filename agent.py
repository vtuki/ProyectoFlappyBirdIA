import gymnasium as gym
import numpy as np

import matplotlib
import matplotlib.pyplot as plt

import random
import torch
from torch import nn
import yaml

from experience_replay import ReplayMemory
from dqn import DQN

from datetime import datetime, timedelta
import argparse
import itertools

# Importar la nueva clase de entorno FlappyBirdEnv desde la carpeta flappy_gym_env
from flappy_gym_env.flappy_bird_env import FlappyBirdEnv

import os

# Para imprimir fecha y hora
DATE_FORMAT = "%m-%d %H:%M:%S"

# Directorio para guardar información de las ejecuciones
RUNS_DIR = "runs"
os.makedirs(RUNS_DIR, exist_ok=True)

# 'Agg': utilizado para generar gráficos como imágenes y guardarlos en un archivo en lugar de renderizarlos en pantalla
matplotlib.use('Agg')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu' # forzar CPU, a veces la GPU no es más rápida que la CPU debido a la sobrecarga de mover datos a la GPU

# Agente de Deep Q-Learning
class Agent():

    def __init__(self, hyperparameter_set):
        with open('hyperparameters.yml', 'r') as file:
            all_hyperparameter_sets = yaml.safe_load(file)
            hyperparameters = all_hyperparameter_sets[hyperparameter_set]
            # print(hyperparameters)

        self.hyperparameter_set = hyperparameter_set

        # Hiperparámetros (ajustables)
        self.env_id             = hyperparameters['env_id']
        self.learning_rate_a    = hyperparameters['learning_rate_a']        # Tasa de aprendizaje (alpha)
        self.discount_factor_g  = hyperparameters['discount_factor_g']      # Factor de descuento (gamma)
        self.network_sync_rate  = hyperparameters['network_sync_rate']      # Tasa de sincronización de la red target
        self.replay_memory_size = hyperparameters['replay_memory_size']     # Tamaño de la memoria de repetición
        self.mini_batch_size    = hyperparameters['mini_batch_size']        # Tamaño del mini-batch para el entrenamiento
        self.epsilon_init       = hyperparameters['epsilon_init']           # Valor inicial de epsilon
        self.epsilon_decay      = hyperparameters['epsilon_decay']          # Tasa de decaimiento de epsilon
        self.epsilon_min        = hyperparameters['epsilon_min']            # Valor mínimo de epsilon
        self.stop_on_reward     = hyperparameters['stop_on_reward']         # Detener entrenamiento si la recompensa supera este valor
        self.fc1_nodes          = hyperparameters['fc1_nodes']              # Número de nodos en la primera capa oculta
        self.env_make_params    = hyperparameters.get('env_make_params', {}) # Parámetros adicionales para la creación del entorno

        # Habilitar Dueling DQN y Double DQN
        self.enable_double_dqn  = hyperparameters.get('enable_double_dqn', False)
        self.enable_dueling_dqn = hyperparameters.get('enable_dueling_dqn', False)

        # Inicializar render_mode para el entorno
        self.render_mode = None # Se establecerá en __main__

        # Crear el entorno Flappy Bird (ahora usando la versión Gymnasium)
        # Se pasa render_mode y los parámetros adicionales del entorno.
        # El entorno se cerrará y recreará en el main si el render_mode cambia.
        self.env = FlappyBirdEnv(render_mode=self.render_mode, **self.env_make_params)

        # Obtener el tamaño del espacio de estado y acción del entorno Gymnasium
        # El estado de esta versión del entorno es de 12 dimensiones por defecto sin LIDAR.
        self.num_states = self.env.observation_space.shape[0]
        self.num_actions = self.env.action_space.n

        # Inicializar el buffer de repetición de experiencias
        self.memory = ReplayMemory(self.replay_memory_size)

        # Inicializar las redes DQN (política y target)
        self.policy_dqn = DQN(
            self.num_states,
            self.num_actions,
            hidden_dim=self.fc1_nodes,
            enable_dueling_dqn=self.enable_dueling_dqn
        ).to(device)
        self.target_dqn = DQN(
            self.num_states,
            self.num_actions,
            hidden_dim=self.fc1_nodes,
            enable_dueling_dqn=self.enable_dueling_dqn
        ).to(device)
        
        # Sincronizar las redes al inicio
        self.target_dqn.load_state_dict(self.policy_dqn.state_dict())
        self.target_dqn.eval() # Poner la red target en modo evaluación

        # Optimizador y función de pérdida
        self.optimizer = torch.optim.Adam(
            self.policy_dqn.parameters(),
            lr=self.learning_rate_a
        )
        self.loss_fn = nn.MSELoss() # O nn.HuberLoss() para mayor estabilidad

        # Inicializar epsilon
        self.epsilon = self.epsilon_init

        # Para gráficos y guardar el modelo
        self.run_dir = os.path.join(
            RUNS_DIR,
            f"run_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{self.hyperparameter_set}"
        )
        os.makedirs(self.run_dir, exist_ok=True)
        self.scores = []
        self.avg_scores = []
        self.epsilons = []

    def choose_action(self, state):
        # Epsilon-greedy para la exploración
        if random.random() < self.epsilon:
            return random.randrange(self.num_actions)
        else:
            # Seleccionar la acción con el valor Q más alto
            with torch.no_grad():
                # Convertir el estado de numpy a tensor de PyTorch
                state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(device)
                q_values = self.policy_dqn(state_tensor)
                return torch.argmax(q_values).item()

    def train(self, num_episodes):
        start_time = datetime.now()
        for episode in range(1, num_episodes + 1):
            # Reiniciar el entorno y obtener el estado inicial
            # El método reset de Gymnasium devuelve observación e info
            observation, info = self.env.reset()
            state = observation # El estado inicial es la observación

            terminated = False # Indicador de finalización del episodio
            truncated = False  # Indicador de truncamiento del episodio (por ejemplo, límite de tiempo)
            total_reward = 0
            score = 0 # El score real del juego, se obtendrá de info['score']

            # Bucle del episodio
            while not terminated and not truncated:
                # Elegir una acción
                action = self.choose_action(state)

                # Ejecutar la acción en el entorno
                # El método step de Gymnasium devuelve obs, reward, terminated, truncated, info
                next_observation, reward, terminated, truncated, info = self.env.step(action)
                next_state = next_observation
                
                # Obtener el score real del juego desde la información del entorno
                score = info['score']

                # Almacenar la experiencia en el buffer de repetición
                self.memory.append((state, action, reward, next_state, terminated))

                # Actualizar el estado actual
                state = next_state
                total_reward += reward

                # Entrenar la red DQN si hay suficientes experiencias en el buffer
                if len(self.memory) > self.mini_batch_size:
                    self.replay()
            
            # Decaer epsilon
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
            
            # Sincronizar la red target cada cierto número de episodios
            if episode % self.network_sync_rate == 0:
                self.target_dqn.load_state_dict(self.policy_dqn.state_dict())
                self.target_dqn.eval()
                print(f"Episodio {episode}: Red target sincronizada.")

            # Guardar resultados para gráficos
            self.scores.append(score) # Guardar el score real del juego
            self.epsilons.append(self.epsilon)
            
            # Calcular la media de los últimos 100 episodios
            self.avg_scores.append(np.mean(self.scores[-100:]))

            # Imprimir progreso
            print(f"Episodio {episode}/{num_episodes} | Score: {score} | Recompensa Total: {total_reward:.2f} | Epsilon: {self.epsilon:.4f} | Memoria: {len(self.memory)} | Avg Score (100 episodios): {self.avg_scores[-1]:.2f}")

            # Guardar el modelo cada 50 episodios
            if episode % 50 == 0:
                self.save_model(os.path.join(self.run_dir, f"dqn_model_episode_{episode}.pth"))
                self.save_plots(episode)
            
            # Condición de parada si se alcanza una recompensa alta
            if self.avg_scores[-1] >= self.stop_on_reward and self.stop_on_reward != 100000:
                print(f"¡Recompensa media objetivo alcanzada! Entrenamiento detenido en el episodio {episode}.")
                break
        
        end_time = datetime.now()
        total_time = end_time - start_time
        print(f"Entrenamiento completado en {total_time}.")
        self.save_model(os.path.join(self.run_dir, "dqn_model_final.pth"))
        self.save_plots(num_episodes)


    def replay(self):
        # Si la memoria es menor que el tamaño del mini-batch, no hacemos replay
        if len(self.memory) < self.mini_batch_size:
            return

        # Muestrear un mini-batch de la memoria de repetición
        mini_batch = self.memory.sample(self.mini_batch_size)
        
        # Desempaquetar el mini-batch
        states, actions, rewards, next_states, dones = zip(*mini_batch)

        # Convertir a tensores de PyTorch
        states = torch.tensor(np.array(states), dtype=torch.float32).to(device)
        actions = torch.tensor(np.array(actions), dtype=torch.int64).to(device)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float32).to(device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32).to(device)
        dones = torch.tensor(np.array(dones), dtype=torch.bool).to(device)

        # Calcular Q-valores para los estados actuales usando la red de política
        current_q_values = self.policy_dqn(states).gather(1, actions.unsqueeze(1)).squeeze()

        # Calcular los Q-valores objetivo
        with torch.no_grad():
            if self.enable_double_dqn:
                # Double DQN: Usar la política para elegir la mejor acción futura
                # y la target network para evaluar esa acción.
                next_action_from_policy = self.policy_dqn(next_states).argmax(dim=1).unsqueeze(1)
                next_q_values = self.target_dqn(next_states).gather(1, next_action_from_policy).squeeze()
            else:
                # DQN estándar: Usar el máximo Q-valor de la target network
                next_q_values = self.target_dqn(next_states).max(1)[0] # [0] para obtener los valores máximos

            # Calcular los Q-valores esperados (target)
            # Si el episodio terminó (done es True), el Q-valor objetivo es solo la recompensa
            # De lo contrario, es recompensa + gamma * Q_siguiente_máximo
            target_q_values = rewards + (self.discount_factor_g * next_q_values * (~dones))

        # Calcular la pérdida y realizar la retropropagación
        loss = self.loss_fn(current_q_values, target_q_values)
        
        self.optimizer.zero_grad() # Limpiar gradientes
        loss.backward()             # Calcular gradientes
        # Opcional: Recortar gradientes para evitar el problema de "exploding gradients"
        # torch.nn.utils.clip_grad_norm_(self.policy_dqn.parameters(), max_norm=1.0)
        self.optimizer.step()       # Actualizar los parámetros de la red

    def save_model(self, path):
        torch.save(self.policy_dqn.state_dict(), path)
        print(f"Modelo guardado en {path}")

    def load_model(self, path):
        self.policy_dqn.load_state_dict(torch.load(path, map_location=device))
        self.target_dqn.load_state_dict(torch.load(path, map_location=device))
        self.policy_dqn.eval()
        self.target_dqn.eval()
        print(f"Modelo cargado desde {path}")

    def save_plots(self, episode):
        # Plotting scores
        plt.figure(figsize=(12, 6))
        plt.plot(self.scores, label='Score por Episodio')
        plt.plot(self.avg_scores, label='Media de Scores (últimos 100 episodios)', color='red')
        plt.xlabel('Episodio')
        plt.ylabel('Score')
        plt.title(f'Rendimiento del Agente DQN - {self.hyperparameter_set} (Episodio {episode})')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.run_dir, f'scores_episode_{episode}.png'))
        plt.close()

        # Plotting epsilon decay
        plt.figure(figsize=(12, 6))
        plt.plot(self.epsilons, label='Valor de Epsilon', color='green')
        plt.xlabel('Episodio')
        plt.ylabel('Epsilon')
        plt.title(f'Decaimiento de Epsilon - {self.hyperparameter_set} (Episodio {episode})')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.run_dir, f'epsilon_decay_episode_{episode}.png'))
        plt.close()


if __name__ == '__main__':
    # Parsear entradas de línea de comandos
    parser = argparse.ArgumentParser(description='Entrenar o probar el modelo.')
    parser.add_argument('hyperparameters', help='Nombre del conjunto de hiperparámetros del archivo hyperparameters.yml')
    parser.add_argument('--train', help='Modo de entrenamiento', action='store_true')
    parser.add_argument('--test', help='Modo de prueba (requiere --model_path)', action='store_true')
    parser.add_argument('--model_path', help='Ruta al archivo del modelo .pth para cargar')
    parser.add_argument('--episodes', type=int, default=5000, help='Número de episodios para entrenar')
    parser.add_argument('--render', help='Renderizar el entorno durante la ejecución', action='store_true') # Nuevo argumento para renderizado
    args = parser.parse_args()

    agent = Agent(hyperparameter_set=args.hyperparameters)

    # Configurar el render_mode del agente
    if args.render:
        agent.render_mode = "human"
    else:
        agent.render_mode = None # Desactivar renderizado por defecto

    # Re-crear el entorno con el render_mode correcto
    # Esto es importante porque el render_mode se usa en la inicialización del entorno
    agent.env.close() # Cerrar el entorno anterior si existe
    agent.env = FlappyBirdEnv(render_mode=agent.render_mode, **agent.env_make_params)

    if args.train:
        print(f"Iniciando entrenamiento con hiperparámetros: {args.hyperparameters}")
        agent.train(num_episodes=args.episodes)
    elif args.test:
        if not args.model_path:
            print("ERROR: El modo de prueba requiere --model_path para cargar un modelo.")
        else:
            print(f"Cargando modelo desde: {args.model_path} para prueba.")
            agent.load_model(args.model_path)
            print(f"Iniciando prueba con hiperparámetros: {args.hyperparameters}")
            # Para el modo de prueba, puedes ejecutar algunos episodios y renderizarlos
            test_episodes = 5 # Por ejemplo, 5 episodios de prueba
            for i in range(test_episodes):
                observation, info = agent.env.reset()
                state = observation
                terminated = False
                truncated = False
                total_reward = 0
                while not terminated and not truncated:
                    action = agent.choose_action(state) # Usar el modelo cargado
                    next_observation, reward, terminated, truncated, info = agent.env.step(action)
                    state = next_observation
                    total_reward += reward
                    if agent.render_mode:
                        agent.env.render() # Renderizar cada paso
                print(f"Prueba - Episodio {i+1} | Score: {info['score']} | Recompensa Total: {total_reward:.2f}")
    else:
        parser.print_help()