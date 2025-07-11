import gymnasium as gym # Se usa para el tipo de entorno, aunque la clase se importa directamente
import numpy as np

import matplotlib
import matplotlib.pyplot as plt

import random
import torch
from torch import nn # Se importa para nn.HuberLoss
import yaml # Para leer el archivo de hiperparámetros

# Importar las clases auxiliares y el entorno
from experience_replay import ReplayMemory
from dqn import DQN

# Importar el entorno de Flappy Bird de Gymnasium desde su subcarpeta
from flappy_gym_env.flappy_bird_env import FlappyBirdEnv

from datetime import datetime
import argparse
import os

# Configuración de matplotlib para guardar gráficos sin mostrar ventana
matplotlib.use('Agg')

# Determinar el dispositivo de cómputo (CUDA/GPU si está disponible, de lo contrario CPU)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# Puedes forzar la CPU si experimentas problemas o si tu GPU no es significativamente más rápida para este caso
# device = 'cpu'

# Directorio para guardar los resultados de las ejecuciones (modelos, gráficos)
RUNS_DIR = "runs"
os.makedirs(RUNS_DIR, exist_ok=True) # Crea el directorio si no existe

class Agent():
    """
    Agente de Deep Q-Learning (DQL) para Flappy Bird.
    Utiliza redes neuronales (DQN), replay de experiencias y epsilon-greedy
    para aprender a jugar el juego. Soporta Double DQN y Dueling DQN.
    """

    def __init__(self, hyperparameter_set):
        """
        Inicializa el agente con un conjunto de hiperparámetros.

        Args:
            hyperparameter_set (str): El nombre del conjunto de hiperparámetros
                                      a cargar desde 'hyperparameters.yml'.
        """
        # Cargar hiperparámetros desde el archivo YAML
        with open('hyperparameters.yml', 'r') as file:
            all_hyperparameter_sets = yaml.safe_load(file)
            hyperparameters = all_hyperparameter_sets[hyperparameter_set]

        self.hyperparameter_set = hyperparameter_set

        # Asignar hiperparámetros del archivo
        self.learning_rate_a    = hyperparameters['learning_rate_a']        # Tasa de aprendizaje
        self.discount_factor_g  = hyperparameters['discount_factor_g']      # Factor de descuento (gamma)
        self.network_sync_rate  = hyperparameters['network_sync_rate']      # Frecuencia de sincronización de la red target
        self.replay_memory_size = hyperparameters['replay_memory_size']     # Tamaño del buffer de memoria
        self.mini_batch_size    = hyperparameters['mini_batch_size']        # Tamaño del mini-batch para entrenamiento
        self.epsilon_init       = hyperparameters['epsilon_init']           # Epsilon inicial
        self.epsilon_decay      = hyperparameters['epsilon_decay']          # Tasa de decaimiento de epsilon
        self.epsilon_min        = hyperparameters['epsilon_min']            # Epsilon mínimo
        self.stop_on_reward     = hyperparameters['stop_on_reward']         # Score promedio para detener el entrenamiento
        self.fc1_nodes          = hyperparameters['fc1_nodes']              # Nodos en la primera capa oculta
        self.env_make_params    = hyperparameters.get('env_make_params', {}) # Parámetros para el entorno (e.g., use_lidar)

        self.enable_double_dqn  = hyperparameters.get('enable_double_dqn', False)
        self.enable_dueling_dqn = hyperparameters.get('enable_dueling_dqn', False)

        # Establecer el modo de renderizado (se sobrescribe en __main__ si se usa --render)
        self.render_mode = None 

        # Crear el entorno Flappy Bird (Gymnasium)
        # Se inicializa con el render_mode y otros parámetros especificados en hyperparameters.yml
        self.env = FlappyBirdEnv(render_mode=self.render_mode, **self.env_make_params)

        # Obtener el tamaño del espacio de estado y acción del entorno Gymnasium
        # self.env.observation_space.shape[0] es el número de características de observación (e.g., 12 sin LIDAR)
        self.num_states = self.env.observation_space.shape[0]
        # self.env.action_space.n es el número de acciones discretas (e.g., 2 para Flappy Bird: IDLE, FLAP)
        self.num_actions = self.env.action_space.n

        # Inicializar el buffer de repetición de experiencias
        self.memory = ReplayMemory(self.replay_memory_size)

        # Inicializar las dos redes DQN:
        # 1. policy_dqn: La red que el agente usa para tomar decisiones y que se entrena.
        # 2. target_dqn: La red objetivo, cuyos pesos se actualizan periódicamente desde policy_dqn.
        #                Proporciona objetivos Q más estables para el entrenamiento.
        self.policy_dqn = DQN(
            self.num_states,
            self.num_actions,
            hidden_dim=self.fc1_nodes,
            enable_dueling_dqn=self.enable_dueling_dqn
        ).to(device) # Mover la red al dispositivo (GPU/CPU)
        
        self.target_dqn = DQN(
            self.num_states,
            self.num_actions,
            hidden_dim=self.fc1_nodes,
            enable_dueling_dqn=self.enable_dueling_dqn
        ).to(device)
        
        # Sincronizar las redes al inicio: copiar los pesos de policy_dqn a target_dqn
        self.target_dqn.load_state_dict(self.policy_dqn.state_dict())
        self.target_dqn.eval() # Poner la red target en modo evaluación (desactiva dropout/batchnorm si los hubiera)

        # Optimizador: Adam es una buena opción para DQN
        self.optimizer = torch.optim.Adam(
            self.policy_dqn.parameters(), # Solo optimizar los parámetros de la red de política
            lr=self.learning_rate_a       # Usar la tasa de aprendizaje definida
        )
        # Función de pérdida: HuberLoss es más robusta que MSE para grandes errores iniciales
        self.loss_fn = nn.HuberLoss() # <--- AJUSTADO: Usar HuberLoss para mayor estabilidad

        # Inicializar el valor de epsilon para la estrategia epsilon-greedy
        self.epsilon = self.epsilon_init

        # Configuración para guardar resultados y gráficos
        self.run_dir = os.path.join(
            RUNS_DIR,
            f"run_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{self.hyperparameter_set}"
        )
        os.makedirs(self.run_dir, exist_ok=True) # Crea el directorio si no existe

        # Listas para almacenar métricas y visualizarlas más tarde
        self.scores = []       # Scores (puntuaciones) por episodio
        self.avg_scores = []   # Promedio de scores (ej. últimos 100 episodios)
        self.epsilons = []     # Valores de epsilon a lo largo de los episodios

    def choose_action(self, state):
        """
        Selecciona una acción utilizando la estrategia epsilon-greedy.
        Elige una acción aleatoria con probabilidad epsilon, o la acción
        con el mayor Q-valor según la red de política.

        Args:
            state (np.array): El estado actual del entorno.

        Returns:
            int: La acción elegida (0 para IDLE, 1 para FLAP).
        """
        # Epsilon-greedy para la exploración
        if random.random() < self.epsilon:
            return random.randrange(self.num_actions) # Acción aleatoria
        else:
            # Seleccionar la acción con el valor Q más alto (explotación)
            with torch.no_grad(): # Desactiva el cálculo de gradientes para la inferencia
                # Convertir el estado de numpy a tensor de PyTorch y añadir una dimensión de batch
                state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(device)
                q_values = self.policy_dqn(state_tensor) # Obtener los Q-valores para el estado
                return torch.argmax(q_values).item()     # Devolver el índice de la acción con el Q-valor más alto

    def train(self, num_episodes):
        """
        Ejecuta el proceso de entrenamiento del agente.

        Args:
            num_episodes (int): El número total de episodios para entrenar.
        """
        start_time = datetime.now()
        print(f"Iniciando entrenamiento... ({start_time.strftime('%Y-%m-%d %H:%M:%S')})")

        for episode in range(1, num_episodes + 1):
            # Reiniciar el entorno y obtener el estado inicial
            # .reset() en Gymnasium devuelve (observación, información)
            observation, info = self.env.reset()
            state = observation # El estado inicial para el agente es la observación

            terminated = False # True si el episodio termina (e.g., colisión, victoria)
            truncated = False  # True si el episodio es truncado (e.g., límite de tiempo, pasos)
            total_reward = 0   # Recompensa acumulada en el episodio
            score = 0          # Score real del juego (número de tuberías pasadas)

            # Bucle principal de un episodio
            while not terminated and not truncated:
                # Elegir una acción usando la estrategia del agente
                action = self.choose_action(state)

                # Ejecutar la acción en el entorno
                # .step() en Gymnasium devuelve (next_observation, reward, terminated, truncated, info)
                next_observation, reward, terminated, truncated, info = self.env.step(action)
                next_state = next_observation
                
                # El score real del juego se obtiene del diccionario 'info'
                score = info['score']

                # Almacenar la transición (experiencia) en el buffer de repetición
                # Se almacena 'terminated' directamente como 'done'
                self.memory.append((state, action, reward, next_state, terminated))

                # Actualizar el estado actual al siguiente estado
                state = next_state
                total_reward += reward

                # Realizar un paso de entrenamiento de la red DQN si hay suficientes experiencias en el buffer
                if len(self.memory) > self.mini_batch_size:
                    self.replay()
                
                # Renderizar el entorno si el modo de renderizado está activo
                if self.render_mode:
                    self.env.render()
            
            # Decaer epsilon al final de cada episodio
            if self.epsilon > self.epsilon_min:
                self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            
            # Sincronizar la red target con los pesos de la red de política periódicamente
            if episode % self.network_sync_rate == 0:
                self.target_dqn.load_state_dict(self.policy_dqn.state_dict())
                self.target_dqn.eval() # Asegurarse de que la red target esté en modo evaluación
                print(f"Episodio {episode}: Red target sincronizada.")

            # Guardar métricas para los gráficos
            self.scores.append(score)
            self.epsilons.append(self.epsilon)
            
            # Calcular la media de los scores de los últimos 100 episodios
            self.avg_scores.append(np.mean(self.scores[-100:]))

            # Imprimir el progreso en la consola
            print(f"Episodio {episode}/{num_episodes} | Score: {score} | Recompensa Total: {total_reward:.2f} | Epsilon: {self.epsilon:.4f} | Memoria: {len(self.memory)} | Avg Score (100 episodios): {self.avg_scores[-1]:.2f}")

            # Guardar el modelo y los gráficos cada 50 episodios
            if episode % 50 == 0:
                self.save_model(os.path.join(self.run_dir, f"dqn_model_episode_{episode}.pth"))
                self.save_plots(episode)
            
            # Condición de parada si el score promedio alcanza un umbral definido
            if self.avg_scores[-1] >= self.stop_on_reward and self.stop_on_reward != 100000:
                print(f"¡Recompensa media objetivo alcanzada! Entrenamiento detenido en el episodio {episode}.")
                break
        
        end_time = datetime.now()
        total_time = end_time - start_time
        print(f"Entrenamiento completado en {total_time}.")
        # Guardar el modelo final y los gráficos al finalizar el entrenamiento
        self.save_model(os.path.join(self.run_dir, "dqn_model_final.pth"))
        self.save_plots(num_episodes)


    def replay(self):
        """
        Realiza el paso de entrenamiento de la red DQN utilizando un mini-batch
        muestreado de la memoria de repetición.
        """
        # No hay entrenamiento si la memoria no tiene suficientes elementos para un mini-batch
        if len(self.memory) < self.mini_batch_size:
            return

        # Muestrear un mini-batch de transiciones de la memoria
        mini_batch = self.memory.sample(self.mini_batch_size)
        
        # Desempaquetar el mini-batch en tensores separados para estados, acciones, etc.
        states, actions, rewards, next_states, dones = zip(*mini_batch)

        # Convertir las listas de numpy arrays a tensores de PyTorch
        states = torch.tensor(np.array(states), dtype=torch.float32).to(device)
        actions = torch.tensor(np.array(actions), dtype=torch.int64).to(device) # Las acciones son índices
        rewards = torch.tensor(np.array(rewards), dtype=torch.float32).to(device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32).to(device)
        dones = torch.tensor(np.array(dones), dtype=torch.bool).to(device) # Las banderas 'done' son booleanas

        # Calcular los Q-valores de las acciones tomadas en los estados actuales
        # Esto se hace con la red de política (policy_dqn)
        # .gather(1, actions.unsqueeze(1)) selecciona el Q-valor de la acción específica tomada para cada estado
        current_q_values = self.policy_dqn(states).gather(1, actions.unsqueeze(1)).squeeze()

        # Calcular los Q-valores objetivo para el siguiente estado
        # No necesitamos gradientes para este cálculo, ya que es el objetivo
        with torch.no_grad():
            if self.enable_double_dqn:
                # Double DQN:
                # 1. Usar la red de política para seleccionar la mejor acción futura (argmax)
                next_action_from_policy = self.policy_dqn(next_states).argmax(dim=1).unsqueeze(1)
                # 2. Usar la red target para evaluar el Q-valor de esa acción seleccionada
                next_q_values = self.target_dqn(next_states).gather(1, next_action_from_policy).squeeze()
            else:
                # DQN Estándar:
                # Usar la red target para obtener el máximo Q-valor del siguiente estado
                next_q_values = self.target_dqn(next_states).max(1)[0] # [0] para obtener solo los valores

            # Calcular los Q-valores esperados (target) para las acciones tomadas
            # target_Q = recompensa + gamma * max(Q(s',a')) si no terminó
            # target_Q = recompensa si terminó
            # (~dones) crea una máscara booleana: True si NO terminó, False si SÍ terminó
            target_q_values = rewards + (self.discount_factor_g * next_q_values * (~dones))

        # Calcular la pérdida entre los Q-valores actuales y los Q-valores objetivo
        loss = self.loss_fn(current_q_values, target_q_values)
        
        # Optimizar el modelo:
        self.optimizer.zero_grad()  # Pone a cero los gradientes acumulados de la iteración anterior
        loss.backward()             # Calcula los gradientes de la pérdida con respecto a los parámetros de la red
        # Opcional: Recortar gradientes para evitar el problema de "exploding gradients"
        # torch.nn.utils.clip_grad_norm_(self.policy_dqn.parameters(), max_norm=1.0)
        self.optimizer.step()       # Actualiza los pesos de la red usando el optimizador

    def save_model(self, path):
        """
        Guarda el estado del modelo (pesos) de la red de política.

        Args:
            path (str): La ruta completa del archivo donde se guardará el modelo.
        """
        torch.save(self.policy_dqn.state_dict(), path)
        print(f"Modelo guardado en {path}")

    def load_model(self, path):
        """
        Carga los pesos de un modelo guardado en las redes de política y target.

        Args:
            path (str): La ruta completa del archivo del modelo a cargar.
        """
        self.policy_dqn.load_state_dict(torch.load(path, map_location=device))
        self.target_dqn.load_state_dict(torch.load(path, map_location=device))
        self.policy_dqn.eval()  # Poner ambas redes en modo evaluación para inferencia
        self.target_dqn.eval()
        print(f"Modelo cargado desde {path}")

    def save_plots(self, episode):
        """
        Genera y guarda gráficos del rendimiento del agente (scores y decaimiento de epsilon).

        Args:
            episode (int): El número del episodio actual para incluirlo en el nombre del archivo.
        """
        # Gráfico de scores por episodio y media móvil
        plt.figure(figsize=(12, 6))
        plt.plot(self.scores, label='Score por Episodio')
        plt.plot(self.avg_scores, label='Media de Scores (últimos 100 episodios)', color='red')
        plt.xlabel('Episodio')
        plt.ylabel('Score')
        plt.title(f'Rendimiento del Agente DQN - {self.hyperparameter_set} (Episodio {episode})')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.run_dir, f'scores_episode_{episode}.png'))
        plt.close() # Cierra la figura para liberar memoria

        # Gráfico del decaimiento de epsilon
        plt.figure(figsize=(12, 6))
        plt.plot(self.epsilons, label='Valor de Epsilon', color='green')
        plt.xlabel('Episodio')
        plt.ylabel('Epsilon')
        plt.title(f'Decaimiento de Epsilon - {self.hyperparameter_set} (Episodio {episode})')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.run_dir, f'epsilon_decay_episode_{episode}.png'))
        plt.close() # Cierra la figura


# Punto de entrada principal del script
if __name__ == '__main__':
    # Configurar el parser de argumentos de línea de comandos
    parser = argparse.ArgumentParser(description='Entrenar o probar el modelo DQN para Flappy Bird.')
    parser.add_argument('hyperparameters', help='Nombre del conjunto de hiperparámetros a usar del archivo hyperparameters.yml (e.g., flappybird1)')
    parser.add_argument('--train', action='store_true', help='Activa el modo de entrenamiento del agente.')
    parser.add_argument('--test', action='store_true', help='Activa el modo de prueba del agente (requiere --model_path).')
    parser.add_argument('--model_path', type=str, help='Ruta al archivo del modelo .pth para cargar en modo de prueba o para continuar entrenamiento.')
    parser.add_argument('--episodes', type=int, default=5000, help='Número de episodios para entrenar (por defecto: 5000).')
    parser.add_argument('--render', action='store_true', help='Si se incluye, renderiza el entorno visualmente durante la ejecución.')
    args = parser.parse_args()

    # Crear una instancia del agente con el conjunto de hiperparámetros especificado
    agent = Agent(hyperparameter_set=args.hyperparameters)

    # Configurar el modo de renderizado del agente basado en los argumentos de la línea de comandos
    if args.render:
        agent.render_mode = "human" # Modo de renderizado visible
    else:
        agent.render_mode = None    # Sin renderizado (para entrenamiento más rápido)

    # CERRAR y RE-CREAR el entorno con el render_mode CORRECTO.
    # Esto es crucial porque el render_mode se configura en la inicialización del entorno
    # y no se puede cambiar después.
    agent.env.close() # Cierra cualquier instancia anterior del entorno
    agent.env = FlappyBirdEnv(render_mode=agent.render_mode, **agent.env_make_params)


    if args.train:
        print(f"--- Modo: ENTRENAMIENTO ---")
        print(f"Hiperparámetros cargados: '{args.hyperparameters}'")
        # Iniciar el entrenamiento del agente
        agent.train(num_episodes=args.episodes)
    elif args.test:
        print(f"--- Modo: PRUEBA ---")
        if not args.model_path:
            print("ERROR: El modo de prueba requiere especificar la ruta del modelo con '--model_path'.")
        else:
            print(f"Cargando modelo desde: {args.model_path}")
            # Cargar el modelo guardado para la prueba
            agent.load_model(args.model_path)
            # Desactivar la exploración para la prueba
            agent.epsilon = 0.0 
            print(f"Iniciando prueba (Epsilon fijado a 0.0 para explotar el modelo aprendido).")
            
            test_episodes = 5 # Ejecutar 5 episodios para la prueba

            for i in range(test_episodes):
                observation, info = agent.env.reset()
                state = observation
                terminated = False
                truncated = False
                total_reward = 0
                while not terminated and not truncated:
                    action = agent.choose_action(state) # Elige la mejor acción según el modelo
                    next_observation, reward, terminated, truncated, info = agent.env.step(action)
                    state = next_observation
                    total_reward += reward
                    if agent.render_mode:
                        agent.env.render() # Renderizar cada paso durante la prueba
                print(f"Prueba - Episodio {i+1} | Score: {info['score']} | Recompensa Total: {total_reward:.2f}")
    else:
        # Si no se especifica --train ni --test, mostrar la ayuda
        parser.print_help()