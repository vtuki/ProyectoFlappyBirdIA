import pygame
import sys
import os
import numpy as np
from dqn_agent import DQNAgent  
from flappy_env import FlappyBirdEnv 

# --- PARÁMETROS DE ENTRENAMIENTO ---
EPISODES = 500          # Número de episodios para entrenar
BATCH_SIZE = 32         # Tamaño del minibatch para el replay
UPDATE_TARGET_MODEL_FREQ = 10 # Actualiza la target network cada 10 episodios
SAVE_MODEL_FREQ = 50    # Guarda el modelo cada 50 episodios

# Crea la carpeta para guardar los modelos si no existe
if not os.path.exists('models'):
    os.makedirs('models')

if __name__ == "__main__":
    # Inicializar el entorno y el agente
    env = FlappyBirdEnv()
    state_size = env._get_state().shape[0] # Obtiene el tamaño del estado (e.g., 5)
    action_size = 2 # 0: no saltar, 1: saltar
    agent = DQNAgent(state_size, action_size)

    # Intentar cargar un modelo pre-entrenado
    model_path = 'models/flappy_bird_dqn_final.weights.h5'
    if os.path.exists(model_path):
         agent.load(model_path)
         print("Modelo cargado exitosamente.")
         # Reducir epsilon si se carga un modelo para no explorar tanto
         agent.epsilon = 0.1 # Reducir la exploración si ya hay un modelo entrenado
    else:
        print("No se encontró un modelo. Comenzando entrenamiento desde cero.")

    # Bucle principal de entrenamiento
    for e in range(EPISODES):
        state = env.reset()
        # 'state' tiene forma (state_size,), e.g., (5,)
        
        done = False
        
        # Bucle para un solo episodio
        while not done:
            # ===> MANEJO DE EVENTOS DE PYGAME (¡CRUCIAL!) <===
            # Este bucle DEBE estar aquí para que la ventana de Pygame responda.
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    env.close()
                    sys.exit()

            # Lógica del agente
            # agent.act espera un estado de forma (state_size,) y lo reformará internamente
            action = agent.act(state) 
            
            # Paso en el entorno
            next_state, reward, done, _ = env.step(action)
            # 'next_state' tiene forma (state_size,), e.g., (5,)

            # Almacenar la experiencia en el buffer
            # Pasamos 'state' y 'next_state' sin la dimensión de batch extra
            agent.remember(state, action, reward, next_state, done)
            
            # Actualizar el estado para la siguiente iteración
            state = next_state
            
        # Al final del episodio, entrenar el modelo si el buffer tiene suficientes experiencias
        if len(agent.memory) > BATCH_SIZE:
            agent.replay(BATCH_SIZE)
            
        # Actualizar la target network
        if (e + 1) % UPDATE_TARGET_MODEL_FREQ == 0:
            agent.update_target_model()
            print(f"Episodio {e+1}: Target model actualizado.")

        # Imprimir el progreso y el score del episodio
        # Usamos env.score que es el score real del juego
        print(f"Episodio {e+1}/{EPISODES} | Score: {env.score} | Exploración (Epsilon): {agent.epsilon:.2f} | Memoria: {len(agent.memory)}")
        
        # Guardar el modelo cada cierto número de episodios
        if (e + 1) % SAVE_MODEL_FREQ == 0:
            agent.save(f"models/flappy_bird_dqn_episode_{e+1}.weights.h5")
            print(f"Modelo guardado en models/flappy_bird_dqn_episode_{e+1}.weights.h5.")

    # Limpieza final
    env.close()
    print("Entrenamiento finalizado.")
    agent.save("models/flappy_bird_dqn_final.weights.h5")
    print("Modelo final guardado en models/flappy_bird_dqn_final.weights.h5.")