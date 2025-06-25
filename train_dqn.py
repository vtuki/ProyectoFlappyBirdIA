from flappy_env import FlappyBirdEnv
from dqn_agent import DQNAgent
import os

# Parámetros de entrenamiento
EPISODES = 1000 # Número de episodios para entrenar
BATCH_SIZE = 32
UPDATE_TARGET_MODEL_FREQ = 10 # Actualiza la target network cada X episodios
SAVE_MODEL_FREQ = 50 # Guarda el modelo cada X episodios

# Crea la carpeta para guardar los modelos si no existe
if not os.path.exists('models'):
    os.makedirs('models')

if __name__ == "__main__":
    env = FlappyBirdEnv()
    state_size = env._get_state().shape[0]
    action_size = 2 # 0: no saltar, 1: saltar

    agent = DQNAgent(state_size, action_size)

    # Opcional: cargar un modelo pre-entrenado
    # if os.path.exists('models/flappy_bird_dqn.weights.h5'):
    #     agent.load('models/flappy_bird_dqn.weights.h5')
    #     print("Modelo cargado.")


    for e in range(EPISODES):
        state = env.reset()
        done = False
        score = 0
        while not done:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            score += reward # Aunque la recompensa es del step, el score es del juego

        print(f"Episodio: {e+1}/{EPISODES}, Score: {env.score}, Epsilon: {agent.epsilon:.2f}") # Mostrar el score del juego

        agent.replay(BATCH_SIZE)

        if (e + 1) % UPDATE_TARGET_MODEL_FREQ == 0:
            agent.update_target_model()
            print("Target model actualizado.")
        
        if (e + 1) % SAVE_MODEL_FREQ == 0:
            agent.save(f"models/flappy_bird_dqn_episode_{e+1}.weights.h5")
            print(f"Modelo guardado en models/flappy_bird_dqn_episode_{e+1}.weights.h5")

    env.close()
    print("Entrenamiento finalizado.")
    agent.save("models/flappy_bird_dqn_final.weights.h5")
    print("Modelo final guardado en models/flappy_bird_dqn_final.weights.h5")