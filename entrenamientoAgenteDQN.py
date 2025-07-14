from entornoFlappy import FlappyEnv
from agenteDQN import DQNAgent
from entornoFlappy import FACTOR_DISCRETIZACION_X, FACTOR_DISCRETIZACION_Y
import numpy as np

env = FlappyEnv()
# State size is 2 (dist_x_pipe, dist_y_pipe)
# Action size is 2 (0 for no jump, 1 for jump)
agent = DQNAgent(state_size=2, action_size=2) 

OVER_TRAIN = False

path_model = "models/model_dqn_best.h5" # Path to save/load the best model
if OVER_TRAIN:
    try:
        agent.load(path_model)
        print(f"Loaded existing model from {path_model} and epsilon.")
    except Exception as e:
        print(f"Could not load model: {e}. Starting fresh training.")

episodes = 2000 # Reduced for quicker testing, increase for better training

max_surpassed_pipes = 0
mean_score = 0
total_scores_per_episode = []

# Parameters for target model update
TARGET_UPDATE_FREQ = 10 # Update target model every N episodes

for episode in range(episodes):
    state = env.reset()
    # Ensure state is a numpy array for the neural network
    state = np.array(state) 
    
    total_reward = 0
    surpassed_pipes = 0
    done = False
    
    while not done:
        action = agent.choose_action(state)
        
        next_state, reward, done, pasoTuberia = env.step(action)
        next_state = np.array(next_state) # Ensure next_state is a numpy array

        # Store the experience in replay memory
        agent.remember(state, action, reward, next_state, done)
        
        state = next_state
        total_reward += reward

        if pasoTuberia:
            surpassed_pipes += 1
        
        # Learn from experiences in the replay buffer
        agent.learn()
    
    # Update target model periodically
    if episode % TARGET_UPDATE_FREQ == 0:
        agent.update_target_model()

    if surpassed_pipes > max_surpassed_pipes:
        max_surpassed_pipes = surpassed_pipes
        # Optionally save the model if it achieves a new high score for pipes
        agent.save(f"models/model_dqn_pipes_{surpassed_pipes}.h5")

    total_scores_per_episode.append(total_reward)
    mean_score = np.mean(total_scores_per_episode[-100:]) # Mean score of last 100 episodes

    print(f'Episode {episode + 1}: Score {total_reward}, Tuberias superadas: {surpassed_pipes}, eps: {agent.epsilon:.4f}, Mean Score (last 100): {mean_score:.2f}')

# Final save
agent.save(path_model)
print(f"Training finished. Best model saved to {path_model}.")
print(f"Máxima cantidad de tuberias superadas: {max_surpassed_pipes}")
print(f"Score medio (últimos 100 episodios): {mean_score:.2f}")