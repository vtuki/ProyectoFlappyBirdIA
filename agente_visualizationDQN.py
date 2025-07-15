import pygame
from entornoFlappy import FlappyEnv
from agenteDQN import DQNAgent
from flappy import SCREEN_WIDHT, SCREEN_HEIGHT
import numpy as np # Import numpy

pygame.init()

screen = pygame.display.set_mode((SCREEN_WIDHT, SCREEN_HEIGHT))
pygame.display.set_caption('Flappy Bird DQN - Visualización')
BACKGROUND = pygame.image.load('assets/sprites/background-day.png')
BACKGROUND = pygame.transform.scale(BACKGROUND, (SCREEN_WIDHT, SCREEN_HEIGHT))

VIEWING_MODE = True # Set to True to only visualize trained agent

clock = pygame.time.Clock()
env = FlappyEnv()
# Initialize DQN agent with state_size=2 and action_size=2
agent = DQNAgent(state_size=2, action_size=2)

pipes_group = pygame.sprite.Group()
bird_group = pygame.sprite.Group()

episodes = 100 # Reduced for visualization

path = "models/model_dqn_pipes_13.h5" # Path to the trained model
if VIEWING_MODE:
    try:
        agent.load(path)
        print(f"Loaded trained model from {path}.")
        agent.epsilon = 0.0 # Set epsilon to 0 for pure exploitation during viewing
    except Exception as e:
        print(f"Error loading model: {e}. Ensure 'models/model_dqn_best.h5' exists after training.")
        print("Continuing in training mode or with a new agent if no model is found.")
        VIEWING_MODE = False # Revert to training if load fails
        agent.epsilon = 1.0 # Reset epsilon for training if no model

for episode in range(episodes):
    state = env.reset()
    state = np.array(state) # Ensure state is a numpy array
    
    total_reward = 0
    tuberias_superadas = 0
    done = False
    
    while not done:
        clock.tick(120)  # Adjust speed for observation

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

        # Choose action with DQN agent (exploitation in viewing mode)
        action = agent.choose_action(state)
        
        next_state, reward, done, pasoTuberia = env.step(action)
        next_state = np.array(next_state) # Ensure next_state is a numpy array

        if pasoTuberia:
            tuberias_superadas += 1

        # In viewing mode, we don't learn
        if not VIEWING_MODE:
            agent.remember(state, action, reward, next_state, done)
            agent.learn() # Learn if not in viewing mode
        
        state = next_state
        total_reward += reward
        
        # Visualization
        screen.blit(BACKGROUND, (0, 0))
        env.render(screen, BACKGROUND, pipes_group, bird_group)
    
    # In viewing mode, epsilon should remain 0.0, no decay needed
    # If not VIEWING_MODE, you would decay epsilon here
    
    print(f"Episode {episode + 1}: Score = {total_reward}, Tuberias superadas: {tuberias_superadas}")

# Save the model if not in viewing mode (e.g., if training completed)
if not VIEWING_MODE:
    # You would typically save the model after a full training run
    # This part might be better handled in the 'entrenamientoAgenteDQN.py' script
    print("Training mode completed. Model not saved here during visualization script.")