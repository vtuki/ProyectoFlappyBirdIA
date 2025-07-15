import pygame
from entornoFlappy import FlappyEnv
from agenteQLearning import QAgent
from flappy import SCREEN_WIDHT, SCREEN_HEIGHT

pygame.init()

screen = pygame.display.set_mode((SCREEN_WIDHT, SCREEN_HEIGHT))
pygame.display.set_caption('Flappy Bird Q-learning - Entrenamiento visual')
BACKGROUND = pygame.image.load('assets/sprites/background-day.png')
BACKGROUND = pygame.transform.scale(BACKGROUND, (SCREEN_WIDHT, SCREEN_HEIGHT))

VIEWING_MODE = True

clock = pygame.time.Clock()
env = FlappyEnv()
agent = QAgent()

pipes_group = pygame.sprite.Group()
bird_group = pygame.sprite.Group()

episodes = 100000

path = "models/qtable123Pipes-40ScoreBEST.pkl"
if VIEWING_MODE:
    agent.load(path)

for episode in range(episodes):
    state = env.reset()
    state_changed = True
    total_reward = 0
    done = False
    tuberias_superadas = 0

    while not done:
        clock.tick(120)  # Baja la velocidad si quieres observarlo más lentamente

        # Cerrar la ventana si el usuario la cierra
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

        # Elegir acción con Q-learning
        if (not VIEWING_MODE):
            action = agent.choose_action(state)
        else:
            action = agent.choose_action(state, True)

        action_done = action
        if (not state_changed):
            action_done = 0
        else:
            state_changed = False 
        # Ejecutar acción
        next_state, reward, done, pasoTuberia = env.step(action_done)

        if (state != next_state or done):
            state_changed = True
            if pasoTuberia:
                tuberias_superadas+=1

            # Aprender
            if not VIEWING_MODE:
                agent.learn(state, action, reward, next_state)

            state = next_state
            total_reward += reward
        
        if (not state_changed and pasoTuberia):
            tuberias_superadas+=1
            total_reward += reward
                
            if not VIEWING_MODE:
                agent.learn(state, action, reward, next_state)

        # Visualización del entorno
        screen.blit(BACKGROUND, (0, 0))
        # Dibujamos bird y pipes como antes
        env.render(screen, BACKGROUND, pipes_group, bird_group)
    
    agent.decay_epsilon()
    if (not VIEWING_MODE):
        print(f"Episode {episode + 1}: Score = {total_reward}, Tuberias superadas: {tuberias_superadas}, eps: {agent.epsilon}")
    else:
        print(f"Episode {episode + 1}: Score = {total_reward}, Tuberias superadas: {tuberias_superadas}")

if not VIEWING_MODE:
    agent.save()
