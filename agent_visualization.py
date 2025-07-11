import pygame
from entornoFlappy import FlappyEnv
from agenteQLearning import QAgent
from flappy import SCREEN_WIDHT, SCREEN_HEIGHT

pygame.init()

screen = pygame.display.set_mode((SCREEN_WIDHT, SCREEN_HEIGHT))
pygame.display.set_caption('Flappy Bird Q-learning - Entrenamiento visual')
BACKGROUND = pygame.image.load('assets/sprites/background-day.png')
BACKGROUND = pygame.transform.scale(BACKGROUND, (SCREEN_WIDHT, SCREEN_HEIGHT))

clock = pygame.time.Clock()
env = FlappyEnv()
agent = QAgent()

pipes_group = pygame.sprite.Group()
bird_group = pygame.sprite.Group()

episodes = 2000
for episode in range(episodes):
    state = env.reset()
    total_reward = 0
    done = False
    frame_count = 0

    while not done:
        clock.tick(30)  # Baja la velocidad si quieres observarlo más lentamente

        # Cerrar la ventana si el usuario la cierra
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

        # Elegir acción con Q-learning
        action = agent.choose_action(state)

        # Ejecutar acción
        next_state, reward, done = env.step(action)

        # Aprender
        agent.learn(state, action, reward, next_state)

        state = next_state
        total_reward += reward
        frame_count += 1

        # Visualización del entorno
        screen.blit(BACKGROUND, (0, 0))
        # Dibujamos bird y pipes como antes
        env.render(screen, BACKGROUND, pipes_group, bird_group)

    print(f"Episodio {episode + 1}: Recompensa total = {total_reward}, Frames: {frame_count}")
