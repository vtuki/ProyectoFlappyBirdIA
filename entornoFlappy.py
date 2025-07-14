from flappy import Bird, get_random_pipes, is_off_screen, SCREEN_WIDHT, SCREEN_HEIGHT, GROUND_HEIGHT, PIPE_GAP, GAME_SPEED, PIPE_WIDHT
import pygame

FACTOR_DISCRETIZACION_Y = 60
FACTOR_DISCRETIZACION_X = 40

class FlappyEnv:
    def __init__(self):
        self.bird = Bird()
        self.pipes = []
        self.pipe_passed = False
        self.reset()

    def reset(self):
        self.bird = Bird()
        self.pipes = list(get_random_pipes(SCREEN_WIDHT + 100))
        self.done = False
        self.pipe_passed = False
        return self.get_state()

    def get_state(self):
        # Próximo tubo
        future_pipes = [pipe for pipe in self.pipes if pipe.rect[0] > self.bird.rect[0]]
        # Distancias discretizadas
        if not future_pipes:
            dist_x_pipe = (SCREEN_WIDHT - self.bird.rect[0])//FACTOR_DISCRETIZACION_X
            dist_y_pipe = ((SCREEN_HEIGHT-GROUND_HEIGHT)//2 - self.bird.rect[1])//FACTOR_DISCRETIZACION_Y

        else:
            pipe = future_pipes[0]
            dist_x_pipe = (pipe.rect[0] - self.bird.rect[0]) // FACTOR_DISCRETIZACION_X
            dist_y_pipe = (pipe.rect[1] - self.bird.rect[1]) // (FACTOR_DISCRETIZACION_Y)
        
        speed = self.bird.speed//3
        if (speed < -2):
            speed = -2

        return (dist_x_pipe, dist_y_pipe)

    def step(self, action):
        future_pipes = [pipe for pipe in self.pipes if pipe.rect[0] > (self.bird.rect[0]-PIPE_WIDHT-5)]

        if action == 1:
            self.bird.bump()

        self.bird.update()

        for pipe in self.pipes:
            pipe.update()

        # Si los tubos salen de pantalla
        if is_off_screen(self.pipes[0]):
            self.pipes = list(get_random_pipes(SCREEN_WIDHT + 100))
            self.pipe_passed = False

        # Colisiones
        for pipe in self.pipes:
            if pygame.sprite.collide_mask(self.bird, pipe):
                self.done = True
                reward = -200
                # # Calcula al recompensa en base a que tan lejos esta del tunel
                # if pipe.rect[1] < 0:
                #     pos_y_tunel = (pipe.rect[3]+pipe.rect[1]) + PIPE_GAP/2
                #     reward = -30 * (pos_y_tunel//FACTOR_DISCRETIZACION_Y - self.bird.rect[1]//FACTOR_DISCRETIZACION_Y)

                # else:
                #     pos_y_tunel = pipe.rect[1] - PIPE_GAP/2
                #     reward = -30 *(self.bird.rect[1]//FACTOR_DISCRETIZACION_Y - pos_y_tunel//FACTOR_DISCRETIZACION_Y)

                return self.get_state(), reward, True, False

        # Penalizacion por chocar con el piso
        if self.bird.rect[1] > SCREEN_HEIGHT - GROUND_HEIGHT:
            self.done = True
            return self.get_state(), -200, True, False
        
        # Para que no vuele por encima de las tuberias
        if self.bird.rect[1] < 0:
            self.done = True
            return self.get_state(), -200, True, False
        
        if future_pipes:
            pipe = future_pipes[0]

            # Si el pájaro está completamente más allá del pipe, lo consideramos superado
            if self.bird.rect[0] > pipe.rect[0] + PIPE_WIDHT and not self.pipe_passed:
                self.pipe_passed = True
                return self.get_state(), 1000, False, True

        # Recompensa por sobrevivir
        return self.get_state(), 1, False, False

    def render(self, screen, background, pipes_group, bird_group):
        screen.blit(background, (0, 0))

        # Dibujar los tubos
        for pipe in self.pipes:
            pipes_group.add(pipe)
        pipes_group.draw(screen)
        pipes_group.empty()  # Limpiamos el grupo después de dibujar

        # Dibujar el pájaro
        bird_group.add(self.bird)
        bird_group.draw(screen)
        bird_group.empty()  # Limpiamos el grupo después de dibujar

        pygame.display.update()

