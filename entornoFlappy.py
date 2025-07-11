from flappy import Bird, get_random_pipes, is_off_screen, SCREEN_WIDHT, SCREEN_HEIGHT, GROUND_HEIGHT
import pygame

class FlappyEnv:
    def __init__(self):
        self.bird = Bird()
        self.pipes = []
        self.reset()

    def reset(self):
        self.bird = Bird()
        self.pipes = list(get_random_pipes(SCREEN_WIDHT + 100))
        self.done = False
        return self.get_state()

    def get_state(self):
        # Próximo tubo
        future_pipes = [pipe for pipe in self.pipes if pipe.rect[0] > self.bird.rect[0]]
        # Distancias discretizadas
        if not future_pipes:
            dist_x = (SCREEN_WIDHT - self.bird.rect[0])//10
            dist_y = 0
        else:
            pipe = future_pipes[0]
            dist_x = (pipe.rect[0] - self.bird.rect[0]) // 10
            dist_y = (pipe.rect[1] - self.bird.rect[1]) // 10
        
        vel = int(self.bird.speed)

        return (dist_x, dist_y, vel)

    def step(self, action):
        if action == 1:
            self.bird.bump()

        self.bird.update()

        for pipe in self.pipes:
            pipe.update()

        # Si los tubos salen de pantalla
        if is_off_screen(self.pipes[0]):
            self.pipes = list(get_random_pipes(SCREEN_WIDHT + 100))

        # Colisiones
        for pipe in self.pipes:
            if pygame.sprite.collide_mask(self.bird, pipe):
                self.done = True
                return self.get_state(), -100, True

        if self.bird.rect[1] > SCREEN_HEIGHT - GROUND_HEIGHT:
            self.done = True
            return self.get_state(), -100, True

        

        # Recompensa por sobrevivir
        return self.get_state(), 1, False

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

