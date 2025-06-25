import pygame, random, time
from pygame.locals import *
import numpy as np

# VARIABLES (mantén las mismas que en tu flappy.py)
SCREEN_WIDHT = 400
SCREEN_HEIGHT = 600
SPEED = 20
GRAVITY = 2.5
GAME_SPEED = 15

GROUND_WIDHT = 2 * SCREEN_WIDHT
GROUND_HEIGHT= 100

PIPE_WIDHT = 80
PIPE_HEIGHT = 500

PIPE_GAP = 150

wing = 'assets/audio/wing.wav'
hit = 'assets/audio/hit.wav'
point = 'assets/audio/point.wav' # Añadido para recompensa

pygame.mixer.init()

# Clases Bird, Pipe, Ground (Copia y pega las clases de tu flappy.py aquí)
class Bird(pygame.sprite.Sprite):
    def __init__(self):
        pygame.sprite.Sprite.__init__(self)
        self.images =  [pygame.image.load('assets/sprites/bluebird-upflap.png').convert_alpha(),
                        pygame.image.load('assets/sprites/bluebird-midflap.png').convert_alpha(),
                        pygame.image.load('assets/sprites/bluebird-downflap.png').convert_alpha()]
        self.speed = SPEED
        self.current_image = 0
        self.image = pygame.image.load('assets/sprites/bluebird-upflap.png').convert_alpha()
        self.mask = pygame.mask.from_surface(self.image)
        self.rect = self.image.get_rect()
        self.rect[0] = SCREEN_WIDHT / 6
        self.rect[1] = SCREEN_HEIGHT / 2

    def update(self):
        self.current_image = (self.current_image + 1) % 3
        self.image = self.images[self.current_image]
        self.speed += GRAVITY
        self.rect[1] += self.speed

    def bump(self):
        self.speed = -SPEED

    def begin(self):
        self.current_image = (self.current_image + 1) % 3
        self.image = self.images[self.current_image]

class Pipe(pygame.sprite.Sprite):
    def __init__(self, inverted, xpos, ysize):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.image.load('assets/sprites/pipe-green.png').convert_alpha()
        self.image = pygame.transform.scale(self.image, (PIPE_WIDHT, PIPE_HEIGHT))
        self.rect = self.image.get_rect()
        self.rect[0] = xpos
        if inverted:
            self.image = pygame.transform.flip(self.image, False, True)
            self.rect[1] = - (self.rect[3] - ysize)
        else:
            self.rect[1] = SCREEN_HEIGHT - ysize
        self.mask = pygame.mask.from_surface(self.image)

    def update(self):
        self.rect[0] -= GAME_SPEED

class Ground(pygame.sprite.Sprite):
    def __init__(self, xpos):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.image.load('assets/sprites/base.png').convert_alpha()
        self.image = pygame.transform.scale(self.image, (GROUND_WIDHT, GROUND_HEIGHT))
        self.mask = pygame.mask.from_surface(self.image)
        self.rect = self.image.get_rect()
        self.rect[0] = xpos
        self.rect[1] = SCREEN_HEIGHT - GROUND_HEIGHT

    def update(self):
        self.rect[0] -= GAME_SPEED

def is_off_screen(sprite):
    return sprite.rect[0] < -(sprite.rect[2])

def get_random_pipes(xpos):
    size = random.randint(100, 300)
    pipe = Pipe(False, xpos, size)
    pipe_inverted = Pipe(True, xpos, SCREEN_HEIGHT - size - PIPE_GAP)
    return pipe, pipe_inverted


class FlappyBirdEnv:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDHT, SCREEN_HEIGHT))
        pygame.display.set_caption('Flappy Bird DQN')

        self.BACKGROUND = pygame.image.load('assets/sprites/background-day.png')
        self.BACKGROUND = pygame.transform.scale(self.BACKGROUND, (SCREEN_WIDHT, SCREEN_HEIGHT))

        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 50)  # Para mostrar el score

        self.reset()

    def reset(self):
        self.bird_group = pygame.sprite.Group()
        self.bird = Bird()
        self.bird_group.add(self.bird)

        self.ground_group = pygame.sprite.Group()
        for i in range (2):
            ground = Ground(GROUND_WIDHT * i)
            self.ground_group.add(ground)

        self.pipe_group = pygame.sprite.Group()
        for i in range (2):
            pipes = get_random_pipes(SCREEN_WIDHT * i + 300) # Ajustado para que las tuberías no empiecen tan lejos
            self.pipe_group.add(pipes[0])
            self.pipe_group.add(pipes[1])
        
        self.score = 0
        self.game_over = False
        return self._get_state()

    def _get_state(self):
        # Define tu estado aquí. Esto es crucial para el aprendizaje.
        # Un buen estado debe contener toda la información relevante para tomar una decisión.
        # Ejemplo de estado:
        # [bird_y, bird_speed, dist_to_next_pipe_x, next_pipe_top_y, next_pipe_bottom_y]

        # Encontrar la próxima tubería
        next_pipe = None
        for pipe in self.pipe_group.sprites():
            if pipe.rect[0] + pipe.rect[2] > self.bird.rect[0]:  # Si la tubería está delante del pájaro
                if next_pipe is None or pipe.rect[0] < next_pipe.rect[0]:
                    next_pipe = pipe
        
        # Necesitamos la tubería superior e inferior para el mismo par
        next_pipe_top_y = 0
        next_pipe_bottom_y = 0
        dist_to_next_pipe_x = SCREEN_WIDHT # Valor por defecto si no hay tuberías

        if next_pipe:
            # Asumimos que la primera tubería es la de arriba o abajo
            # Para encontrar el par, podrías necesitar una lógica más robusta
            # Por simplicidad, asumamos que las tuberías se agregan en pares ordenados
            pipes_in_order = sorted(self.pipe_group.sprites(), key=lambda p: p.rect[0])
            for i, p in enumerate(pipes_in_order):
                if p is next_pipe:
                    # Encuentra el otro pipe del par (puede ser el anterior o el siguiente)
                    if i > 0 and pipes_in_order[i-1].rect[0] == p.rect[0]:
                        if pipes_in_order[i-1].rect[1] < p.rect[1]: # si el anterior es el top
                            next_pipe_top_y = pipes_in_order[i-1].rect[1] + pipes_in_order[i-1].rect[3] # bottom of top pipe
                            next_pipe_bottom_y = p.rect[1] # top of bottom pipe
                        else: # si el anterior es el bottom
                            next_pipe_top_y = p.rect[1] + p.rect[3]
                            next_pipe_bottom_y = pipes_in_order[i-1].rect[1]
                    elif i < len(pipes_in_order) - 1 and pipes_in_order[i+1].rect[0] == p.rect[0]:
                        if pipes_in_order[i+1].rect[1] < p.rect[1]: # si el siguiente es el top
                            next_pipe_top_y = pipes_in_order[i+1].rect[1] + pipes_in_order[i+1].rect[3]
                            next_pipe_bottom_y = p.rect[1]
                        else: # si el siguiente es el bottom
                            next_pipe_top_y = p.rect[1] + p.rect[3]
                            next_pipe_bottom_y = pipes_in_order[i+1].rect[1]
                    
                    dist_to_next_pipe_x = next_pipe.rect[0] - self.bird.rect[0]
                    break
        
        # Normaliza los valores para que estén en un rango similar
        state = [
            self.bird.rect[1] / SCREEN_HEIGHT, # bird_y (0 a 1)
            self.bird.speed / SPEED, # bird_speed (relativo a la velocidad máxima de salto/caída)
            dist_to_next_pipe_x / SCREEN_WIDHT, # dist_to_next_pipe_x (0 a 1)
            next_pipe_top_y / SCREEN_HEIGHT, # next_pipe_top_y (0 a 1)
            next_pipe_bottom_y / SCREEN_HEIGHT # next_pipe_bottom_y (0 a 1)
        ]
        return np.array(state, dtype=np.float32)


    def step(self, action):
        reward = 0.1 # Recompensa pequeña por sobrevivir cada frame
        done = False

        # Actualizar el juego basado en la acción
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                exit()
        
        if action == 1: # Si la acción es saltar
            self.bird.bump()
            pygame.mixer.music.load(wing)
            pygame.mixer.music.play()

        self.bird_group.update()
        self.ground_group.update()
        self.pipe_group.update()

        # Generar nuevas tuberías y suelo
        if is_off_screen(self.ground_group.sprites()[0]):
            self.ground_group.remove(self.ground_group.sprites()[0])
            new_ground = Ground(GROUND_WIDHT - 20)
            self.ground_group.add(new_ground)

        # Manejo de tuberías: si se salen de la pantalla, se quitan y se añaden nuevas
        # También se otorga recompensa si se pasa una tubería
        pipes_to_remove = []
        passed_pipe_in_this_step = False
        for pipe in self.pipe_group.sprites():
            if is_off_screen(pipe):
                pipes_to_remove.append(pipe)
            # Recompensa por pasar una tubería
            # La lógica para esto es un poco más compleja ya que las tuberías vienen en pares
            # Y no queremos recompensar dos veces por el mismo par.
            # Una forma es verificar si el centro del pájaro ha pasado el centro de la tubería inferior.
            if pipe.rect[0] + pipe.rect[2] < self.bird.rect[0] and not hasattr(pipe, 'passed'):
                # Asegurarse de que es la tubería inferior del par
                other_pipe = None
                for p_other in self.pipe_group.sprites():
                    if p_other.rect[0] == pipe.rect[0] and p_other is not pipe:
                        other_pipe = p_other
                        break
                
                if other_pipe and pipe.rect[1] > other_pipe.rect[1]: # Si 'pipe' es la tubería inferior
                    reward += 10 # Recompensa por pasar
                    pygame.mixer.music.load(point) # Sonido de punto
                    pygame.mixer.music.play()
                    pipe.passed = True # Marcar la tubería como pasada
                    other_pipe.passed = True
                    passed_pipe_in_this_step = True
                    self.score += 1 # Aumentar el score

        for pipe in pipes_to_remove:
            self.pipe_group.remove(pipe)
        
        # Si se quitaron todas las tuberías de un par, añadir nuevas
        if len(self.pipe_group) < 4: # Suponiendo que siempre hay 2 pares visibles (4 tuberías)
            pipes = get_random_pipes(SCREEN_WIDHT * 2) # Aparecen más lejos para no solaparse
            self.pipe_group.add(pipes[0])
            self.pipe_group.add(pipes[1])


        # Detección de colisiones
        if (pygame.sprite.groupcollide(self.bird_group, self.ground_group, False, False, pygame.sprite.collide_mask) or
            pygame.sprite.groupcollide(self.bird_group, self.pipe_group, False, False, pygame.sprite.collide_mask) or
            self.bird.rect[1] < 0): # También si el pájaro sale por arriba
            reward = -100 # Gran penalización por chocar
            done = True
            self.game_over = True
            pygame.mixer.music.load(hit)
            pygame.mixer.music.play()
            # time.sleep(1) # No pausar el juego durante el entrenamiento automático

        # Dibujar en pantalla (solo para visualización durante el entrenamiento)
        self.screen.blit(self.BACKGROUND, (0, 0))
        self.bird_group.draw(self.screen)
        self.pipe_group.draw(self.screen)
        self.ground_group.draw(self.screen)

        # Mostrar score
        score_text = self.font.render(str(self.score), True, (255, 255, 255))
        self.screen.blit(score_text, (SCREEN_WIDHT // 2 - score_text.get_width() // 2, 50))


        pygame.display.update()
        self.clock.tick(GAME_SPEED) # Controla la velocidad de simulación

        next_state = self._get_state()
        return next_state, reward, done, {} # El último diccionario es para información adicional

    def close(self):
        pygame.quit()