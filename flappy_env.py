import pygame, random, time
from pygame.locals import *
import numpy as np
from sys import exit

# -- VARIABLES GLOBALES (Asegúrate de que sean las mismas que en flappy.py) --
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

# -- RUTAS DE AUDIOS --
# Rutas de los archivos de audio
wing_path = 'assets/audio/wing.wav'
hit_path = 'assets/audio/hit.wav'
point_path = 'assets/audio/point.wav'

pygame.mixer.init()

# -- SPRITES (Copiados directamente de flappy.py) --
class Bird(pygame.sprite.Sprite):
    def __init__(self):
        pygame.sprite.Sprite.__init__(self)
        self.images =  [
            pygame.image.load('assets/sprites/bluebird-upflap.png').convert_alpha(),
            pygame.image.load('assets/sprites/bluebird-midflap.png').convert_alpha(),
            pygame.image.load('assets/sprites/bluebird-downflap.png').convert_alpha()
        ]
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
        self.rect[1] += self.speed # El índice [1] es la coordenada Y

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
        # No se necesita el atributo 'passed' si la recompensa es solo por supervivencia
        # Si quieres recompensa por pasar tuberías, deberás añadirlo aquí y en step()
        # self.passed = False # Si quieres mantener la recompensa por pasar tuberías

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

# -- FUNCIONES AUXILIARES (Copiadas de flappy.py) --
def is_off_screen(sprite):
    return sprite.rect[0] < -(sprite.rect[2])

def get_random_pipes(xpos):
    size = random.randint(100, 300)
    pipe = Pipe(False, xpos, size)
    pipe_inverted = Pipe(True, xpos, SCREEN_HEIGHT - size - PIPE_GAP)
    return pipe, pipe_inverted

# -- ENTORNO FLAPPY BIRD PARA EL AGENTE --
class FlappyBirdEnv:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDHT, SCREEN_HEIGHT))
        pygame.display.set_caption('Flappy Bird DQN')

        self.BACKGROUND = pygame.image.load('assets/sprites/background-day.png')
        self.BACKGROUND = pygame.transform.scale(self.BACKGROUND, (SCREEN_WIDHT, SCREEN_HEIGHT))

        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 50)

        # Cargar sonidos una sola vez al inicio para optimizar
        self.wing_sound = pygame.mixer.Sound(wing_path)
        self.hit_sound = pygame.mixer.Sound(hit_path)
        self.point_sound = pygame.mixer.Sound(point_path)

        # Atributos de la clase para las constantes
        self.SCREEN_WIDHT = SCREEN_WIDHT
        self.SCREEN_HEIGHT = SCREEN_HEIGHT
        self.SPEED = SPEED
        self.PIPE_GAP = PIPE_GAP
        self.GROUND_WIDHT = GROUND_WIDHT
        self.GAME_SPEED = GAME_SPEED

        self.reset()

    def reset(self):
        self.bird_group = pygame.sprite.Group()
        self.bird = Bird()
        self.bird_group.add(self.bird)

        self.ground_group = pygame.sprite.Group()
        for i in range (2):
            ground = Ground(self.GROUND_WIDHT * i)
            self.ground_group.add(ground)

        self.pipe_group = pygame.sprite.Group()
        # Generar tuberías para que el agente tenga objetivos
        pipes1 = get_random_pipes(SCREEN_WIDHT + 200)
        self.pipe_group.add(pipes1[0])
        self.pipe_group.add(pipes1[1])

        pipes2 = get_random_pipes(SCREEN_WIDHT + 200 + SCREEN_WIDHT // 2)
        self.pipe_group.add(pipes2[0])
        self.pipe_group.add(pipes2[1])

        self.score = 0
        self.game_over = False
        return self._get_state()

    def _get_state(self):
        """
        Calcula el estado del juego de forma más robusta y simple,
        usando posiciones relativas y coordenadas absolutas normalizadas para las tuberías.
        """

        pipes_in_front = [p for p in self.pipe_group.sprites() if p.rect.right > self.bird.rect.left]
        pipes_in_front.sort(key=lambda p: p.rect.x)

        # Valores por defecto si no hay tuberías visibles
        dist_to_next_pipe_x = self.SCREEN_WIDHT # Distancia horizontal a la pared derecha si no hay tuberías
        next_pipe_top_y = 0                     # Borde superior de la pantalla
        next_pipe_bottom_y = self.SCREEN_HEIGHT # Borde inferior de la pantalla

        if pipes_in_front:
            first_pipe_x = pipes_in_front[0].rect.x
            current_pipe_pair = [p for p in pipes_in_front if p.rect.x == first_pipe_x]

            if len(current_pipe_pair) == 2:
                # Identifica cuál es la superior y cuál es la inferior
                if current_pipe_pair[0].rect.y < current_pipe_pair[1].rect.y:
                    top_pipe = current_pipe_pair[0]
                    bottom_pipe = current_pipe_pair[1]
                else:
                    top_pipe = current_pipe_pair[1]
                    bottom_pipe = current_pipe_pair[0]

                dist_to_next_pipe_x = top_pipe.rect.x - self.bird.rect.x
                next_pipe_top_y = top_pipe.rect.y + top_pipe.rect.height
                next_pipe_bottom_y = bottom_pipe.rect.y
            else: 
                # Este caso debería ser raro si las tuberías se generan en pares.
                # Asume que el hueco está a la altura media si solo se ve una tubería.
                dist_to_next_pipe_x = pipes_in_front[0].rect.x - self.bird.rect.x
                if pipes_in_front[0].rect.y < self.SCREEN_HEIGHT / 2: # Probablemente una tubería superior
                    next_pipe_top_y = pipes_in_front[0].rect.y + pipes_in_front[0].rect.height
                    next_pipe_bottom_y = next_pipe_top_y + self.PIPE_GAP
                else: # Probablemente una tubería inferior
                    next_pipe_bottom_y = pipes_in_front[0].rect.y
                    next_pipe_top_y = next_pipe_bottom_y - self.PIPE_GAP

        # Normalizamos los valores para que estén en un rango similar
        state = np.array([
            self.bird.rect.y / self.SCREEN_HEIGHT,                     # Posición Y del pájaro (0 a 1)
            self.bird.speed / self.SPEED,                              # Velocidad vertical del pájaro (normalizada por la velocidad de salto)
            dist_to_next_pipe_x / self.SCREEN_WIDHT,                  # Distancia horizontal a la siguiente tubería (0 a 1)
            next_pipe_top_y / self.SCREEN_HEIGHT,                     # Coordenada Y de la parte superior del hueco (0 a 1)
            next_pipe_bottom_y / self.SCREEN_HEIGHT                   # Coordenada Y de la parte inferior del hueco (0 a 1)
        ], dtype=np.float32)

        return state

    def step(self, action):
        reward = 1.0 # Recompensa pequeña por sobrevivir cada frame
        done = False

        if action == 1: # Si la acción es saltar
            self.bird.bump()
            self.wing_sound.play()

        # Actualizar la posición de los sprites
        self.bird_group.update()
        self.ground_group.update()
        self.pipe_group.update()

        # Generar nuevos elementos del juego
        if is_off_screen(self.ground_group.sprites()[0]):
            self.ground_group.remove(self.ground_group.sprites()[0])
            new_ground = Ground(GROUND_WIDHT - 20)
            self.ground_group.add(new_ground)

        # Manejo de tuberías (removido la lógica de recompensa específica por pasar tuberías para simplificar,
        # siguiendo el ejemplo funcional que solo recompensa por supervivencia + penalización por colisión)
        pipes_to_remove = []
        for pipe in self.pipe_group.sprites():
            if is_off_screen(pipe):
                pipes_to_remove.append(pipe)
            # Si quieres añadir recompensa por pasar tuberías,
            # necesitas reintroducir el atributo 'passed' en la clase Pipe y la lógica aquí.
            # Ejemplo:
            # if not hasattr(pipe, 'passed'): # Si no tiene el atributo, lo inicializamos
            #     pipe.passed = False
            # if not pipe.passed and self.bird.rect.left > pipe.rect.right:
            #     # Lógica para evitar doble recompensa por par de tuberías y sumar puntos
            #     # ... (la lógica anterior que tenías)
            #     self.score += 1
            #     self.point_sound.play()


        # Eliminar las tuberías que se han salido de la pantalla
        for pipe in pipes_to_remove:
            self.pipe_group.remove(pipe)

        # Si se han quitado tuberías, añadir nuevas
        if len(self.pipe_group) < 4: # Suponiendo que siempre hay 2 pares visibles (4 tuberías)
            pipes = get_random_pipes(SCREEN_WIDHT * 2) # Aparecen más lejos para no solaparse
            self.pipe_group.add(pipes[0])
            self.pipe_group.add(pipes[1])

        # Detección de colisiones
        if (pygame.sprite.groupcollide(self.bird_group, self.ground_group, False, False, pygame.sprite.collide_mask) or
            pygame.sprite.groupcollide(self.bird_group, self.pipe_group, False, False, pygame.sprite.collide_mask) or
            self.bird.rect.y < 0): # También si el pájaro sale por arriba
            reward = -100 # Penalización por chocar (revertido a -100 para coincidir con el ejemplo funcional)
            done = True
            self.game_over = True
            self.hit_sound.play() # Usa el objeto Sound pre-cargado

        # Dibujar en pantalla (visualización)
        self.screen.blit(self.BACKGROUND, (0, 0))
        self.bird_group.draw(self.screen)
        self.pipe_group.draw(self.screen)
        self.ground_group.draw(self.screen)

        # Mostrar score
        score_text = self.font.render(str(self.score), True, (255, 255, 255))
        self.screen.blit(score_text, (SCREEN_WIDHT // 2 - score_text.get_width() // 2, 50))

        pygame.display.update()
        self.clock.tick(GAME_SPEED)

        next_state = self._get_state()
        return next_state, reward, done, {}

    def close(self):
        pygame.quit()
        exit()