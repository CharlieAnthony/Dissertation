import pygame

# Constants
SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 720


class EnvironmentInterface:
    def __init__(self, environment, map=None):
        # inits
        self.environment = environment


        # ui stuff
        pygame.init()
        pygame.display.set_caption("SLAM simulation")
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.DOUBLEBUF)
        if map is not None:
            self.externalMap = pygame.image.load(map)
            self.screen.blit(self.externalMap, (0, 0))

    def get_environment(self):
        return self.environment

    def get_screen(self):
        return self.screen

    def draw(self):
        self.screen.fill((255, 255, 255))
        self.screen.blit(self.externalMap, (0, 0))
        # self.environment.draw(self.screen)
        pygame.display.update()