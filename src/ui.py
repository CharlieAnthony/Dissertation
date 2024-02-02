import pygame

# Constants
SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 720


class EnvironmentInterface:
    def __init__(self, environment):
        # inits
        self.environment = environment

        # ui stuff
        pygame.init()
        self.externalMap = pygame.image.load("map.png")
        pygame.display.set_caption("SLAM simulation")
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        self.screen.blit(self.externalMap, (0, 0))

    def get_environment(self):
        return self.environment

    def get_screen(self):
        return self.screen
