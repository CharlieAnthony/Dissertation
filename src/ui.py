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
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("2D Environment")

    def get_environment(self):
        return self.environment

    def get_screen(self):
        return self.screen
