# environment.py
import pygame

class Environment:
    def __init__(self, width, height, color=(0, 0, 0)):
        self.width = width
        self.height = height
        self.color = color
        self.obstacles = []
        # Adds environment as an obstacle
        # self.add_obstacle(0, 0, width, height)

    def add_obstacle(self, x, y, width, height):
        # Obstacles will be represented as rectangles for simplicity
        self.obstacles.append(pygame.Rect(x, y, width, height))

    def draw(self, screen):
        pygame.draw.rect(screen, self.color, (0, 0, self.width, self.height), width=2)
        for obstacle in self.obstacles:
            pygame.draw.rect(screen, (0, 0, 0), obstacle, 2)
