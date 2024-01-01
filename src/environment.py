# environment.py
import pygame

class Environment:
    def __init__(self, width, height, color=(0, 0, 0)):
        self.width = width
        self.height = height
        self.color = color
        self.obstacles = []
        self.agents = []

    def add_obstacle(self, x, y, width, height):
        # Obstacles will be represented as rectangles for simplicity
        self.obstacles.append(pygame.Rect(x, y, width, height))

    def add_agent(self, x, y, radius):
        # Agents represented as a circle
        # Triple stores x, y position and radius
        self.agents.append((x, y, radius))

    def draw(self, screen):
        pygame.draw.rect(screen, self.color, (0, 0, self.width, self.height), width=2)
        for obstacle in self.obstacles:
            pygame.draw.rect(screen, (0, 0, 0), obstacle, 2)
        for agent in self.agents:
            pygame.draw.circle(screen, (0, 160, 0), (agent[0], agent[1]), agent[2], width=2)