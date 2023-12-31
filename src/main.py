import pygame
import sys
from agent import Agent
from environment import Environment

# Initialize pygame
pygame.init()

# Constants
SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 720
BACKGROUND_COLOR = (255, 255, 255)

# Create screen and clock objects
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("2D Environment")
clock = pygame.time.Clock()



# Create environment
env_width = 1000  # Adjust as needed
env_height = 600  # Adjust as needed
environment = Environment(env_width, env_height)
environment.add_obstacle(300, 200, 50, 100)  # Sample obstacle

# Create agents
agents = [Agent(env_width, env_height, sensor_range=100, show_sensors=True) for _ in range(5)]

def main():
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        screen.fill(BACKGROUND_COLOR)
        environment.draw(screen)

        for agent in agents:
            agent.update(agents)
            agent.draw(screen)
            agent.handle_agent_collisions(agents, environment)
            agent.handle_environment_collisions(environment)

        pygame.display.flip()
        clock.tick(60)

if __name__ == "__main__":
    main()
