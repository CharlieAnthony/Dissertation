import pygame
import sys
from agent import Agent
from environment import Environment
from ui import EnvironmentInterface


pointcloud = []

def main():
    # Initialize environment
    env_width = 1280
    env_height = 720
    environment = create_environment(env_width, env_height)
    interface = EnvironmentInterface(environment)


    # while True:
    #     for event in pygame.event.get():
    #         if event.type == pygame.QUIT:
    #             pygame.quit()
    #             sys.exit()
    #
    #     screen.fill(BACKGROUND_COLOR)
    #     environment.draw(screen)
    #
    #     for agent in agents:
    #         agent.update(agents)
    #         # agent.draw(screen)
    #         agent.handle_agent_collisions(environment)
    #         agent.handle_environment_collisions(environment)
    #
    #     pygame.display.flip()
#     clock.tick(60)


def data_to_pointcloud(positions):
    """
    Adds data to pointcloud
    :param positions: positions of detected objects
    :return:
    """
    global pointcloud
    for pos in positions:
        if pos not in pointcloud:
            pointcloud.append(pos)


def show_pointcloud(screen):
    """
    Shows pointcloud
    :return:
    """
    global pointcloud
    for pos in pointcloud:
        pygame.draw.circle(screen, (0, 255, 0), (int(pos[0]), int(pos[1])), 5)

def create_environment(width, height):
    env = Environment(width, height)
    return env


if __name__ == "__main__":
    main()
