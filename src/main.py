import pygame
import sys
import cv2
from agent import Agent
from environment import Environment, img_to_env
from sensors import LidarSensor
from ui import EnvironmentInterface


pointcloud = []

def main():
    # Initialize environment
    env_width = 1280
    env_height = 720
    map = cv2.imread('map.png')
    environment = img_to_env(map)
    interface = EnvironmentInterface(environment)
    lidar = LidarSensor(50, 180, environment)


    running = True

    while running:
        sensor_on = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if pygame.mouse.get_focused():
                sensor_on = True
            elif not pygame.mouse.get_focused():
                sensor_on = False

        if sensor_on:
            pos = pygame.mouse.get_pos()
            d = lidar.detect(None, environment, position=pos)
            p = lidar.sensor_to_position(d, position=pos)
            data_to_pointcloud(p)
            show_pointcloud(interface.get_screen())


        pygame.display.update()

    pygame.quit()





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


if __name__ == "__main__":
    main()
