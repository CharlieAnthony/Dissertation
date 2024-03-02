import pygame
import sys
import cv2
from agent import Agent
from environment import Environment
from sensors import LidarSensor
from ui import EnvironmentInterface
from features import *

pointcloud = []


def main():
    # Initialize environment
    env_width = 1280
    env_height = 720
    map_path = "map1.png"
    map = cv2.imread(map_path)
    environment = Environment.img_to_env(map)
    interface = EnvironmentInterface(environment)
    lidar = LidarSensor(300, 180, environment)
    feature_map = feature_dectection()

    running = True
    while running:

        sensor_on = False
        feature_detection = True
        break_point_ind = 0
        endpoints = [0, 0]
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
            if d is not False:
                feature_map.set_laser_points(d)
                while break_point_ind < (feature_map.NP - feature_map.PMIN):
                    seed_seg = feature_map.seed_segment_detection(pos, break_point_ind)
                    if seed_seg == False:
                        break
                    else:
                        seed_segment = seed_seg[0]
                        predicted_points_todraw = seed_seg[1]
                        INDICES = seed_seg[2]
                        results = feature_map.seed_segment_growing(INDICES, break_point_ind)
                        if results == False:
                            break_point_ind = INDICES[1]
                            continue
                        else:
                            line_eq = results[1]
                            m, c = results[5]
                            line_seg = results[0]
                            OUTMOST = results[2]
                            break_point_ind = results[3]

                            endpoints[0] = feature_map.projection_point2line(OUTMOST[0], m, c)
                            endpoints[1] = feature_map.projection_point2line(OUTMOST[1], m, c)

                            feature_map.FEATURES.append([[m, c], endpoints])
                            feature_map.FEATURES = feature_map.lineFeats2point()
                            landmark_association(feature_map.FEATURES)

                            color = (255, 0, 0)
                            for point in line_seg:
                                # px, py = point[0]
                                # pygame.draw.circle(interface.get_screen(), (0, 0, 255), (px, py), 2)
                                # print("point: ", point)
                                pass
                            # pygame.draw.line(interface.get_screen(), color, endpoints[0], endpoints[1], 2)

                points = []
                for reading in d:
                    points.append(feature_map.angle_dist_2_coord(reading[0], reading[1], reading[2]))
                data_to_pointcloud(points)
                # show_pointcloud(interface.get_screen())
            for landmark in Landmarks:
                pygame.draw.line(interface.get_screen(), (0, 0, 255), landmark[1][0], landmark[1][1], 2)
        pygame.display.update()

    pygame.quit()


def main1():
    # Initialize environment
    env_width = 1280
    env_height = 720
    map_path = "map1.png"
    map = cv2.imread(map_path)
    environment = Environment.img_to_env(map)
    interface = EnvironmentInterface(environment, map_path)
    agent = Agent(environment, radius=10, step_size=50)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        agent.move()
        # interface.draw()
        # agent.detect()
        # pygame.time.wait(100)
        agent.draw_agent(interface.get_screen())


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
        pygame.draw.circle(screen, (0, 255, 0), (int(pos[0]), int(pos[1])), 2)


if __name__ == "__main__":
    main()
