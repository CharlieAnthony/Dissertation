from features import *
from sensors import LidarSensor
from ekf import EKF
import pygame
import numpy as np


class Agent:

    def __init__(self, environment, radius=3, step_size=5, noise=0.5):
        self.real_position = (10, 10)
        self.predicted_position = (10, 10)
        self.bearing = np.random.randint(low=0, high=359)
        self.velocity = 0
        self.EKF = EKF(0.1, 0.1, 0.1, 0.1)
        self.feature_detection = feature_dectection()
        self.radius = radius
        self.step_size = step_size
        self.env = environment
        self.lidar = LidarSensor(300, 180, self.env)


    def move(self):
        # random walk algorithm
        angle = np.random.randint(low=0, high=90)
        new_bearing = self.bearing + angle - 45
        self.bearing = new_bearing % 360
        self.step(self.step_size, self.bearing)



    def step(self, dist, angle):
        # check to see if there are obstacles between (x1, y1) and (x2, y2)
        for i in range(dist):
            x, y = self.feature_detection.angle_dist_2_coord(i, angle, self.real_position)
            x += np.random.normal(0, 0.5)
            y += np.random.normal(0, 0.5)
            pred_x, pred_y = self.feature_detection.angle_dist_2_coord(i, angle, self.predicted_position)
            if self.env.get_cell_val(x, y) != 1:
                self.real_position = (x, y)
                self.predicted_position = (pred_x, pred_y)
            else:
                break




    def detect(self):
        readings = self.lidar.detect(None, self.env, position=self.real_position)
        break_point_ind = 0
        endpoints = [0, 0]
        if readings is not False:
            self.feature_detection.set_laser_points(readings)
            while break_point_ind < (self.feature_detection.NP - self.feature_detection.PMIN):
                seed_seg = self.feature_detection.seed_segment_detection(self.real_position, break_point_ind)
                if seed_seg == False:
                    break
                else:
                    seed_segment = seed_seg[0]
                    INDICES = seed_seg[2]
                    results = self.feature_detection.seed_segment_growing(INDICES, break_point_ind)
                    if results == False:
                        break_point_ind = INDICES[1]
                        continue
                    else:
                        line_eq = results[1]
                        m, c = results[5]
                        OUTMOST = results[2]
                        break_point_ind = results[3]

                        endpoints[0] = self.feature_detection.projection_point2line(OUTMOST[0], m, c)
                        endpoints[1] = self.feature_detection.projection_point2line(OUTMOST[1], m, c)

                        self.feature_detection.FEATURES.append([[m, c], endpoints])
                        self.feature_detection.FEATURES = self.feature_detection.lineFeats2point()
                        landmark_association(self.feature_detection.FEATURES)

    def draw_agent(self, screen):
        x, y = self.real_position
        pygame.draw.circle(screen, (255, 0, 0), (x, y), self.radius)
        pygame.draw.circle(screen, (0, 255, 0), (int(self.predicted_position[0]), int(self.predicted_position[1])), self.radius - 1)
        return screen

    def draw_landmarks(self, screen):
        for landmark in Landmarks:
            pygame.draw.line(screen, (0, 0, 255), landmark[1][0], landmark[1][1], 2)
