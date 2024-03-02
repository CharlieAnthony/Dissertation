from features import *
from sensors import LidarSensor
import pygame
import numpy as np


class Agent:

    def __init__(self, environment, radius=3, step_size=5):
        self.position = (10, 10)
        self.bearing = np.random.randint(low=0, high=359)
        self.velocity = 0
        self.feature_detection = feature_dectection()
        self.radius = radius
        self.step_size = step_size
        self.env = environment
        self.lidar = LidarSensor(300, 180, self.env)

    def move(self):
        # random walk algorithm
        angle = np.random.randint(low=0, high=90)
        new_bearing = self.bearing + angle - 45
        self.bearing = new_bearing if new_bearing < 360 else new_bearing - 360
        self.step(self.step_size, self.bearing)

    def step(self, dist, angle):
        x1, y1 = self.position
        # check to see if there are obstacles between (x1, y1) and (x2, y2)
        for i in range(dist):
            x, y = self.feature_detection.angle_dist_2_coord(i, angle, (x1, y1))
            if self.env.get_cell_val(x, y) != 1:
                self.position = (x, y)
            else:
                break

    def detect(self):
        readings = self.lidar.detect(None, self.env, position=self.position)
        break_point_ind = 0
        endpoints = [0, 0]
        if readings is not False:
            self.feature_detection.set_laser_points(readings)
            while break_point_ind < (self.feature_detection.NP - self.feature_detection.PMIN):
                seed_seg = self.feature_detection.seed_segment_detection(self.position, break_point_ind)
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
        x, y = self.position
        pygame.draw.circle(screen, (255, 0, 0), (x, y), self.radius)
        return screen
