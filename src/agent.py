from features import *
from sensors import LidarSensor
from ekf import EKF
import pygame
import numpy as np


class Agent:

    def __init__(self, environment, radius=3, step_size=5, noise=0.5, init_pos=np.array([1, 1, 0])):
        # self.real_position = (1, 1)
        # self.bearing = np.random.randint(low=0, high=359)
        self.set_state(init_pos)
        self.velocity = step_size
        self.feature_detection = feature_dectection()
        self.radius = radius
        self.step_size = step_size
        self.env = environment
        self.lidar = LidarSensor(300, 180, self.env)
        self.n_state = 3
        self.n_landmarks = 1
        self.mu = np.zeros((self.n_state + 2 * self.n_landmarks, 1))
        self.mu[:] = np.nan
        self.mu[0:3] = np.expand_dims(init_pos, axis=1)
        self.sigma = np.zeros((self.n_state+2*self.n_landmarks,self.n_state+2*self.n_landmarks))
        np.fill_diagonal(self.sigma, 100)
        self.ekf = EKF()
        self.sigma[0:3, 0:3] = 0.05 * np.eye(3)

    def set_state(self, state):
        """
        state = robot state
            state[0] = x position (m)
            state[1] = y position (m)
            state[2] = bearing (rad)
        """
        self.state = state
        self.state[2] = self.state[2] % (2 * np.pi)

    def get_state(self):
        return self.state

    def get_pos(self):
        return self.state[0:2]

    def update_u(self, u):
        # random walk algorithm
        # angle = np.random.randint(low=0, high=90)
        # new_bearing = self.bearing + angle - 45
        # self.bearing = new_bearing % 360
        # self.step(self.step_size, self.bearing)
        """
        u = controls
            u[0] = forward velocity (m/s)
            u[1] = angular velocity (deg/s)
        """
        u[0] = 2
        r = np.random.randint(low=0, high=3)
        u[1] = (r-1) * 10
        return u

    def move(self, u, dt):
        # old code
        # dist, angle = u[0], u[1]
        # # check to see if there are obstacles between (x1, y1) and (x2, y2)
        # for i in range(dist):
        #     x, y = self.feature_detection.angle_dist_2_coord(i, angle, self.real_position)
        #     x += np.random.normal(0, 0.5)
        #     y += np.random.normal(0, 0.5)
        #     pred_x, pred_y = self.feature_detection.angle_dist_2_coord(i, angle, self.predicted_position)
        #     if self.env.get_cell_val(x, y) != 1:
        #         self.real_position = (x, y)
        #         self.predicted_position = (pred_x, pred_y)
        #         self.velocity = self.step_size
        #     else:
        #         self.velocity = 0
        #         break
        # new code
        bearing_rad = self.state[2]
        # print(f"u0: {u[0]} | state2: {self.state[2]} | dt: {dt}")
        self.state[0] += u[0] * np.cos(bearing_rad) * dt
        self.state[1] += u[0] * np.sin(bearing_rad) * dt
        self.state[2] += u[1] * dt


    def detect(self):
        pos = self.state[0:2]
        readings = self.lidar.detect(None, self.env, position=pos)
        break_point_ind = 0
        endpoints = [0, 0]
        if readings is not False:
            self.feature_detection.set_laser_points(readings)
            while break_point_ind < (self.feature_detection.NP - self.feature_detection.PMIN):
                seed_seg = self.feature_detection.seed_segment_detection(pos, break_point_ind)
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
        x = int(self.state[0] / 0.02)
        y = int(self.state[1] / 0.02)
        pygame.draw.circle(screen, (255, 0, 0), (x, y), self.radius)
        return screen

    def show_agent_estimate(self, screen, mu, sigma):
        x = int(mu[0] / 0.02)
        y = int(mu[1] / 0.02)
        eigenvals, angle = self.ekf.sigma2transform(sigma[0:2, 0:2])
        width = (int(eigenvals[0]/0.02), int(eigenvals[1]/0.02))
        angle = 0
        self.draw_agent_uncertainty(screen, (x, y), width, angle)

    def draw_agent_uncertainty(self, screen, center, width, angle):
        l = center[0] - int(width[0] / 2)
        t = center[1] - int(width[1] / 2)
        target_rect = pygame.Rect(l, t, width[0], width[1])
        shape_surf = pygame.Surface(target_rect.size, pygame.SRCALPHA)
        pygame.draw.ellipse(shape_surf, (255, 0, 0), (0, 0, *target_rect.size), 2)
        rotated_surf = pygame.transform.rotate(shape_surf, angle)
        screen.blit(rotated_surf, rotated_surf.get_rect(center=target_rect.center))
        return screen

    def draw_landmarks(self, screen):
        for landmark in Landmarks:
            pygame.draw.line(screen, (0, 0, 255), landmark[1][0], landmark[1][1], 2)
