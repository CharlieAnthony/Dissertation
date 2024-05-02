import math

import scipy

from features import *
from sensors import LidarSensor
from ekf import EKF
import pygame
import pygame.gfxdraw
import numpy as np


class Agent:

    def __init__(self, environment, landmarks, radius=3, noise=0.5, init_pos=np.array([1, 1, 0])):
        # self.real_position = (1, 1)
        # self.bearing = np.random.randint(low=0, high=359)
        self.set_state(init_pos)
        self.feature_detection = feature_dectection()
        self.radius = radius
        self.env = environment
        self.lidar = LidarSensor(150, 360, self.env)
        self.n_state = 3
        self.max_velocity = 2.0
        self.max_omega = 2.0
        self.n_landmarks = len(landmarks)
        self.mu = np.zeros((self.n_state + 2 * self.n_landmarks, 1))
        self.mu[:] = np.nan
        self.mu[0:3] = np.expand_dims(init_pos, axis=1)
        self.sigma = np.zeros((self.n_state + 2 * self.n_landmarks, self.n_state + 2 * self.n_landmarks))
        np.fill_diagonal(self.sigma, 100)
        self.ekf = EKF(landmarks=landmarks)
        self.sigma[0:3, 0:3] = 0.05 * np.eye(3)

    def set_state(self, state):
        """
        Set the state of the agent
        """
        # state[0] = x position (m)
        # state[1] = y position (m)
        # state[2] = bearing (rad)
        self.state = state
        self.state[2] = self.state[2] % (2 * np.pi)

    def get_state(self):
        """
        Get the state of the agent
        """
        return self.state

    def get_pos(self):
        """
        Get the position of the agent
        """
        return self.state[0:2]

    def update_u(self, u):
        """
        Update the control input of the agent
        :param u: control input
        :return: None
        """
        # u = controls
        #     u[0] = forward velocity (m/s)
        #     u[1] = angular velocity (deg/s)
        # Exploration strat: wall follow
        #     - continues to walk forward whilst no obstacles detected directly in front
        #     - if obstacle detected, turn left or right
        # convert pos from meters into pixels
        x = int(self.state[0] / 0.02)
        y = int(self.state[1] / 0.02)
        reading = self.lidar.detect(None, self.env, position=(x, y))  # [[(dist, angle), ...], [], ...]
        # if no points detected, move forward
        if reading is False:
            u[0] = 2
            u[1] = 0
            return u
        r = {int(450 - v) % 360: k for k, v, (_, _) in reading}
        agent_bearing = self.state_to_deg(self.state[2])
        flag = None
        # if points ahead, turn
        for i in range(360 + agent_bearing - 5, 360 + agent_bearing):
            i %= 360
            if i in r.keys() and r[i] < 30:
                flag = i
                break
        if flag is not None:
            u[0] = 0
            u[1] = 2
        else:
            for i in range(360 + agent_bearing, 360 + agent_bearing + 5):
                i %= 360
                if i in r.keys() and r[i] < 30:
                    flag = i
                    break
            if flag is not None:
                u[0] = 0
                u[1] = -2
            else:
                u[0] = 2
                u[1] = 0
        return u

    def move(self, u, dt):
        """
        Move the agent according to the control input
        :param u: control input
        :param dt: time step
        :return: None
        """
        y = np.zeros(5)
        y[:3] = self.state
        y[3:] = u
        result = scipy.integrate.solve_ivp(self.motion_eqs, [0, dt], y)
        self.state = result.y[:3, -1]
        self.state[2] = self.state[2] % (2 * np.pi)

    def motion_eqs(self, t, y):
        """
        Differential equations for the motion of the agent
        :param t: time
        :param y: state
        :return: derivative of the state
        """
        theta = y[2]
        v = max(min(y[3], self.max_velocity), -self.max_velocity)  # forward velocity
        omega = max(min(y[4], self.max_omega), -self.max_omega)  # forward and angular velocity
        new_state = np.zeros(5)
        new_state[0] = v * np.cos(theta)
        new_state[1] = v * np.sin(theta)
        new_state[2] = omega
        return new_state

    def detect(self):
        """
        Detect features in the environment applying seeded region growing
        :return: detected features
        """
        pos = (int(self.state[0] / 0.02), int(self.state[1] / 0.02))
        readings = self.lidar.detect(None, self.env, position=pos)
        break_point_index = 0
        results = None
        if readings is not False:
            self.feature_detection.set_laser_points(readings)
            # iterate through all the points
            while break_point_index < (self.feature_detection.Np - self.feature_detection.Pmin):
                # seed segment detection
                seed_seg = self.feature_detection.seed_segment_detection(pos, break_point_index)
                if seed_seg == False:
                    break
                else:
                    indices = seed_seg[2]
                    # seed segment growing
                    results = self.feature_detection.seed_segment_growing(indices, break_point_index)
                    # if no features detected, continue
                    if results == False:
                        break_point_index = indices[1]
                        continue
                    else:
                        break_point_index = results[3]
            return results
        return False

    def draw_agent(self, screen):
        """
        Draw the agent on the screen
        :param screen: pygame screen
        :return: None
        """
        x = int(self.state[0] / 0.02)
        y = int(self.state[1] / 0.02)
        circle = pygame.draw.circle(screen, (255, 0, 0), (x, y), self.radius)  # agent, no aa
        # pygame.gfxdraw.filled_circle(screen, x, y, self.radius, (255, 0, 0))  # agent, aa
        # direction indicator
        x2 = x + 10 * np.cos(self.state[2])
        y2 = y + 10 * np.sin(self.state[2])
        line = pygame.draw.line(screen, (0, 255, 0), (x, y), (x2, y2), 2)
        return [circle, line]

    def show_agent_estimate(self, screen, mu, sigma):
        """
        Show the estimated position of the agent
        :param screen: pygame screen
        :param mu: mean
        :param sigma: covariance
        :return: list of objects
        """
        x = int(mu[0] / 0.02)
        y = int(mu[1] / 0.02)
        # extract the eigenvalues and eigenvectors
        eigenvals, angle = self.ekf.sigma2transform(sigma[0:2, 0:2])
        # convert the eigenvalues from meters to pixels
        width = (int(eigenvals[0] / 0.02), int(eigenvals[1] / 0.02))
        return self.draw_agent_uncertainty(screen, (x, y), width, angle)

    def draw_agent_uncertainty(self, screen, center, width, angle):
        """
        Draw the uncertainty of the agent on the screen
        :param screen: pygame screen
        :param center: center of the ellipse
        :param width: width of the ellipse
        :param angle: angle of the ellipse
        :return: list of objects
        """
        x = center[0] - int(width[0] / 2)
        y = center[1] - int(width[1] / 2)
        target_rect = pygame.Rect(x, y, width[0], width[1])
        shape_surf = pygame.Surface(target_rect.size, pygame.SRCALPHA)
        ellipse = pygame.draw.ellipse(shape_surf, (255, 0, 0), (0, 0, *target_rect.size), 2)
        rotated_surf = pygame.transform.rotate(shape_surf, angle)
        screen.blit(rotated_surf, rotated_surf.get_rect(center=target_rect.center))
        return [ellipse]

    def show_landmark_uncertainty(self, mu, sigma, screen):
        """
        Show the uncertainty of the landmarks
        :param mu: mean
        :param sigma: covariance
        :param screen: pygame screen
        :return: list of objects
        """
        objects = []
        # iterate through all the landmarks
        for i in range(self.n_landmarks):
            x = mu[self.n_state + i * 2]
            y = mu[self.n_state + i * 2 + 1]
            sig = self.get_sigma_value(sigma, i)
            # if landmark has been discovered
            if ~np.isnan(x):
                pixel_pos = (int(x / 0.02), int(y / 0.02))
                eigenvals, angle = self.ekf.sigma2transform(sig)
                # if the uncertainty is small enough, draw the ellipse
                if np.max(eigenvals) < 20:
                    sigma_pixel = (int(eigenvals[0] / 0.02), int(eigenvals[1] / 0.02))
                    objects.append(self.draw_agent_uncertainty(screen, pixel_pos, sigma_pixel, angle)[0])
        return objects

    def get_sigma_value(self, sigma, i):
        """
        Get the sigma value for landmark i
        :param sigma:
        :param i:
        :return:
        """
        return sigma[self.n_state + i * 2:self.n_state + i * 2 + 2, self.n_state + i * 2:self.n_state + i * 2 + 2]


    def state_to_deg(self, angle):
        """
        Convert bearing from radians to degrees
        """
        angle = int(np.rad2deg(angle))
        if angle < 0:
            angle = 360 - abs(angle)
        angle += 90
        return angle % 360

    def simple_detect(self, state, landmarks):
        """
        Detects landmarks and returns the measurements
        :param state: state of the robot
        :param landmarks: landmark positions
        :return: measurements
        """
        robot_fov = 5
        pos_x, pos_y, bearing = state[0], state[1], state[2]
        res = []
        for (n, l) in enumerate(landmarks):
            landmark_x, landmark_y = l
            dist = np.linalg.norm(np.array([landmark_x - pos_x, landmark_y - pos_y]))
            phi = np.arctan2(landmark_y - pos_y, landmark_x - pos_x) - bearing
            phi = np.arctan2(np.sin(phi), np.cos(phi))
            if dist < robot_fov:
                res.append([dist, phi, n])
        return res
