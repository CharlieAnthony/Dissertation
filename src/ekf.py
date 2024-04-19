import numpy as np
from scipy.linalg import block_diag
from filterpy.kalman import ExtendedKalmanFilter


class EKF:

    def __init__(self, landmarks):
        # robot parameters
        self.n_state = 3
        self.landmarks = landmarks
        self.n_landmarks = len(self.landmarks)

        # ekf estimation variables.
        self.mu = np.zeros((self.n_state + 2 * self.n_landmarks, 1))
        self.sigma = np.zeros((self.n_state + 2 * self.n_landmarks, self.n_state + 2 * self.n_landmarks))
        self.mu = np.nan

        # helpful matrices
        self.Fx = np.block([[np.eye(3), np.zeros((self.n_state, 2 * self.n_landmarks))]])

        # noise: m/s, m/s, rad/s
        self.R = np.diag([0.001, 0.001, 0.0005])

    def prediction_update(self, mu, sigma, u, dt):
        """
        Update the state estimate
        :param mu: previous state estimate
        :param sigma: previous state uncertainty
        :param u: control input
        :param dt: time step
        :return: updated mu and sigma
        """
        # Fx defined in init
        theta = mu[2]
        v, w = u[0], u[1]  # velocity and angular velocity
        # update state estimate
        motion_model = np.zeros((self.n_state, 1))
        motion_model[0] = -(v / w) * np.sin(theta) + (v / w) * np.sin(theta + w * dt) if abs(
            w) > 0.01 else v * dt * np.cos(theta)
        motion_model[1] = (v / w) * np.cos(theta) - (v / w) * np.cos(theta + w * dt) if abs(
            w) > 0.01 else v * dt * np.sin(theta)
        motion_model[2] = w * dt
        mu += np.transpose(self.Fx).dot(motion_model)
        # update state uncertainty with model + noise
        state_jacobian = np.zeros((self.n_state, self.n_state))
        state_jacobian[0, 2] = -(v / w) * np.cos(theta) + (v / w) * np.cos(theta + w * dt) if abs(
            w) > 0.01 else -v * dt * np.sin(theta)
        state_jacobian[1, 2] = -(v / w) * np.sin(theta) + (v / w) * np.sin(theta + w * dt) if abs(
            w) > 0.01 else v * dt * np.cos(theta)
        # G has to be same size as sigma for matrix multiplication
        G = np.eye(sigma.shape[0]) + np.transpose(self.Fx).dot(state_jacobian).dot(self.Fx)
        sigma = G.dot(sigma).dot(np.transpose(G)) + np.transpose(self.Fx).dot(self.R).dot(self.Fx)
        return mu, sigma

    def measurement_update(self, mu, sigma, measurements):
        """
        Update agent and landmark estimates
        :param mu: previous state estimate
        :param sigma: previous state uncertainty
        :param measurements: list of measurements
        """
        rx, ry, theta = mu[0, 0], mu[1, 0], mu[2, 0]
        delta_zs = [np.zeros((2, 1)) for _ in range(self.n_landmarks)]
        Ks = [np.zeros((mu.shape[0], 2)) for _ in range(self.n_landmarks)]
        Hs = [np.zeros((2, mu.shape[0])) for _ in range(self.n_landmarks)]
        Q = np.diag([0.003, 0.005])
        for z in measurements:
            (dist, phi, lidx) = z
            mu_landmark = mu[self.n_state + 2 * lidx: self.n_state + 2 * lidx + 2]
            if np.isnan(mu_landmark[0]):
                mu_landmark[0] = rx + dist * np.cos(phi + theta)
                mu_landmark[1] = ry + dist * np.sin(phi + theta)
                mu[self.n_state + 2 * lidx: self.n_state + 2 * lidx + 2] = mu_landmark
            delta = mu_landmark - np.array([[rx], [ry]])
            q = np.linalg.norm(delta) ** 2

            dist_est = np.sqrt(q)
            phi_est = np.arctan2(delta[1, 0], delta[0, 0]) - theta;
            phi_est = np.arctan2(np.sin(phi_est), np.cos(phi_est))
            z_est_arr = np.array([[dist_est], [phi_est]])
            z_act_arr = np.array([[dist], [phi]])
            delta_zs[lidx] = z_act_arr - z_est_arr

            Fxj = np.block([[self.Fx], [np.zeros((2, self.Fx.shape[1]))]])
            Fxj[self.n_state:self.n_state + 2, self.n_state + 2 * lidx:self.n_state + 2 * lidx + 2] = np.eye(2)
            H = np.array([[-delta[0, 0] / np.sqrt(q), -delta[1, 0] / np.sqrt(q), 0, delta[0, 0] / np.sqrt(q),
                           delta[1, 0] / np.sqrt(q)], \
                          [delta[1, 0] / q, -delta[0, 0] / q, -1, -delta[1, 0] / q, +delta[0, 0] / q]])

            H = H.dot(Fxj)
            Hs[lidx] = H  # Added to list of matrices
            Ks[lidx] = sigma.dot(np.transpose(H)).dot(
                np.linalg.inv(H.dot(sigma).dot(np.transpose(H)) + Q))  # Add to list of matrices
        mu_offset = np.zeros(mu.shape)  # Offset to be added to state estimate
        sigma_factor = np.eye(sigma.shape[0])  # Factor to multiply state uncertainty
        for lidx in range(self.n_landmarks):
            mu_offset += Ks[lidx].dot(delta_zs[lidx])  # Compute full mu offset
            sigma_factor -= Ks[lidx].dot(Hs[lidx])  # Compute full sigma factor
        mu = mu + mu_offset  # Update state estimate
        sigma = sigma_factor.dot(sigma)  # Update state uncertainty
        return mu, sigma

    def sigma2transform(self, sigma):
        """
        Convert covariance matrix to ellipse parameters
        :param sigma: covariance matrix
        :return: eigenvalues and degree angle (required for pygame)
        """
        [eigenvals, eigenvecs] = np.linalg.eig(sigma)
        angle = 180. * np.arctan2(eigenvecs[1][0], eigenvecs[0][0]) / np.pi
        return eigenvals, angle
