import numpy as np
from scipy.linalg import block_diag
from filterpy.kalman import ExtendedKalmanFilter

class old_EKF(ExtendedKalmanFilter):
    def __init__(self, dt, wheelbase, std_acc, std_yawrate):
        ExtendedKalmanFilter.__init__(self, 4, 2)

        self.dt = dt
        self.wheelbase = wheelbase
        self.std_acc = std_acc
        self.std_yawrate = std_yawrate

        self.M = np.diag([self.std_acc**2, self.std_yawrate**2])

        self.x = np.array([[0, 0, 0, 0]]).T  # state
        self.P = np.eye(4)  # uncertainty covariance
        self.R = np.eye(2)  # state uncertainty
        self.Q = block_diag(self.M, self.M)  # process uncertainty

    def f(self, x, u):
        # State transition function
        X, Y, V, YAW = x[0], x[1], x[2], x[3]
        a, yawrate = u[0], u[1]

        if abs(yawrate) < 0.001:  # Driving straight
            X1 = X + V*self.dt*np.cos(YAW)
            Y1 = Y + V*self.dt*np.sin(YAW)
            V1 = V + a*self.dt
            YAW1 = YAW
        else:
            X1 = X + (V/yawrate)*(np.sin(YAW + yawrate*self.dt) - np.sin(YAW))
            Y1 = Y + (V/yawrate)*(np.cos(YAW) - np.cos(YAW + yawrate*self.dt))
            V1 = V + a*self.dt
            YAW1 = YAW + yawrate*self.dt

        return np.array([[X1, Y1, V1, YAW1]]).T

    def h(self, x):
        # Measurement function
        return x[0:2, :]

    def Fx(self, x, u):
        # Jacobian of f
        F_result = np.eye(4)
        return F_result

    def Hx(self, x):
        # Jacobian of h
        H_result = np.zeros((2, 4))
        H_result[0, 0] = 1
        H_result[1, 1] = 1
        return H_result

class EKF:

    def __init__(self):
        # robot parameters
        self.n_state = 3
        self.n_landmarks = 1

        # ekf estimation variables
        self.mu = np.zeros((self.n_state + 2*self.n_landmarks, 1))
        self.sigma = np.zeros((self.n_state + 2*self.n_landmarks, self.n_state + 2*self.n_landmarks))

        # helpful matrices
        self.Fx = np.block([[np.eye(3), np.zeros((self.n_state, 2*self.n_landmarks))]])

    def prediction_update(self, mu, sigma, u, dt):
        rx, ry, theta = mu[0:3]
        v, w = u[0], u[1]
        # update state estimate
        state_model_mat = np.zeros((self.n_state, 1))
        state_model_mat[0] = -(v/w)*np.sin(theta) + (v/w)*np.sin(theta + w*dt) if abs(w) > 0.001 else v*np.cos(theta)
        state_model_mat[1] = (v/w)*np.cos(theta) - (v/w)*np.cos(theta + w*dt) if abs(w) > 0.001 else v*np.sin(theta)
        state_model_mat[2] = w*dt
        mu += np.transpose(self.Fx).dot(state_model_mat)



        return mu, sigma

    def measurement_update(self, mu, sigma):
        pass