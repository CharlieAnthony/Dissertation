import numpy as np
from scipy.linalg import block_diag
from filterpy.kalman import ExtendedKalmanFilter

class EKF(ExtendedKalmanFilter):
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
        else:  # Turning
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