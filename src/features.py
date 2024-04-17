import numpy as np
import math
from fractions import Fraction
from scipy.odr import *
class feature_dectection:

    def __init__(self):
        self.two_points = None

        self.epsilon = 10  # maximum distance from a point to a line
        self.delta = 20  # maximum distance between two points
        self.Snum = 6  # number of points to fit a line
        self.Pmin = 20  # minimum points seed segment should have
        self.Gmax = 20  # maximum distance between two points in a line segment

        self.seed_segs = []
        self.line_segs = []
        self.points = []
        self.line_params = None
        self.Np = len(self.points) - 1
        self.Lmin = 20  # minimum length of a line segment
        self.Lr = 0  # real length of line segment
        self.Pr = 0  # number of points in line segment
        self.feats = []
        self.association_thres = 1

    @staticmethod
    def euclidean_distance(point1, point2):
        """
        Calculates the Euclidean distance between two points
        :param point1:
        :param point2:
        :return:
        """
        x = (point1[0] - point2[0]) ** 2
        y = (point1[1] - point2[1]) ** 2
        return math.sqrt(x + y)

    def dist_point_to_line(self, params, point):
        """
        Calculates the distance between a point and a line
        :param params:
        :param point:
        :return:
        """
        a, b, c = params
        distance = abs(a * point[0] + b * point[1] + c) / math.sqrt(a ** 2 + b ** 2)
        return distance

    def line_to_point(self, m, b):
        """
        Extract two points from a line given a line equation
        :param m: slope
        :param b: y-intercept
        :return: line parameters
        """
        x = 5
        y = m * x + b
        x2 = 2000
        y2 = m * x2 + b
        return [(x, y), (x2, y2)]

    def line_form_to_slope_intercept(self, a, b, c):
        """
        Converts a general line equation to slope-intercept form
        :param a: x coefficient
        :param b: y coefficient
        :param c: constant
        :return: slope and y-intercept
        """
        m = -a / b
        n = -c / b
        return m, n

    def slope_intercept_to_general(self, m, n):
        """
        Converts a slope-intercept line equation to general form
        :param m: slope
        :param n: y-intercept
        :return: general form parameters
        """
        a = -m
        b = 1
        c = -n
        if a < 0:
            a = -a
            b = -b
            c = -c
        den_a = Fraction(a).limit_denominator(1000).as_integer_ratio()[1]
        den_c = Fraction(c).limit_denominator(1000).as_integer_ratio()[1]
        gcd = np.gcd(den_a, den_c)
        lcm = den_a * den_c / gcd

        return a * lcm, b * lcm, c * lcm

    def line_intersect_general(self, params1, params2):
        """
        Calculates the intersection between two lines
        :param params1: 1st line parameters
        :param params2: 2nd line parameters
        :return: intersection point
        """
        a1, b1, c1 = params1
        a2, b2, c2 = params2

        denominator = (b1 * a2 - a1 * b2)
        if denominator == 0:
            return None

        x = (c1 * b2 - b1 * c2) / denominator
        y = (a1 * c2 - a2 * c1) / denominator
        return x, y

    def points_to_line(self, point1, point2):
        """
        Extract the line parameters from two points
        :param point1:
        :param point2:
        :return: line parameters
        """
        m, b = 0, 0
        if point2[0] == point1[0]:
            pass
        else:
            m = (point2[1] - point1[1]) / (point2[0] - point1[0])
            b = point1[1] - m * point1[0]
        return m, b

    def projection_point_to_line(self, point, m, b):
        """
        Projects a point onto a line
        :param point:
        :param m: slope
        :param b: y-intercept
        :return: projected point
        """
        x, y = point
        m2 = -1 / m
        c2 = y - m2 * x
        x_intercept = - (b - c2) / (m - m2)
        y_intercept = m2 * x_intercept + c2
        return x_intercept, y_intercept

    def angle_dist_to_coord(self, distance, angle, robot_pos):
        """
        Converts a distance and angle to a position
        :param distance: distance from the robot
        :param angle: angle from the robot
        :param robot_pos: robot position
        :return: position
        """
        angle = math.radians(angle)
        x = (distance * math.cos(angle)) + robot_pos[0]
        y = (-distance * math.sin(angle)) + robot_pos[1]
        return (int(x), int(y))

    def set_laser_points(self, data):
        """
        Sets the laser points
        :param data: laser data
        :return:
        """
        self.points = []
        if not data:
            pass
        else:
            for p in data:
                coord = self.angle_dist_to_coord(p[0], p[1], p[2])
                # coord = (int(p[0]), int(p[1]))
                self.points.append([coord, p[1]])
        self.Np = len(self.points) - 1

    def linear_func(self, p, x):
        """
        Linear function
        :param p: parameters
        :param x: x value
        :return: y value
        """
        m, b = p
        return m * x + b

    def fit_line(self, laser_points):
        """
        Fits a line to the laser points
        :param laser_points: laser points
        :return: line parameters
        """
        x = np.array([p[0][0] for p in laser_points])
        y = np.array([p[0][1] for p in laser_points])
        linear = Model(self.linear_func)
        data = RealData(x, y)
        odr_model = ODR(data, linear, beta0=[0., 0.])
        out = odr_model.run()
        m, b = out.beta
        return m, b

    def predict_point(self, line_params, sensed_points, robot_pos):
        """
        Predicts the next point
        :param line_params: line parameters
        :param sensed_points: sensed points
        :param robot_pos: robot position
        :return: predicted point
        """
        m, b = self.points_to_line(robot_pos, sensed_points)
        params1 = self.slope_intercept_to_general(m, b)
        preds = self.line_intersect_general(params1, line_params)
        if preds:
            return preds[0], preds[1]
        else:
            return None

    def seed_segment_detection(self, robot_pos, break_point_ind):
        flag = True
        self.Np = max(0, self.Np)
        self.seed_segs = []
        for i in range(break_point_ind, (self.Np - self.Pmin)):
            predicted_points_to_draw = []
            j = i + self.Snum
            m, c = self.fit_line(self.points[i:j])

            params = self.slope_intercept_to_general(m, c)

            for k in range(i, j):
                predicted_point = self.predict_point(params, self.points[k][0], robot_pos)
                if not predicted_point:
                    continue
                predicted_points_to_draw.append(predicted_point)
                d1 = self.euclidean_distance(predicted_point, self.points[k][0])

                if d1 > self.delta:
                    flag = False
                    break

                d2 = self.dist_point_to_line(params, self.points[k][0])

                if d2 > self.epsilon:
                    flag = False
                    break

            if flag:
                self.line_params = params
                return [self.points[i:j], predicted_points_to_draw, (i, j)]
        return False

    def seed_segment_growing(self, indices, break_point):
        line_eq = self.line_params
        i, j = indices
        PB, PF = max(break_point, i - 1), min(j + 1, len(self.points) - 1)
        while self.dist_point_to_line(line_eq, self.points[PF][0]) < self.epsilon:
            if PF > self.Np - 1:
                break
            else:
                m, b = self.fit_line(self.points[PB:PF])
                line_eq = self.slope_intercept_to_general(m, b)
                POINT = self.points[PF][0]

            PF = PF + 1
            NEXTPOINT = self.points[PF][0]
            if self.euclidean_distance(POINT, NEXTPOINT) > self.Gmax:
                break
        PF = PF - 1

        while self.dist_point_to_line(line_eq, self.points[PB][0]):
            if PB < break_point:
                break
            else:
                m, b = self.fit_line(self.points[PB:PF])
                line_eq = self.slope_intercept_to_general(m, b)
                POINT = self.points[PB][0]
            PB = PB - 1
            NEXTPOINT = self.points[PB][0]
            if self.euclidean_distance(POINT, NEXTPOINT) > self.Gmax:
                break
        PB = PB + 1

        LR = self.euclidean_distance(self.points[PB][0], self.points[PF][0])
        # PR = len(self.LASERPOINTS[PB:PF])
        PR = PF - PB
        # print(f"PR: {PR}, PB: {PB}, PF: {PF}")

        if (LR >= self.Lmin) and (PR >= self.Pmin):
            self.line_params = line_eq
            m, b = self.line_form_to_slope_intercept(line_eq[0], line_eq[1], line_eq[2])
            self.two_points = self.line_to_point(m, b)
            self.line_segs.append((self.points[PB + 1][0], self.points[PF - 1][0]))
            return [self.points[PB:PF], self.two_points,
                    (self.points[PB + 1][0], self.points[PF - 1][0]), PF, line_eq, (m, b)]
        else:
            return False

    def line_feats_to_point(self):
        new_rep = []

        for feature in self.feats:
            projection = self.projection_point_to_line((0, 0), feature[0][0], feature[0][1])
            new_rep.append([feature[0], feature[1], projection])

        return new_rep

    def line_similarity(self, line1_endpoint1, line1_endpoint2, line2_endpoint1, line2_endpoint2):
        """
        Data association
        :param line1_endpoint1: line 1 endpoint 1
        :param line1_endpoint2: line 1 endpoint 2
        :param line2_endpoint1: line 2 endpoint 1
        :param line2_endpoint2: line 2 endpoint 2
        :return: True if the lines are the same
        """
        d1 = self.euclidean_distance(line1_endpoint1, line2_endpoint1)
        d2 = self.euclidean_distance(line1_endpoint2, line2_endpoint2)
        d3 = self.euclidean_distance(line1_endpoint1, line2_endpoint2)
        d4 = self.euclidean_distance(line1_endpoint2, line2_endpoint1)
        if d1 < self.association_thres and d2 < self.association_thres:
            return True
        elif d3 < self.association_thres and d4 < self.association_thres:
            return True
        else:
            return False


    def landmark_association(self, endpoint1, endpoint2, landmarks, state):
        """
        Landmark association
        :param features: features
        :return:
        """
        measurements = []
        agent_x, agent_y, agent_bearing = state[0], state[1], state[2]
        for n, landmark in enumerate(landmarks):
            landmark_1 = (landmark[0][0] * 0.02, landmark[0][1] * 0.02)
            landmark_2 = (landmark[1][0] * 0.02, landmark[1][1] * 0.02)
            if self.line_similarity(endpoint1, endpoint2, landmark_1, landmark_2):
                landmark_x, landmark_y = landmark_1[0] + landmark_2[0] / 2, landmark_1[1] + landmark_2[1] / 2
                dist = np.linalg.norm(np.array([landmark_x - agent_x, landmark_y - agent_y]))
                angle = np.arctan2(landmark_y - agent_y, landmark_x - agent_x) - agent_bearing
                angle = np.arctan2(np.sin(angle), np.cos(angle))
                measurements.append([dist, angle, n])
                # print("data associated")
        return measurements
