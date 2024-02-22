from features import *
from sensors import LidarSensor


class Agent():

    def __init__(self, environment):
        self.position = (0, 0)
        self.bearing = 0
        self.velocity = 0
        self.feature_detection = feature_dectection()
        self.env = environment
        self.lidar = LidarSensor(300, 180, self.env)


    def move(self, dist):
        pass


    def rotate(self, deg):
        pass

    def next_move(self):
        pass

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