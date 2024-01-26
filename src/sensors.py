import math

class ProximitySensor:

    def __init__(self, agent, environment, angle_offset, detection_range):
        """
        Creates a Proximity sensor
        :param agent: the agent which the sensor is attached to
        :param environment: the environment object
        :param angle_offset: the angle offset of the sensor from the front of the agent
        :param detection_range: the detection range of the sensor
        """
        self.angle_offset = angle_offset
        self.detection_range = detection_range
        self.env = environment
        self.agent = agent

    def old_detect(self, agent, agents, environment):
        """
        Detects if the sensor detects presence of another object
        :param environment: environment to check
        :param agent: agent to check
        :param agents: other agents
        :return: value between 0 and 1 indicating the distance to the detected object
        """
        sensor_bearing = agent.bearing + self.angle_offset
        sensor_end_x = agent.x + (self.detection_range * math.cos(sensor_bearing))
        sensor_end_y = agent.y + (self.detection_range * math.sin(sensor_bearing))

        for other_agent in agents:
            if other_agent is not agent and self.line_circle_collision(agent.x, agent.y, sensor_end_x, sensor_end_y,
                                                                       other_agent.x, other_agent.y,
                                                                       agent.AGENT_RADIUS):
                dist_to_agent = ((agent.x - other_agent.x) ** 2 + (agent.y - other_agent.y) ** 2) ** 0.5
                x = (self.detection_range ** 2 + self.detection_range ** 2) ** 0.5
                if dist_to_agent > x:
                    return 0
                else:
                    return dist_to_agent / x

        for obstacle in environment.obstacles:
            if line_rect_collision(agent.x, agent.y, sensor_end_x, sensor_end_y, obstacle):
                return 1.0  # Detected an obstacle

        return 0.0

    def detect(self):
        """
        Detects presence of other objects
        :return: value between 0 and 1 indicating the distance to the detected object
        """
        agent_x, agent_y = self.agent.x, self.agent.y
        agent_bearing = self.agent.bearing
        sensor_bearing = angle_below_360(agent_bearing + self.angle_offset)
        i = self.detection_range
        while i >= 0:
            px = agent_x + i * math.cos(math.radians(sensor_bearing))
            py = agent_y + i * math.sin(math.radians(sensor_bearing))
            if self.env.get_cell_val(px, py) == 1:
                return i / self.detection_range
            i -= 1
        return 0



class LidarSensor:

    def __init__(self, detection_range, num_rays):
        self.detection_range = detection_range
        self.num_rays = num_rays

    def detect(self, agent, agents, environment):
        """
        Simulates Lidar readings
        :param agent:
        :param agents:
        :param environment:
        :return:
        """
        data = []
        x1, y1 = agent.x, agent.y
        # for i in range(self.num_rays):
        # TODO: sort this


def line_circle_collision(x1, y1, x2, y2, cx, cy, cr):
    # Check if any point on the line segment (x1,y1) to (x2,y2) is inside the circle with center (cx,
    # cy) and radius cr.
    line_len = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
    dot = (((cx - x1) * (x2 - x1)) + ((cy - y1) * (y2 - y1))) / line_len ** 2
    closest_x = x1 + (dot * (x2 - x1))
    closest_y = y1 + (dot * (y2 - y1))
    distance = ((closest_x - cx) ** 2 + (closest_y - cy) ** 2) ** 0.5
    return distance < cr

def line_rect_collision(x1, y1, x2, y2, rect):
    # Define the four sides of the rectangle
    left = rect.left
    right = rect.right
    top = rect.top
    bottom = rect.bottom

    # Check each side of the rectangle for intersection with the line segment
    if line_line_collision(x1, y1, x2, y2, left, top, left, bottom):  # Left side
        return True
    if line_line_collision(x1, y1, x2, y2, right, top, right, bottom):  # Right side
        return True
    if line_line_collision(x1, y1, x2, y2, left, top, right, top):  # Top side
        return True
    if line_line_collision(x1, y1, x2, y2, left, bottom, right, bottom):  # Bottom side
        return True
    return False


def line_line_collision(x1, y1, x2, y2, x3, y3, x4, y4):
    # Determine the direction of the lines
    uA = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / ((y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1))
    uB = ((x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)) / ((y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1))

    # If uA and uB are between 0-1, lines are colliding
    if 0 <= uA <= 1 and 0 <= uB <= 1:
        return True

    return False

def angle_below_360(angle):
    while angle > 360:
        angle -= 360
    return angle


if __name__ == "__main__":
    print(angle_below_360(730))