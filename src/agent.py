import pygame
import random
import math

# Constants
AGENT_RADIUS = 10
AGENT_COLOR = (0, 0, 255)
AGENT_SPEED = 2


class Agent:
    def __init__(self, screen_width, screen_height, x=None, y=None, sensor_range=50, show_sensors=True):
        if x:
            self.x = x
        else:
            self.x = random.randint(AGENT_RADIUS, screen_width - AGENT_RADIUS)
        if y:
            self.y = y
        else:
            self.y = random.randint(AGENT_RADIUS, screen_height - AGENT_RADIUS)
        self.screen_width = screen_width
        self.screen_height = screen_height

        # Random initial direction and velocity
        self.bearing = random.uniform(0, 2 * math.pi)
        self.velocity = AGENT_SPEED
        # 8 Sensors evenly spaced around the agent
        self.sensors = [ProximitySensor(0, sensor_range),
                        ProximitySensor(math.pi / 4, sensor_range),
                        ProximitySensor(math.pi / 2, sensor_range),
                        ProximitySensor((math.pi * 3) / 4, sensor_range),
                        ProximitySensor(math.pi, sensor_range),
                        ProximitySensor((-math.pi * 3) / 4, sensor_range),
                        ProximitySensor(-math.pi / 2, sensor_range),
                        ProximitySensor(-math.pi / 4, sensor_range)]
        self.show_sensors = show_sensors

    def update(self, agents):
        # Move the agent
        self.x += self.velocity * math.cos(self.bearing)
        self.y += self.velocity * math.sin(self.bearing)

    def draw(self, screen):
        pygame.draw.circle(screen, AGENT_COLOR, (self.x, self.y), AGENT_RADIUS)

        # Draw sensors
        if self.show_sensors:
            for sensor in self.sensors:
                sensor_bearing = self.bearing + sensor.angle_offset
                sensor_end_x = self.x + (sensor.detection_range * math.cos(sensor_bearing))
                sensor_end_y = self.y + (sensor.detection_range * math.sin(sensor_bearing))
                pygame.draw.line(screen, (255, 0, 0), (self.x, self.y), (sensor_end_x, sensor_end_y))

    def collide_with(self, other_agent):
        distance = ((self.x - other_agent.x) ** 2 + (self.y - other_agent.y) ** 2) ** 0.5
        return distance < 2 * AGENT_RADIUS

    def handle_agent_collisions(self, agents, environment):
        left_sensor_detected = self.sensors[-2].detect(self, agents, environment)
        right_sensor_detected = self.sensors[1].detect(self, agents, environment)
        # if left_sensor_detected > 0:
        #     print("Left sensor detected: {}".format(left_sensor_detected))
        # if right_sensor_detected > 0:
        #     print("Right sensor detected: {}".format(right_sensor_detected))

        if left_sensor_detected > 0 and right_sensor_detected > 0 and self.velocity > 0.1:
            self.velocity -= 0.1
        elif left_sensor_detected > 0:
            # Turn right
            self.bearing = self.bearing + left_sensor_detected * math.pi / 4
            self.velocity -= 0.1
        elif right_sensor_detected > 0:
            # Turn left
            self.bearing = self.bearing - right_sensor_detected * math.pi / 4
            self.velocity -= 0.1
        else:
            if self.velocity < AGENT_SPEED:
                self.velocity += 0.1

    def handle_environment_collisions(self, environment):
        # Check if agent collides with any obstacle
        for obstacle in environment.obstacles:
            if pygame.Rect(obstacle).colliderect(
                    pygame.Rect(self.x - AGENT_RADIUS, self.y - AGENT_RADIUS, 2 * AGENT_RADIUS, 2 * AGENT_RADIUS)):
                # Handle collision, e.g., change direction
                self.bearing = random.uniform(0, 2 * math.pi)
        # Check if agent collides with environment boundary
        # TODO: implement this
        """ old code
        if self.x - AGENT_RADIUS < 0 or self.x + AGENT_RADIUS > self.screen_width:
            self.bearing = math.pi - self.bearing
        if self.y - AGENT_RADIUS < 0 or self.y + AGENT_RADIUS > self.screen_height:
            self.bearing = -self.bearing """


        if self.x - AGENT_RADIUS < 0 or self.x + AGENT_RADIUS > self.screen_width:
            self.bearing = math.pi - self.bearing
        if self.y - AGENT_RADIUS < 0 or self.y + AGENT_RADIUS > self.screen_height:
            self.bearing = -self.bearing


class ProximitySensor:
    def __init__(self, angle_offset, detection_range):
        self.angle_offset = angle_offset
        self.detection_range = detection_range

    def detect(self, agent, agents, environment):
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
                                                                       other_agent.x, other_agent.y, AGENT_RADIUS):
                dist_to_agent = ((agent.x - other_agent.x) ** 2 + (agent.y - other_agent.y) ** 2) ** 0.5
                x = (self.detection_range ** 2 + self.detection_range ** 2) ** 0.5
                if dist_to_agent > x:
                    return 0
                else:
                    return dist_to_agent / x

        for obstacle in environment.obstacles:
            if self.line_rect_collision(agent.x, agent.y, sensor_end_x, sensor_end_y, obstacle):
                return 1.0  # Detected an obstacle

        return 0.0

    @staticmethod
    def line_circle_collision(x1, y1, x2, y2, cx, cy, cr):
        # Check if any point on the line segment (x1,y1) to (x2,y2) is inside the circle with center (cx,
        # cy) and radius cr.
        line_len = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
        dot = (((cx - x1) * (x2 - x1)) + ((cy - y1) * (y2 - y1))) / line_len ** 2
        closest_x = x1 + (dot * (x2 - x1))
        closest_y = y1 + (dot * (y2 - y1))
        distance = ((closest_x - cx) ** 2 + (closest_y - cy) ** 2) ** 0.5
        return distance < cr

    @staticmethod
    def line_rect_collision(x1, y1, x2, y2, rect):
        # Define the four sides of the rectangle
        left = rect.left
        right = rect.right
        top = rect.top
        bottom = rect.bottom

        # Check each side of the rectangle for intersection with the line segment
        if ProximitySensor.line_line_collision(x1, y1, x2, y2, left, top, left, bottom):  # Left side
            return True
        if ProximitySensor.line_line_collision(x1, y1, x2, y2, right, top, right, bottom):  # Right side
            return True
        if ProximitySensor.line_line_collision(x1, y1, x2, y2, left, top, right, top):  # Top side
            return True
        if ProximitySensor.line_line_collision(x1, y1, x2, y2, left, bottom, right, bottom):  # Bottom side
            return True

        return False

    @staticmethod
    def line_line_collision(x1, y1, x2, y2, x3, y3, x4, y4):
        # Determine the direction of the lines
        uA = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / ((y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1))
        uB = ((x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)) / ((y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1))

        # If uA and uB are between 0-1, lines are colliding
        if 0 <= uA <= 1 and 0 <= uB <= 1:
            return True

        return False
