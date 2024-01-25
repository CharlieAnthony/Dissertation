import pygame
import random
import math

# Constants
AGENT_RADIUS = 10
AGENT_COLOR = (0, 0, 255)
AGENT_SPEED = 2


class Agent:
    def __init__(self, screen_width, screen_height, x=None, y=None, sensors=None, sensor_range=50, show_sensors=True):
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

        # initiate sensors
        if sensors:
            self.sensors = sensors
        else:
            self.sensors = []


    def update(self, agents):
        # Move the agent
        self.x += self.velocity * math.cos(self.bearing)
        self.y += self.velocity * math.sin(self.bearing)

    # now drawn in environment
    # def draw(self, screen):
    #     pygame.draw.circle(screen, AGENT_COLOR, (self.x, self.y), AGENT_RADIUS)
    #
    #     # Draw sensors
    #     if self.show_sensors:
    #         for sensor in self.sensors:
    #             sensor_bearing = self.bearing + sensor.angle_offset
    #             sensor_end_x = self.x + (sensor.detection_range * math.cos(sensor_bearing))
    #             sensor_end_y = self.y + (sensor.detection_range * math.sin(sensor_bearing))
    #             pygame.draw.line(screen, (255, 0, 0), (self.x, self.y), (sensor_end_x, sensor_end_y))

    def collide_with(self, other_agent):
        distance = ((self.x - other_agent.x) ** 2 + (self.y - other_agent.y) ** 2) ** 0.5
        return distance < 2 * AGENT_RADIUS

    # def handle_agent_collisions(self, environment):
        # left_sensor_detected = self.sensors[-2].detect(self, agents, environment)
        # right_sensor_detected = self.sensors[1].detect(self, agents, environment)
        # # if left_sensor_detected > 0:
        # #     print("Left sensor detected: {}".format(left_sensor_detected))
        # # if right_sensor_detected > 0:
        # #     print("Right sensor detected: {}".format(right_sensor_detected))
        #
        # if left_sensor_detected > 0 and right_sensor_detected > 0 and self.velocity > 0.1:
        #     self.velocity -= 0.1
        # elif left_sensor_detected > 0:
        #     # Turn right
        #     self.bearing = self.bearing + left_sensor_detected * math.pi / 4
        #     self.velocity -= 0.1
        # elif right_sensor_detected > 0:
        #     # Turn left
        #     self.bearing = self.bearing - right_sensor_detected * math.pi / 4
        #     self.velocity -= 0.1
        # else:
        #     if self.velocity < AGENT_SPEED:
        #         self.velocity += 0.1

        # For

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

    def add_sensor(self, sensor):
        self.sensors.append(sensor)
        return self

