import numpy as np
import pygame
import sys
import cv2
from agent import Agent
from environment import Environment
from sensors import LidarSensor
from ui import EnvironmentInterface
from features import *
from ekf import EKF


def main():
	# Initialize environment
	env_width = 1280
	env_height = 720
	map_path = "map4.png"

	"""
    map2.png = [(570, 360), (710, 360), (640, 290), (640, 430)]
    map3.png = [(370, 360), (910, 360), (640, 90), (640, 630)]
    map4.png = [(640, 280), (560, 360), (720, 360), (640, 440), (320, 75), (960, 645), (960, 75), (320, 645), (75, 360), (1205, 360)]
    
    map2 lines = [[(570, 410), (570, 310)], [(710, 410), (710, 310)], [(690, 290), (590, 290)], [(690, 430), (590, 430)]]
    map4 lines = [[(640, 205), (640, 355)], [(485, 360), (575, 360)], [(645, 360), (795, 360)], [(640, 365), (640, 515)], [(320, 0), (320, 150)], [(960, 570), (960, 720)], [(960, 0), (960, 150)], [(320, 570), (320, 720)], [(0, 360), (150, 360)], [(1130, 360), (1280, 360)]]
    """

	landmarks_pixels = [(640, 280), (560, 360), (720, 360), (640, 440), (320, 75), (960, 645), (960, 75), (320, 645), (75, 360), (1205, 360)]
	landmarks_lines = [[(640, 205), (640, 355)], [(485, 360), (575, 360)], [(645, 360), (795, 360)], [(640, 365), (640, 515)], [(320, 0), (320, 150)], [(960, 570), (960, 720)], [(960, 0), (960, 150)], [(320, 570), (320, 720)], [(0, 360), (150, 360)], [(1130, 360), (1280, 360)]]
	landmarks = [(l[0] * 0.02, l[1] * 0.02) for l in landmarks_pixels]

	map = cv2.imread(map_path)
	environment = Environment.img_to_env(map)
	interface = EnvironmentInterface(environment, map_path)
	init_pos = np.array([1., 8., 0.])
	agent = Agent(environment, landmarks, radius=10, init_pos=init_pos)
	clock = pygame.time.Clock()
	fps_limit = 30
	ekf = EKF(landmarks)
	fd = feature_dectection()
	endpoints = [0., 0.]

	dt = 0.1
	u = [0.25, 0.]

	positions = ""

	show_landmarks(interface.get_screen(), landmarks_pixels)

	running = True
	while running:
		clock.tick(fps_limit)
		# wait for 50 ms
		pygame.time.wait(50)
		# show_landmarks(interface.get_screen(), landmarks_pixels)

		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				running = False

		u = agent.update_u(u)
		agent.move(u, dt)

		display_objects = []

		# results = agent.detect()
		# if results is not False and results is not None:
		# 	outmost_points = results[2]
		# 	m, c = results[5]
		#
		# 	endpoints[0] = fd.projection_point_to_line(outmost_points[0], m, c)
		# 	endpoints[1] = fd.projection_point_to_line(outmost_points[1], m, c)
		# 	# print(f"line =[{m}, {c}, {endpoints}]")
		# 	display_objects.append(pygame.draw.line(interface.get_screen(), (0, 150, 150), endpoints[0], endpoints[1], 2))

		measurements = agent.simple_detect(agent.get_state(), landmarks)

		# try:
		# 	endpoints_m = [[endpoints[0][0] * 0.02, endpoints[0][1] * 0.02], [endpoints[1][0] * 0.02, endpoints[1][1] * 0.02]]
		# 	# measurements = agent.sim_measurements(agent.get_state(), landmarks)
		# 	measurements = fd.landmark_association(endpoints_m[0], endpoints_m[1], landmarks_lines, agent.get_state())
		# except:
		# 	measurements = []

		# ekf logic
		agent.mu, agent.sigma = ekf.prediction_update(agent.mu, agent.sigma, u, dt)
		agent.mu, agent.sigma = ekf.measurement_update(agent.mu, agent.sigma, measurements)

		# if pygame.time.get_ticks() % 10 == 0:
		# 	eigenvals, angle = agent.ekf.sigma2transform(agent.sigma[0:2, 0:2])
		# 	uncertainty = (eigenvals[0] + eigenvals[1]) / 2
		# 	state = agent.get_state()
		# 	pos = (state[0] // 0.02, state[1] // 0.02)
		# 	print(f"pos = {pos} | mu = {agent.mu[0]} | sigma = {uncertainty} | time = {pygame.time.get_ticks()}")
			# print(f"pos = {pos}")
			# positions += f"{pos}, "
		interface.get_screen().fill((255, 255, 255))
		interface.draw()



		display_objects += agent.draw_agent(interface.get_screen())
		display_objects += agent.show_agent_estimate(interface.get_screen(), agent.mu, agent.sigma)
		display_objects += agent.show_landmark_uncertainty(agent.mu, agent.sigma, interface.get_screen())

		# pygame.display.update(display_objects)
		pygame.display.flip()

	print(positions)
	pygame.quit()


def data_to_pointcloud(positions):
	"""
    Adds data to pointcloud
    :param positions: positions of detected objects
    :return:
    """
	global pointcloud
	for pos in positions:
		if pos not in pointcloud:
			pointcloud.append(pos)


def show_pointcloud(screen):
	"""
    Shows pointcloud
    :return:
    """
	global pointcloud
	for pos in pointcloud:
		pygame.draw.circle(screen, (0, 255, 0), (int(pos[0]), int(pos[1])), 2)


def show_landmarks(screen, landmarks):
	for pos in landmarks:
		pygame.draw.circle(screen, (0, 255, 0), (int(pos[0]), int(pos[1])), 2)
		# pygame.gfxdraw.circle(screen, int(pos[0]), int(pos[1]), 10, (0, 255, 0))

if __name__ == "__main__":
	main()
