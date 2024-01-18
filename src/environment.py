# environment.py
import pygame
import cv2

class Environment:
    def __init__(self, width, height, env=None):
        self.width = width
        self.height = height
        if not env:
            self.env = [[0] * width for _ in range(height)]
        else:
            self.env = env

    def get_cell_val(self, x, y):
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return 1
        return self.env[y][x]

@staticmethod
def Img_to_env(img):
    # TODO: take image and create environment
    pass


if __name__ == "__main__":
    e = Environment(10, 10)
    print(e.env)