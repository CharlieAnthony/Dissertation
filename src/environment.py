import cv2


class Environment:
    def __init__(self, width, height, env=None):
        self.width = width
        self.height = height
        if env is None:
            self.env = [[0] * width for _ in range(height)]
        else:
            self.env = env

    def get_cell_val(self, x, y):
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return 1
        return self.env[int(y)][int(x)]

    @staticmethod
    def img_to_env(img):
        """
        Converts an image to an environment
        :param img: cv2 image
        :return: Environment object
        """
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gray = gray / 255
        gray[gray > 0.5] = 1
        gray[gray <= 0.5] = 0
        gray = 1 - gray
        height, width = gray.shape
        env = Environment(width, height, env=gray)
        return env


if __name__ == "__main__":
    e = Environment(10, 10)
    print(e.env)
