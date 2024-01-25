import unittest
from unittest.mock import Mock
from src.sensors import ProximitySensor


class ProximitySensorTest(unittest.TestCase):
    def setUp(self):
        self.env = Mock()
        self.agent = Mock()
        self.agent.x = 0
        self.agent.y = 0
        self.agent.bearing = 0
        self.sensor = ProximitySensor(self.agent, self.env, 0, 10)

    def test_proximity_sensor_detects_nothing(self):
        # detects nothing when nothing is there
        self.env.get_cell_val.return_value = 0
        self.assertEqual(self.sensor.detect(), 0)

    def test_proximity_sensor_detects_obstacle(self):
        # detects obstacle when obstacle is there
        self.env.get_cell_val.return_value = 1
        self.assertEqual(self.sensor.detect(), 1)

    def test_proximity_sensor_half_range(self):
        # detects obstacle when obstacle is at half detection distance
        self.env.get_cell_val.return_value = 0
        self.assertEqual(self.sensor.detect(), 0.5)

if __name__ == '__main__':
    unittest.main()
