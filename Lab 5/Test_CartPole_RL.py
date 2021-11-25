import unittest
from CartPole_RL import CartPoleModel, run_experiment


class TestCartPole(unittest.TestCase):

    def setUp(self) -> None:
        self.cartPole_model = CartPoleModel()

    def tearDown(self) -> None:
        pass

    def test_something(self):
        self.assertEqual(True, False)  # add assertion here


if __name__ == '__main__':
    unittest.main()
