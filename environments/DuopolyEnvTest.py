import sys
import unittest

sys.path.append("..")
from environments.DuopolyEnv import DuopolyEnv

## HAVING ISSUES WITH SYSPATH FOR TEST
## THESE WILL BE MOVED TO test FOLDER


class TestDuopolyEnvConfigs(unittest.TestCase):
    def setUp(self) -> None:
        self.NUM_CUSTOMER = 500
        self.NUM_SELLER = 5
        self.MAX_CAP = 5000
        self.MIN_PRICE = 20
        self.MAX_PRICE = 50
        self.MEMORY_SIZE = 10

        return super().setUp()

    def tearDown(self) -> None:
        return super().tearDown()

    def _make_env(self):
        self.env = DuopolyEnv(self.env_config)

    def test_states_sanity(self):
        self.env_config = {
            "num_customer": self.NUM_CUSTOMER,
            "memory_size": self.MEMORY_SIZE,
            "num_seller": self.NUM_SELLER,
        }
        self._make_env()
        states = self.env.reset()

        self.assertEqual(states.shape[0], self.MEMORY_SIZE)
        # We also take the number of previously sold items
        self.assertEqual(states.shape[1], self.NUM_SELLER + 1)

    def test_num_customer(self):
        self.env_config = {"num_customer": self.NUM_CUSTOMER}
        self._make_env()
        _ = self.env.reset()
        self.assertEqual(self.env.num_customer, self.NUM_CUSTOMER)

    def test_max_price(self):
        self.env_config = {"max_price": self.MAX_PRICE}
        self._make_env()
        _ = self.env.reset()
        self.assertEqual(self.env.max_price, self.MAX_PRICE)

    def test_min_price(self):
        self.env_config = {"min_price": self.MIN_PRICE}
        self._make_env()
        _ = self.env.reset()
        self.assertEqual(self.env.min_price, self.MIN_PRICE)

    def test_num_seller(self):
        self.env_config = {"num_seller": self.NUM_SELLER}
        self._make_env()
        _ = self.env.reset()
        self.assertEqual(self.env.num_seller, self.NUM_SELLER)

    def test_max_capacity(self):
        self.env_config = {"max_capacity": self.MAX_CAP}
        self._make_env()
        _ = self.env.reset()
        self.assertEqual(self.env.max_capacity, self.MAX_CAP)


if __name__ == "__main__":
    unittest.main()
