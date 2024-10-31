import unittest

from rankwise.classify.calculations import is_best


class TestClassify(unittest.TestCase):
    def test_is_best(self):
        self.assertTrue(is_best(lambda x, y: abs(x - y), lambda x, y: x < y, {1, 2, 3}, 1, 1))
        self.assertFalse(is_best(lambda x, y: abs(x - y), lambda x, y: x < y, {1, 2, 3}, 1, 2))
        self.assertFalse(is_best(lambda x, y: abs(x - y), lambda x, y: x < y, {1, 2, 3}, 1, 3))
        self.assertTrue(is_best(lambda x, y: abs(x - y), lambda x, y: x < y, {1, 2, 3}, 2, 2))
        self.assertFalse(is_best(lambda x, y: abs(x - y), lambda x, y: x < y, {1, 2, 3}, 2, 1))
        self.assertFalse(is_best(lambda x, y: abs(x - y), lambda x, y: x < y, {1, 2, 3}, 2, 3))
        self.assertTrue(is_best(lambda x, y: abs(x - y), lambda x, y: x < y, {1, 2, 3}, 3, 3))
        self.assertFalse(is_best(lambda x, y: abs(x - y), lambda x, y: x < y, {1, 2, 3}, 3, 1))
        self.assertFalse(is_best(lambda x, y: abs(x - y), lambda x, y: x < y, {1, 2, 3}, 3, 2))
        self.assertTrue(is_best(lambda x, y: abs(x - y), lambda x, y: x > y, {1, 2, 3}, 3, 1))


if __name__ == "__main__":
    unittest.main()
