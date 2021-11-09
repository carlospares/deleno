import unittest
import numpy as np
import observables as obs

import matplotlib.pyplot as plt

# ALGO = 0
# NAME = 1

class TestObservable(unittest.TestCase):
    def test_identity(self):
        input_data = 3
        np.testing.assert_almost_equal(input_data, obs.identity(input_data))
        input_data = np.array([0,1])
        np.testing.assert_almost_equal(input_data, obs.identity(input_data))
        input_data = np.array([ [[0,1],[2,3]] ])
        np.testing.assert_almost_equal(input_data, obs.identity(input_data))
        input_data = np.array([ [[0,1],[2,3]], [[4,5],[6,7]] ])
        np.testing.assert_almost_equal(input_data, obs.identity(input_data))

    def test_subdomain(self):
        input_data = np.ones([1,16,16])
        np.testing.assert_almost_equal(obs.mass_subdomain(input_data), 1./16)

if __name__ == '__main__':
    unittest.main()