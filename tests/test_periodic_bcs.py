import unittest
from boundary_condition import BoundaryCondition
import numpy as np

class TestPeriodicBC(unittest.TestCase):
    def test_1d_bc_periodic(self):
        """
        Test 1d BCs (apply to ghost cells)
        """
        BC = BoundaryCondition(["nonsense","nonsense","P","P"])
        data = np.zeros(11)
        data[2:9] = [1,2,3,4,5,6,7]
        BC.apply_bc_1d(data, 2, BC.AXIS_EW)
        np.testing.assert_array_equal(data, [6,7,1,2,3,4,5,6,7,1,2])

    def test_2d_bc_periodic(self):
        """
        Test 2d BCs (apply to ghost cells)
        """
        BC = BoundaryCondition(["P","P","P","P"])
        A = np.array([[1.,2.], [3.,4.]])
        data = np.zeros((6,6))
        data[2:4, 2:4] = A
        BC.apply_bc_2d(data, 2)
        ref = np.kron(np.ones((3,3)),A)
        np.testing.assert_array_equal(data, ref)

    def test_1d_bc_periodic_extend(self):
        """
        Test 1d BCs (add ghost cells)
        """
        BC = BoundaryCondition(["nonsense","nonsense","P","P"])
        data = np.array([1,2,3,4,5,6,7])
        out = BC.extend_with_bc_1d(data, 3, BC.AXIS_EW)
        np.testing.assert_array_equal(out, [5,6,7,1,2,3,4,5,6,7,1,2,3])

    def test_2d_bc_periodic_extend(self):
        """
        Test 2d BCs (add ghost cells)
        """
        BC = BoundaryCondition(["P","P","P","P"])
        A = np.array([[1.,2.], [3.,4.]])
        data = BC.extend_with_bc_2d(A, 2)
        ref = np.kron(np.ones((3,3)),A)
        np.testing.assert_array_equal(data, ref)

if __name__ == '__main__':
    unittest.main()