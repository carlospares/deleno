import unittest
from boundary_condition import BoundaryCondition
import numpy as np

class TestPeriodicBC(unittest.TestCase):
    def test_1d_bc_extend(self):
        """
        Test 1d BCs (apply to ghost cells)
        """
        BC = BoundaryCondition(["nonsense","nonsense","E","E"])
        data = np.zeros(11)
        data[2:9] = [1,2,3,4,5,6,7]
        BC.apply_bc_1d(data, 2, BC.AXIS_EW)
        np.testing.assert_array_equal(data, [1,1,1,2,3,4,5,6,7,7,7])

    def test_2d_bc_extend(self):
        """
        Test 2d BCs (apply to ghost cells)
        """
        BC = BoundaryCondition(["E","E","E","E"])
        A = np.array([[1.,2.], [3.,4.]])
        data = np.zeros((1,6,6))
        data[0, 2:4, 2:4] = A
        BC.apply_bc_2d(data, 2)
        ref1 = [[1.,2.],[1.,2.],[1.,2.],[3.,4.],[3.,4.],[3.,4.]]
        ref2 = [[1.,1.,1.,2.,2.,2.],[3.,3.,3.,4.,4.,4.]]
        np.testing.assert_array_equal(data[0, :, 2:4], ref1)
        np.testing.assert_array_equal(data[0, 2:4, :], ref2)

    def test_1d_bc_extend_extend(self):
        """
        Test 1d BCs (add ghost cells)
        """
        BC = BoundaryCondition(["nonsense","nonsense","E","E"])
        data = np.array([1,2,3,4,5,6,7])
        out = BC.extend_with_bc_1d(data, 3, BC.AXIS_EW)
        np.testing.assert_array_equal(out, [1,1,1,1,2,3,4,5,6,7,7,7,7])

    def test_2d_bc_extend_extend(self):
        """
        Test 2d BCs (add ghost cells)
        """
        BC = BoundaryCondition(["E","E","E","E"])
        A = np.array([[[1.,2.], [3.,4.]]])
        data = BC.extend_with_bc_2d(A, 2)
        ref1 = [[1.,2.],[1.,2.],[1.,2.],[3.,4.],[3.,4.],[3.,4.]]
        ref2 = [[1.,1.,1.,2.,2.,2.],[3.,3.,3.,4.,4.,4.]]
        np.testing.assert_array_equal(data[0, :, 2:4], ref1)
        np.testing.assert_array_equal(data[0, 2:4, :], ref2)

if __name__ == '__main__':
    unittest.main()