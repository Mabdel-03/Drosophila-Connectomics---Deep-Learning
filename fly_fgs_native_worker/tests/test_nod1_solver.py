import numpy as np
import pytest

from fly_sensor2behavior.nod1_solver import dense_tree_matrix, hines_solve


def test_hines_matches_dense_on_arbitrarily_ordered_tree():
    parent = np.array([2, 4, -1, 2, 2, 4], dtype=int)
    coupling = np.array([0.8, 0.3, 0.0, 0.7, 0.5, 0.4], dtype=float)
    # Strict diagonal dominance gives a well-conditioned passive system.
    diagonal = np.array([1.5, 0.9, 3.0, 1.4, 2.0, 1.0], dtype=float)
    rhs = np.array([-0.05, 0.02, -0.12, 0.03, -0.08, 0.04], dtype=float)
    dense = np.linalg.solve(dense_tree_matrix(parent, diagonal, coupling), rhs)
    hines = hines_solve(parent, diagonal, coupling, rhs)
    np.testing.assert_allclose(hines, dense, rtol=1e-12, atol=1e-14)


def test_hines_rejects_invalid_or_disconnected_parent_graphs():
    with pytest.raises(ValueError, match="exactly one root"):
        hines_solve([-1, -1], [1.0, 1.0], [0.0, 0.0], [0.0, 0.0])
    with pytest.raises(ValueError, match="cycle|disconnected"):
        hines_solve([-1, 2, 1], [2.0, 2.0, 2.0], [0.0, 0.5, 0.5], [0.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="root coupling"):
        hines_solve([-1], [1.0], [0.2], [0.0])


def test_hines_rejects_nonpositive_pivots():
    with pytest.raises(np.linalg.LinAlgError, match="pivot"):
        hines_solve([-1, 0], [0.5, 0.5], [0.0, 0.5], [0.0, 0.0])
