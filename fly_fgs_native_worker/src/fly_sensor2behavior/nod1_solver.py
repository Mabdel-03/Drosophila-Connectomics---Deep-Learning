"""Dependency-light Hines tree solver used by the authoritative NOD1 worker.

The matrix is represented by a positive diagonal and one symmetric negative
off-diagonal coupling per non-root node.  The implementation accepts arbitrary
node ordering and derives deterministic preorder/postorder traversals.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np


def _tree_orders(parent: np.ndarray) -> Tuple[Tuple[int, ...], Tuple[int, ...], int]:
    count = int(parent.size)
    roots = np.flatnonzero(parent < 0)
    if roots.size != 1:
        raise ValueError("Hines tree must contain exactly one root")
    root = int(roots[0])
    children: List[List[int]] = [[] for _ in range(count)]
    for node, parent_node in enumerate(parent):
        if node == root:
            continue
        if parent_node < 0 or parent_node >= count or parent_node == node:
            raise ValueError("invalid Hines parent index")
        children[int(parent_node)].append(node)
    preorder: List[int] = []
    postorder: List[int] = []
    active = set()

    def visit(node: int) -> None:
        if node in active:
            raise ValueError("Hines parent graph contains a cycle")
        active.add(node)
        preorder.append(node)
        for child in sorted(children[node]):
            visit(child)
        active.remove(node)
        postorder.append(node)

    visit(root)
    if len(preorder) != count:
        raise ValueError("Hines parent graph is disconnected")
    return tuple(preorder), tuple(postorder), root


def hines_solve(
    parent: Sequence[int],
    diagonal: Sequence[float],
    parent_coupling: Sequence[float],
    rhs: Sequence[float],
) -> np.ndarray:
    """Solve one symmetric tree linear system by Hines elimination.

    ``parent_coupling[i]`` is the positive conductance whose matrix entries at
    ``(i,parent[i])`` and ``(parent[i],i)`` are ``-conductance``.  Root coupling
    must be zero.  ``diagonal`` is the complete original matrix diagonal.
    """

    parent_array = np.asarray(parent, dtype=int)
    diagonal_array = np.asarray(diagonal, dtype=float)
    coupling_array = np.asarray(parent_coupling, dtype=float)
    rhs_array = np.asarray(rhs, dtype=float)
    if parent_array.ndim != 1 or parent_array.size == 0:
        raise ValueError("Hines arrays must be non-empty vectors")
    count = parent_array.size
    if any(array.shape != (count,) for array in (diagonal_array, coupling_array, rhs_array)):
        raise ValueError("Hines arrays must have equal vector shapes")
    if not np.all(np.isfinite(diagonal_array)) or not np.all(np.isfinite(coupling_array)):
        raise ValueError("Hines matrix entries must be finite")
    if not np.all(np.isfinite(rhs_array)):
        raise ValueError("Hines rhs must be finite")
    if np.any(diagonal_array <= 0.0) or np.any(coupling_array < 0.0):
        raise ValueError("Hines diagonal must be positive and coupling non-negative")
    preorder, postorder, root = _tree_orders(parent_array)
    if coupling_array[root] != 0.0:
        raise ValueError("Hines root coupling must be zero")

    reduced_diagonal = diagonal_array.copy()
    reduced_rhs = rhs_array.copy()
    for node in postorder:
        if node == root:
            continue
        pivot = reduced_diagonal[node]
        if not np.isfinite(pivot) or pivot <= 0.0:
            raise np.linalg.LinAlgError("Hines elimination encountered a non-positive pivot")
        parent_node = int(parent_array[node])
        coupling = coupling_array[node]
        reduced_diagonal[parent_node] -= coupling * coupling / pivot
        reduced_rhs[parent_node] += coupling * reduced_rhs[node] / pivot

    if reduced_diagonal[root] <= 0.0:
        raise np.linalg.LinAlgError("Hines root pivot is non-positive")
    solution = np.empty(count, dtype=float)
    solution[root] = reduced_rhs[root] / reduced_diagonal[root]
    for node in preorder:
        if node == root:
            continue
        parent_node = int(parent_array[node])
        solution[node] = (
            reduced_rhs[node] + coupling_array[node] * solution[parent_node]
        ) / reduced_diagonal[node]
    return solution


def dense_tree_matrix(
    parent: Sequence[int],
    diagonal: Sequence[float],
    parent_coupling: Sequence[float],
) -> np.ndarray:
    """Construct the equivalent dense matrix for manufactured tests only."""

    parent_array = np.asarray(parent, dtype=int)
    diagonal_array = np.asarray(diagonal, dtype=float)
    coupling_array = np.asarray(parent_coupling, dtype=float)
    _preorder, _postorder, root = _tree_orders(parent_array)
    count = parent_array.size
    if diagonal_array.shape != (count,) or coupling_array.shape != (count,):
        raise ValueError("tree arrays must have equal vector shapes")
    matrix = np.diag(diagonal_array)
    for node in range(count):
        if node == root:
            continue
        parent_node = int(parent_array[node])
        matrix[node, parent_node] = -coupling_array[node]
        matrix[parent_node, node] = -coupling_array[node]
    return matrix


__all__ = ["dense_tree_matrix", "hines_solve"]
