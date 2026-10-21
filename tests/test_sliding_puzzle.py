"""Testes do quebra-cabeça deslizante."""

from dataclasses import FrozenInstanceError

import pytest

from search_lib import BFS

from examples.sliding_puzzle.problem import (
    SlidingPuzzle,
    SlidingPuzzleAction,
    SlidingPuzzleState,
)


@pytest.mark.parametrize(("rows", "cols"), [(3, 3), (4, 4), (4, 3)])
def test_goal_and_state_shapes(rows: int, cols: int) -> None:
    problem = SlidingPuzzle(rows, cols)

    assert len(problem.initial_state.tiles) == rows * cols
    assert problem.is_goal(problem.initial_state)
    assert problem.heuristic(problem.initial_state) == 0.0


@pytest.mark.parametrize("heuristic", ["manhattan", "misplaced"])
def test_both_heuristics_are_zero_at_goal(heuristic: str) -> None:
    problem = SlidingPuzzle(heuristic=heuristic)

    assert problem.heuristic(problem.initial_state) == 0.0


def test_invalid_move_and_immutability() -> None:
    problem = SlidingPuzzle()
    state = problem.initial_state

    result = SlidingPuzzleAction("BAIXO", 3, 3).apply(state)

    assert result is None
    with pytest.raises(FrozenInstanceError):
        state.tiles = (0, 1, 2, 3, 4, 5, 6, 7, 8)  # type: ignore[misc]


@pytest.mark.parametrize(("rows", "cols"), [(3, 3), (4, 4), (4, 3)])
def test_random_walks_are_deterministic_and_solvable(rows: int, cols: int) -> None:
    problem = SlidingPuzzle(rows, cols)
    first = problem.random_state(4, seed=29)
    second = problem.random_state(4, seed=29)

    assert first == second
    assert sorted(first.tiles) == list(range(rows * cols))
    assert not problem.is_goal(first)


@pytest.mark.parametrize("heuristic", ["manhattan", "misplaced"])
def test_heuristics_are_admissible_for_short_scrambles(heuristic: str) -> None:
    problem = SlidingPuzzle(heuristic=heuristic)
    for seed in (4, 11, 28):
        state = problem.random_state(4, seed=seed)
        result = BFS(problem).search(state)

        assert result.solution_node is not None
        assert problem.heuristic(state) <= result.solution_node.cost


@pytest.mark.parametrize(("rows", "cols", "moves"), [(3, 3, 8), (4, 4, 6), (4, 3, 6)])
def test_bfs_solves_short_scrambles(
    rows: int, cols: int, moves: int
) -> None:
    problem = SlidingPuzzle(rows, cols)
    state = problem.random_state(moves, seed=17)
    result = BFS(problem).search(state)

    assert result.solution_node is not None
    assert result.solution_node.cost <= moves
