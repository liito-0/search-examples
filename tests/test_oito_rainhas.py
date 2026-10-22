"""Testes do problema das oito rainhas e das buscas em espaço finito."""

import pytest

from search_lib import BFS, DFS, DLS
from examples.oito_rainhas.problem import BOARD_SIZE, EightQueens, PlaceQueen, QueensState


def assert_valid_solution(problem: EightQueens, state: QueensState) -> None:
    assert len(state.columns) == BOARD_SIZE
    assert all(0 <= column < BOARD_SIZE for column in state.columns)
    for row, column in enumerate(state.columns):
        for other_row in range(row):
            other_column = state.columns[other_row]
            assert column != other_column
            assert abs(column - other_column) != row - other_row
    assert problem.is_goal(state)


def test_actions_reject_same_column_and_diagonal_attacks() -> None:
    assert PlaceQueen(0).apply(QueensState((0,))) is None
    assert PlaceQueen(1).apply(QueensState((0,))) is None
    assert PlaceQueen(2).apply(QueensState((0,))) == (QueensState((0, 2)), 1.0)


@pytest.mark.parametrize("strategy", [BFS, DFS, lambda p: DLS(p, 8)])
def test_basic_searches_find_valid_eight_queens_solution(strategy) -> None:
    problem = EightQueens()
    result = strategy(problem).search(problem.initial_state)

    assert result.solution_node is not None
    assert result.solution_node.depth == 8
    assert result.solution_node.cost == 8.0
    assert_valid_solution(problem, result.solution_node.state)


def test_bfs_finds_optimal_solution_and_depth_limit_is_complete_at_eight() -> None:
    problem = EightQueens()
    bfs_result = BFS(problem).search(problem.initial_state)
    dls_result = DLS(problem, 7).search(problem.initial_state)

    assert bfs_result.solution_node is not None
    assert bfs_result.solution_node.depth == 8  # mínimo: uma ação por linha
    assert dls_result.solution_node is None

