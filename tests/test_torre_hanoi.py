"""Testes da Torre de Hanói."""

from collections import deque
from typing import Dict, List

import pytest

from search_lib import Action, BFS, DLS

from examples.torre_hanoi.problem import HanoiAction, HanoiState, TowerOfHanoi


def distances_to_goal(problem: TowerOfHanoi) -> Dict[HanoiState, int]:
    """Oráculo independente: BFS em grafo a partir do objetivo.

    Todo movimento tem inverso, então a distância até o objetivo é a mesma
    nos dois sentidos.
    """
    distance = {problem.goal_state: 0}
    queue = deque([problem.goal_state])
    while queue:
        state = queue.popleft()
        for action in problem.actions(state):
            result = action.apply(state)
            assert result is not None and isinstance(result[0], HanoiState)
            if result[0] not in distance:
                distance[result[0]] = distance[state] + 1
                queue.append(result[0])
    return distance


def replay(problem: TowerOfHanoi, start: HanoiState, path: List[Action]) -> float:
    """Reexecuta a lista de ações; confere que chega ao objetivo e devolve o custo."""
    state = start
    total = 0.0
    for step in path:
        transition = step.apply(state)
        assert transition is not None
        state, cost = transition
        assert isinstance(state, HanoiState)
        total += cost
    assert problem.is_goal(state)
    return total


def test_initial_and_goal_states() -> None:
    problem = TowerOfHanoi(3, 3)

    assert problem.initial_state == HanoiState(((3, 2, 1), (), ()))
    assert problem.goal_state == HanoiState(((), (), (3, 2, 1)))

    custom = TowerOfHanoi(2, 4, initial_peg=1, goal_peg=0)
    assert custom.initial_state == HanoiState(((), (2, 1), (), ()))
    assert custom.goal_state == HanoiState(((2, 1), (), (), ()))


def test_state_space_has_pegs_to_the_disks_states() -> None:
    # Cada disco está em um pino e a ordem em cada pino é forçada: n_pegs ** n_disks.
    for n_disks, n_pegs in [(1, 3), (3, 3), (4, 3), (3, 4)]:
        problem = TowerOfHanoi(n_disks, n_pegs)
        assert len(distances_to_goal(problem)) == n_pegs**n_disks


def test_actions_list_only_legal_moves() -> None:
    problem = TowerOfHanoi(3, 3)

    assert [repr(a) for a in problem.actions(problem.initial_state)] == [
        "MOVER(0, 1)",
        "MOVER(0, 2)",
    ]
    middle = HanoiState(((3, 2), (1,), ()))
    assert [repr(a) for a in problem.actions(middle)] == [
        "MOVER(0, 2)",
        "MOVER(1, 0)",
        "MOVER(1, 2)",
    ]


def test_action_apply_moves_the_top_disk_without_mutating() -> None:
    state = HanoiState(((3, 2, 1), (), ()))
    result = HanoiAction(0, 2).apply(state)

    assert result == (HanoiState(((3, 2), (), (1,))), 1.0)
    assert state == HanoiState(((3, 2, 1), (), ()))  # estado original intacto


def test_action_apply_rejects_illegal_moves() -> None:
    state = HanoiState(((3, 2), (1,), ()))

    assert HanoiAction(2, 0).apply(state) is None  # pino de origem vazio
    assert HanoiAction(0, 1).apply(state) is None  # disco 2 sobre o disco 1
    assert HanoiAction(0, 0).apply(state) is None  # mesmo pino
    assert HanoiAction(0, 5).apply(state) is None  # pino inexistente
    assert HanoiAction(1, 0).apply(state) == (HanoiState(((3, 2, 1), (), ())), 1.0)


def test_goal_test() -> None:
    problem = TowerOfHanoi(3, 3)

    assert problem.is_goal(problem.goal_state)
    assert not problem.is_goal(problem.initial_state)
    assert not problem.is_goal(HanoiState(((), (3,), (2, 1))))


def test_heuristic_values() -> None:
    problem = TowerOfHanoi(3, 3)

    assert problem.heuristic(problem.initial_state) == 3
    assert problem.heuristic(problem.goal_state) == 0
    assert problem.heuristic(HanoiState(((3,), (2,), (1,)))) == 2
    assert TowerOfHanoi(3, 3, heuristic="zero").heuristic(problem.initial_state) == 0


@pytest.mark.parametrize("n_disks, n_pegs", [(3, 3), (4, 3), (3, 4)])
def test_heuristic_is_admissible_and_consistent(n_disks: int, n_pegs: int) -> None:
    problem = TowerOfHanoi(n_disks, n_pegs)
    true_cost = distances_to_goal(problem)

    for state, cost in true_cost.items():
        assert problem.heuristic(state) <= cost
        for action in problem.actions(state):
            result = action.apply(state)
            assert result is not None
            assert problem.heuristic(state) <= result[1] + problem.heuristic(result[0])


@pytest.mark.parametrize("n_disks", [1, 2, 3])
def test_bfs_finds_the_optimal_solution_with_three_pegs(n_disks: int) -> None:
    problem = TowerOfHanoi(n_disks, 3)
    result = BFS(problem).search(problem.initial_state)

    assert result.solution_node is not None
    assert result.solution_node.depth == 2**n_disks - 1
    assert result.solution_node.cost == 2**n_disks - 1
    path = result.solution_node.get_path()
    assert replay(problem, problem.initial_state, path) == 2**n_disks - 1


def test_optimal_length_follows_two_to_the_n_minus_one() -> None:
    # Confere a fórmula com o oráculo para tamanhos grandes demais para a BFS em árvore.
    for n_disks in range(1, 8):
        problem = TowerOfHanoi(n_disks, 3)
        assert distances_to_goal(problem)[problem.initial_state] == 2**n_disks - 1


def test_extra_peg_shortens_the_solution() -> None:
    problem = TowerOfHanoi(3, 4)
    result = BFS(problem).search(problem.initial_state)

    assert result.solution_node is not None
    assert result.solution_node.depth == 5  # Frame-Stewart: 5 movimentos com 3 discos e 4 pinos
    assert distances_to_goal(problem)[problem.initial_state] == 5


def test_bfs_when_start_is_already_the_goal() -> None:
    problem = TowerOfHanoi(3, 3, initial_peg=2, goal_peg=2)
    result = BFS(problem).search(problem.initial_state)

    assert result.solution_node is not None
    assert result.solution_node.depth == 0
    assert result.metrics.nodes_generated == 1


@pytest.mark.parametrize("limit, solvable", [(6, False), (7, True), (8, True)])
def test_depth_limited_search_respects_the_limit(limit: int, solvable: bool) -> None:
    problem = TowerOfHanoi(3, 3)
    result = DLS(problem, limit).search(problem.initial_state)

    if not solvable:
        assert result.solution_node is None  # a solução ótima precisa de 7 movimentos
        return
    assert result.solution_node is not None
    assert result.solution_node.depth <= limit
    cost = replay(problem, problem.initial_state, result.solution_node.get_path())
    assert cost == result.solution_node.cost


def test_bfs_from_random_states_never_exceeds_the_walk_length() -> None:
    problem = TowerOfHanoi(3, 3)
    for seed in range(20):
        state = problem.random_state(5, seed=seed)
        result = BFS(problem).search(state)
        assert result.solution_node is not None
        assert result.solution_node.depth <= 5
        replay(problem, state, result.solution_node.get_path())


def test_random_state_is_deterministic_and_handles_edge_cases() -> None:
    problem = TowerOfHanoi(3, 3)

    assert problem.random_state(0) == problem.goal_state
    assert problem.random_state(6, seed=3) == problem.random_state(6, seed=3)
    # Com 1 disco e 3 pinos só existem 3 estados: passos além disso repetem estados sem falhar.
    tiny = TowerOfHanoi(1, 3)
    for seed in range(20):
        assert tiny.random_state(10, seed=seed) in distances_to_goal(tiny)


def test_invalid_arguments() -> None:
    with pytest.raises(ValueError):
        TowerOfHanoi(0, 3)
    with pytest.raises(ValueError):
        TowerOfHanoi(3, 2)
    with pytest.raises(ValueError):
        TowerOfHanoi(3, 3, initial_peg=3)
    with pytest.raises(ValueError):
        TowerOfHanoi(3, 3, goal_peg=-1)
    with pytest.raises(ValueError):
        TowerOfHanoi(3, 3, heuristic="manhattan")
    with pytest.raises(ValueError):
        TowerOfHanoi().random_state(-1)
    with pytest.raises(TypeError):
        TowerOfHanoi().heuristic(object())  # type: ignore[arg-type]