"""Testes de completude em um espaço infinito com ramificação finita."""

from search_lib import BFS, DLS
from examples.linha_infinita.problem import InfiniteLine, IntegerState


def test_integer_line_is_unbounded_and_actions_are_reversible() -> None:
    problem = InfiniteLine(goal=4)
    state = problem.initial_state
    for _ in range(100):
        result = problem.actions(state)[0].apply(state)
        assert result is not None
        state = result[0]
    assert state == IntegerState(100)
    assert len(problem.actions(state)) == 2


def test_bfs_is_complete_and_optimal_on_finitely_reachable_goal() -> None:
    problem = InfiniteLine(start=-3, goal=4)
    result = BFS(problem).search(problem.initial_state)

    assert result.solution_node is not None
    assert result.solution_node.state == problem.goal_state
    assert result.solution_node.depth == 7
    assert result.solution_node.cost == 7.0


def test_depth_limited_search_finds_solution_when_limit_is_sufficient() -> None:
    problem = InfiniteLine(start=0, goal=3)
    assert DLS(problem, 2).search(problem.initial_state).solution_node is None
    result = DLS(problem, 3).search(problem.initial_state)

    assert result.solution_node is not None
    assert result.solution_node.state == problem.goal_state
    assert result.solution_node.depth == 3


def test_infinite_line_heuristic_is_exact_and_consistent() -> None:
    problem = InfiniteLine(start=-2, goal=5)
    assert problem.heuristic(problem.initial_state) == 7
    assert problem.heuristic(problem.goal_state) == 0
    for value in range(-5, 10):
        state = IntegerState(value)
        for action in problem.actions(state):
            transition = action.apply(state)
            assert transition is not None
            assert problem.heuristic(state) <= transition[1] + problem.heuristic(transition[0])
