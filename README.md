# search-examples

Problemas de exemplo para a biblioteca [search_lib](https://github.com/GustavoFelipeM/search_lib) (IA, busca clássica).

## Como usar

```powershell
pip install -e ".[dev]"
pytest
```

## Problemas

### Sliding puzzle

Quebra-cabeça de deslizar peças. O `0` é o espaço vazio.

- `rows`, `cols`: tamanho do tabuleiro (padrão `3` e `3`)
- `heuristic`: `"manhattan"` (padrão) ou `"misplaced"`

```python
from search_lib import BFS
from examples.sliding_puzzle.problem import SlidingPuzzle

problem = SlidingPuzzle(3, 3)
start = problem.random_state(4, seed=1)
print(BFS(problem).search(start).solution_node.cost)
```

### Cubo de Rubik

Cubo 2x2 ou 3x3, com diferentes conjuntos de movimentos.

- `size`: `2` ou `3` (padrão `3`)
- `action_set`: `"quarter"` (padrão), `"half"` ou `"full"`

```python
from search_lib import BFS
from examples.rubik.problem import RubikCube

problem = RubikCube(2, "quarter")
start = problem.random_state(2, seed=1)
print(BFS(problem).search(start).solution_node.cost)
```

### Robô em grid

Robô que gira e anda para frente até chegar ao objetivo.

- Mapa: `width` e `height` (padrão `3` e `3`), ou `obstacles` (lista de strings), ou `file` (arquivo de texto). Nos mapas, `#` é obstáculo e `.` é livre.
- `actions`: `"GIRAR_HORARIO"`, `"GIRAR_ANTIHORARIO"`, `"FRENTE"`, `"GIRAR_180"` (padrão: giros de 90° e frente)
- Custos: `turn_cost` (`1.0`), `forward_cost` (`1.0`), `half_turn_cost` (`2.0`; padrão: o dobro de `turn_cost`)
- `goal`: célula objetivo `(x, y)` (padrão: canto inferior direito)
- `goal_direction`: direção final (opcional; se omitida, qualquer direção satisfaz o objetivo)
- `initial_state`: posição e direção de partida (padrão: `(0, 0, "NORTE")`)
- Heurísticas: `"manhattan"` (padrão) e `"manhattan_turns"`

```python
from search_lib import BFS
from examples.robot_grid.problem import GridRobot

problem = GridRobot(width=3, height=3)
start = problem.random_state(4, seed=1)
print(BFS(problem).search(start).solution_node.cost)
```
