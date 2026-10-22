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

### Mapa da Romênia

Caminhos entre cidades do mapa de estradas do livro (Russell & Norvig). Cada ação é `IR_PARA(cidade)`, com custo igual à distância em km.

- `goal`: cidade objetivo (padrão: `"Bucharest"`)
- `initial_city`: cidade de partida (padrão: `"Arad"`)
- `heuristic`: `"straight_line"` (distância em linha reta até Bucharest; padrão quando o objetivo é Bucharest) ou `"zero"`
- `roads`: lista de `(cidade_a, cidade_b, distância)` para usar um mapa personalizado (então só vale `"zero"`)

```python
from search_lib import BFS, DLS
from examples.mapa_romenia.problem import Romania

problem = Romania()
start = problem.initial_state  # Arad
print(BFS(problem).search(start).solution_node.cost)  # 450 (menos estradas, não o mais barato)
print(DLS(problem, 4).search(start).solution_node.cost)
```

Atenção: as buscas em árvore não verificam estados repetidos e as estradas são bidirecionais, então `DFS` sem limite não termina neste problema e qualquer busca sem solução também não. Use `DLS` ou garanta que existe solução.

### Torre de Hanói

Move todos os discos de um pino para outro, um por vez, sem nunca colocar um disco maior sobre um menor. Cada ação é `MOVER(origem, destino)`, com custo 1, e move o disco do topo do pino de origem. Os pinos são numerados a partir de `0`.

- `n_disks`: quantidade de discos (padrão `3`)
- `n_pegs`: quantidade de pinos, no mínimo `3` (padrão `3`)
- `initial_peg`: pino onde todos os discos começam (padrão `0`)
- `goal_peg`: pino onde todos devem terminar (padrão: o último pino)
- `heuristic`: `"disks_off_goal"` (padrão; quantidade de discos fora do pino objetivo, admissível e consistente) ou `"zero"`

O estado é `HanoiState(pegs)`: cada pino é uma tupla do fundo para o topo, com `1` sendo o menor disco. Por exemplo, `((3, 2, 1), (), ())` tem 3 discos no primeiro pino.

```python
from search_lib import BFS
from examples.torre_hanoi.problem import TowerOfHanoi

problem = TowerOfHanoi(n_disks=3, n_pegs=3)
result = BFS(problem).search(problem.initial_state)
print(result.solution_node.cost)  # 7 (= 2**3 - 1)
print(result.solution_node.get_path())  # [MOVER(0, 2), MOVER(0, 1), ...]
```

Com 3 pinos a solução ótima tem `2**n_disks - 1` movimentos; com um pino extra ela encurta (3 discos e 4 pinos: 5 movimentos). Como as buscas em árvore repetem estados, o `BFS` só é viável para poucos discos (3 em 3 pinos já gera milhares de nós; 4 em 3 pinos geraria milhões).



