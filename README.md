# search-examples
Problemas de exemplo para search_lib (IA)

## Problemas implementados

### Sliding puzzle

`examples.sliding_puzzle.problem.SlidingPuzzle` aceita `rows` e `cols` (ambos
com padrão `3`) e `heuristic` (`"manhattan"` ou `"misplaced"`, padrão
`"manhattan"`). O estado é uma tupla plana; `0` representa o espaço vazio e o
objetivo ordena as peças de `1` a `n - 1`, seguido de `0`.

### Cubo de Rubik

`examples.rubik.problem.RubikCube` aceita `size` (`2` ou `3`, padrão `3`) e
`action_set` (`"quarter"`, `"half"` ou `"full"`, padrão `"quarter"`). Os
facelets são armazenados em uma tupla de cores, e os movimentos são permutações
pré-calculadas.

### Robô em grid

`examples.robot_grid.problem.GridRobot` aceita `width` e `height` (padrão
`3` por `3`), ou um mapa retangular por `obstacles` (lista de strings) ou
`file` (células `#` são obstáculos; `.` e espaços são livres). `actions`
escolhe entre `"GIRAR_HORARIO"`, `"GIRAR_ANTIHORARIO"`, `"FRENTE"` e
`"GIRAR_180"`; por padrão, habilita os dois giros de 90 graus e o avanço.
`turn_cost`, `forward_cost` e `half_turn_cost` configuram os custos, sendo o
custo do giro de 180 graus o dobro de `turn_cost` por padrão. `goal` recebe
`(x, y)` (padrão: canto inferior direito), e `goal_direction` opcional restringe
a direção final. `initial_state` permite configurar o ponto e direção de
partida. As heurísticas `"manhattan"` e `"manhattan_turns"` são admissíveis
quando os custos são não negativos; a segunda soma um limite inferior de custo
dos giros necessários. `random_state(n_actions, avoid_repeats=True, seed=None)`
gera uma caminhada aleatória reprodutível desde o objetivo.

### Testes

Execute os testes com `pytest` e verifique os tipos dos exemplos com
`mypy examples`.
