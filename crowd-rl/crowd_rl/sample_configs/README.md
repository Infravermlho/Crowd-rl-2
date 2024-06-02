# Pasta com exemplos de configs

## Exemplo de estrutura de config:

| Campo      | Descrição                                                                                                                                                                     | Exemplo                                                                                                                                                                                                                                                                                                      |
|------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| groups     | Lista com os grupos de agentes, onde é definido o id do grupo e a quantidade de agentes pertencentes ao grupo.                                                                | [{ <br>"name": "generico",    <br>"amount": 1 <br>},  <br>{ <br>"name": "preferencial",    <br>"amount": 1 <br>}]                                                                                                                                                                                    |
| attendants | Lista com os atendentes, onde são definidos o id do atendente, sua posição e o intervalo de tempo que ele costuma tomar para atender um cliente.                              | [{ <br>"id": "atendente1",    <br>"pos": {"x": 2, "y": 1}, <br>"processing_time": [10, 20] <br>}]                                                                                                                                                                                                        |
| queues     | Lista com as filas, onde é definido a ordem da fila, quais grupos de agentes são aceitos por ela, quais atendentes são associados a ela e os lugares ocupados pelo seu corpo. | [{ <br>"order": 0, <br>"accepts": ["common", "pref"], <br>"attendants": ["c1"], <br>"wait_spots": [ <br>  {"x": 2, "y": 3}, <br>  {"x": 2, "y": 4}, <br>  {"x": 2, "y": 5}, <br>], <br>}]                                                                                |
| entrances  | Lista com as entradas, definindo sua posição, taxa de entrada agentes e quais grupos de agentes podem a usar                                                                  | [{  <br>"pos": {"x": 0, "y": 6},   <br>"rate": 5,   <br>"accepts": ["common", "pref"]  <br>}]                                                                                                                                                                                                |
| exits      | Lista com as saidas, definindo sua posição e quais grupos de agentes podem a usar                                                                                             | [{ <br>"pos": {"x":4,"y":0},  <br>"accepts": ["common", "pref"] <br>}]                                                                                                                                                                                                                                       |
| worldmap   | Matriz 2d contendo zeros e uns representando a planta do local                                                                                                                | [ <br>        [1, 1, 1, 1, 0, 1], <br>        [1, 0, 0, 0, 0, 1], <br>        [1, 0, 0, 0, 0, 1], <br>        [1, 0, 0, 0, 0, 1], <br>        [1, 0, 0, 0, 0, 1], <br>        [1, 0, 1, 0, 0, 1], <br>        [0, 0, 0, 0, 0, 1], <br>        [1, 1, 1, 1, 1, 1], <br>], |

```python
config = {
    "attendants": [
        {"id": "c1", "pos": {"x": 2, "y": 1}, "processing_time": 20},
    ],
    "queues": [
        {
            "order": 0,
            "accepts": ["common", "pref"],
            "attendants": ["c1"],
            "wait_spots": [
                {"x": 2, "y": 3},
                {"x": 2, "y": 4},
                {"x": 2, "y": 5},
            ],
        },
    ],
    "entrances": [{"pos": {"x": 0, "y": 6}, "rate": 5, "accepts": ["common", "pref"]}],
    "exits": [
        {"pos": {"x": 4, "y": 0}, "accepts": ["common"]},
        {"pos": {"x": 1, "y": 0}, "accepts": ["pref"]},
    ],
    "groups": [
        {"name": "common", "amount": 1},
        {"name": "pref", "amount": 1},
    ],
    "worldmap": [
        [1, 0, 1, 1, 0, 1],
        [1, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 1],
        [1, 0, 1, 0, 0, 1],
        [0, 0, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 1],
    ],
}

from crowd_rl import crowd_rl_v0 as crowd
from crowd_rl.policies import shortest_distance

env = crowd.parallel_env(config=config, queue_bias=2, render_mode="human", render_fps=8)

observations, infos = env.reset(seed=42)
while env.agents:
    actions = shortest_distance(observations)
    observations, rewards, terminations, truncations, infos = env.step(actions)

env.close()
```
