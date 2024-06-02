# Crowd-RL
Protótipo de ambiente multiagente para aprendizado por reforço utilizando **PettingZoo**.

O ambiente busca resolver o problema de pathfinding e simular o fluxo de atendimento em um local fechado, do modo que, um agente entra em um local, encontra uma fila, é atendido e se retira.

É possivel configurar diversos aspectos do ambiente, como a planta, os tipos de agentes, quais filas podem acessar e etc.


## Instalação
O ambiente pode ser instalado da seguinte maneira:
```bash
git clone https://github.com/Infravermlho/crowd-rl
pip install ./crowd-rl/crowd-rl
```
## Uso
**Executando um ambiente pré-incluidos com ações aleatorias**
```python
from crowd_rl import crowd_rl_v0 as crowd
from crowd_rl.test_configs import mall

env = crowd.env(config=mall, render_fps=10, render_mode="human", max_cycles=900)
env.reset(seed=42)

for agent in env.agent_iter():
  observation, reward, termination, truncation, info = env.last()

  if termination or truncation:
    action = None
  else:
    action = env.action_space(agent).sample()

  env.step(action)

env.close()
```
**Executando a API Paralela com a politica de menor distância**
```python

from crowd_rl import crowd_rl_v0 as crowd
from crowd_rl.policies import shortest_distance
from crowd_rl.sample_configs import mall

env = crowd.parallel_env(config=mall, render_mode="human", render_fps=12)

observations, infos = env.reset(seed=42)
while env.agents:
    actions = shortest_distance(observations)
    observations, rewards, terminations, truncations, infos = env.step(actions)

env.close()
```
O ambiente conta com 4 configurações pré montadas `shop` `mall` `health` `filter` e mais podem ser criadas utilizando a documentação

#### Parametros:
- `config`
  - **Obrigatorio**
  - Aceita um dict ou json no formato defindo aqui
- `render_mode`
  - **Opcional = None** 
  - **Aceita:** None | "human" | "rbg_array"
  - None: Sem renderização
  - "human": Renderização em pygame 
  - "rgb_array": Retorna arrays contendo o valor rgb de cada pixel
- `render_fps`
  - **Opcional = 15**
  - **Aceita:** valores inteiros
  - frames por segundo da renderização, só é relevante no render_mode: "human"
- `max_cycles`
  - **Opcional = 900**
  - **Aceita:** valores inteiros
  - quantidade maxima de steps até o fim da simulação
- `collect_data`
  - **Opcional = False**
  - **Aceita:** Boolean
  - Faz com que a função .close() retorne um dict com os dados da simulação, pode ser usado para analise de dados.
  - **Grande impacto na performance, não use durante o treinamento.**
