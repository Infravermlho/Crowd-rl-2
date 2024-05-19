# Crowd-RL
Protótipo de ambiente multiagente para aprendizado por reforço utilizando PettingZoo.

O ambiente busca simular o fluxo de atendimento em um local fechado, do modo que, um agente entra em um local, encontra uma fila, é atendido e se retira.

É possivel configurar diversos aspectos do ambiente, como a planta, os tipos de agentes, quais filas podem acessar e etc.


## Instalação
O ambiente pode ser instalado da seguinte maneira:
```bash
git clone https://github.com/Infravermlho/crowd-rl
pip install ./crowd-rl
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
O ambiente conta com 4 configurações pré montadas `shop` `mall` `health` `filter` e mais podem ser criadas utilizando os seguintes parametros:

### Todo
