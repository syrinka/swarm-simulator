# Swarm Intelligence Simulator

使用例：

```python
from swarm import ArgInfo, Problem
from swarm.algos.PSO import QPSO

args = [ArgInfo(min=-5, max=5), ArgInfo(min=-5, max=5)]
prob = Problem(args, lambda x: x[0]**2 + x[1]**2)

swarm = QPSO(population=50, problem=prob)
swarm.evolve(50)

for i in swarm.besty_history():
    print(str(i))
```
