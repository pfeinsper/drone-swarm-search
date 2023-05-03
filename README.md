# Drone Swarm Search

## Poetry

### Run Python Script with poetry

```bash
poetry run python <python_file.py>
```

### Run Single Script

```bash
poetry run  <script_name>
```

To configure a new script, just edit the `pyproject.toml` file and add the script in the following format:

```toml
[tool.poetry.scripts]
<name> = "<module>:<function>"
```

As the example below:

```toml
[tool.poetry.scripts]
test = 'scripts:poetry_test'
```

### Poetry Environment

```bash
poetry shell
```

Essentially, this command will create a virtual environment and install all the dependencies in it. You can then run your scripts from the virtual environment.

### Poetry Environment in VSCode

If you are using VSCode, you can the following command to be able to select poetry environment as the default interpreter.

```bash
poetry config virtualenvs.in-project true
```

After that, reload the VSCode window and you should be able to select the poetry environment as the default interpreter.

### Add new dependency

```bash
poetry add <package_name>
```

If you want to add a dependency only for development, you can use the following command:

```bash
poetry add --dev <package_name>
```

If the dependency is only used for testing, you can use the following command:

```bash
poetry add pytest --group test
```

### Remove dependency

```bash
poetry remove <package_name>
```

### Drone Swarm Environment Docs

```python
from core.environment.env import DroneSwarmSearch

env = DroneSwarmSearch(
    grid_size=50, 
    render_mode="human", 
    render_grid = True,
    render_gradient = True,
    n_drones=11, 
    vector=[0.5, 0.5],
    person_initial_position = [5, 10],
    disperse_constant = 3)

def policy(obs, agent):
    actions = {}
    for i in range(11):
        actions["drone{}".format(i)] = 1
    return actions


observations = env.reset()
rewards = 0
done = False

while not done:
    actions = policy(observations, env.get_agents())
    observations, reward, _, done, info = env.step(actions)
    rewards += reward["total_reward"]
    done = True if True in [e for e in done.values()] else False

print(rewards)
```
