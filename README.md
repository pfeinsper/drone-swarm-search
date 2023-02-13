# Drone Swarm Search

### Run Python code with poetry

```bash
poetry run python <python_file.py>
```

### Run Script

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
