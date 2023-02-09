# Drone Swarm Search

### Run Python code with poetry

```bash
poetry run python <python_file.py>
```

### Run Script

```bash
poetry run  <script_name>
```

Para configurar um novo script, basta editar o arquivo `pyproject.toml` e adicionar o script no seguinte formato:

```toml
[tool.poetry.scripts]
<name> = "<module>:<function>"
```

Como no exemplo abaixo:

```toml
[tool.poetry.scripts]
test = 'scripts:poetry_test'
```
