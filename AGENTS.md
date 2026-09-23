# tricolour: A numba accelerated SumTreshold flagging suite

## Tooling

Tricolour is managed by the `uv` tool.

- Installation: `uv sync --all-extras`
- Testing: `uv run --group test --all-extras py.test -s -vvv tests/`
- Pre-commit hooks: `uv run pre-commit run -a`
