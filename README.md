# alexwu

Alex's data science helper functions

## Installation

```sh
pip install git+https://github.com/alexanderwu/aw.git
```

OR

```sh
uv add --no-sync git+https://github.com/alexanderwu/aw.git
uv pip compile pyproject.toml -o requirements.txt
# uv pip sync requirements.txt
uv pip install -r requirements.txt
```

## Development

```sh
# uv venv
# source .venv/bin/activate
# uv pip compile -o requirements.txt
uv pip sync requirements.txt
uv pip install -e .
```
