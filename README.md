# agentic-repo-insights
Agentic tool for ad-hoc analysis

install uv
```shell
curl -LsSf https://astral.sh/uv/install.sh | sh
```

```shell
uv init rnd-python
uv venv  --python 3.12 &&
source .venv/bin/activate 
uv pip install -r ../requirements.txt
uv pip freeze > temp_requirements.txt
uv add $(cat temp_requirements.txt | tr '\n' ' ')
```

Run analysis
```shell
REPO=/Users/adzhumurat/PycharmProjects/fintopio-admin-new \
    make run
```