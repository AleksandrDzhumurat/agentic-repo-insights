# agentic-repo-insights
Agentic tool for ad-hoc analysis

target repo: [fintopio-admin-new](https://github.com/fintopio/fintopio-admin-new/tree/dev#)

# V0.2: codex agent

```shell
make run-codex
# V0.2: codex agent

```shell
make run-claude
```
```

```
The code interacts with a backend API to fetch data. Generate all SQL queries that app uses to fetch the data ans save this queries to '/Users/adzhumurat/PycharmProjects/agentic-repo-insights/EXPLANATION_TABLES.md'. Attach names of source code files where you extracted SQL queries. Avoid analyze images and binary code, only text files (code, json etc) 
```

# V1.0: RAG approach
install uv
```shell
curl -LsSf https://astral.sh/uv/install.sh | sh
```

```shell
uv init rnd-python
uv venv  --python 3.12 &&
source .venv/bin/activate 

cd rnd-python & \
uv pip install -r ../requirements.txt \
uv pip freeze > temp_requirements.txt \
uv add $(cat temp_requirements.txt | tr '\n' ' ') \
cd ..
```

Run analysis
```shell
REPO=/Users/adzhumurat/PycharmProjects/fintopio-admin-new \
    make run
```

test vector db
```shell
curl -X POST "http://localhost:8080/query" -H "Content-Type: application/json" -d '{ "query": "what database does this app use?", "max_results": 7 }'
```

