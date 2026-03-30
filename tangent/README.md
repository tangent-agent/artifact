# ✿ Tangent ✿ — Testing for Agents


## Install

This project is managed with `uv`.

```bash
uv sync
source .venv/bin/activate
```

## CLI usage

Generate `analysis.json` for a target repo:

```bash
tangent analyze --repo /path/to/python/repo --out /path/to/python/repo/analysis.json
```

Optional flags:

- `--repo-name` override the `Application.name`
- `--codeql` path to the CodeQL executable
- `--keep-workdir` keep intermediate CodeQL artifacts (useful when debugging queries)




