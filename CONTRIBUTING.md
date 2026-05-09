# Contributing to Generative Football Society

Thanks for your interest in contributing.

## Development Setup

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Configure environment variables:

```bash
cp .env.example .env
```

For LLM-enabled flows, set `API_KEY` or `OPENAI_API_KEY` in `.env`.

## Local Validation

Before opening a PR, run:

```bash
python -m compileall src run_world_cup_2026_tactical.py
python smoke_llm_json_parser.py
python run_world_cup_2026_tactical.py --quick --no-interactive
```

## Pull Request Guidelines

- Keep PRs focused and small.
- Add clear rationale in the PR description.
- Do not commit secrets (`.env`, private keys, credentials).
- Update `README.md` when behavior or CLI usage changes.
- Update `CHANGELOG.md` for user-facing changes.

## Commit Message Style (recommended)

Use short imperative messages, for example:

- `add social dialogue market module`
- `fix tactical no-interactive LLM init path`
- `update README for open-source release`
