# Publishing to GitHub (clean release)

## Before you push

1. Copy `.env.example` to `.env` locally and fill in secrets. **Never commit `.env`.** It is listed in `.gitignore`.
2. Do not commit `outputs/`, `*.log`, or `.cursor/` — they are ignored.
3. If you ever pasted an API key into chat or a script, rotate that key at your provider before making the repo public.

## First-time repo setup

If Git has no author configured yet, set it once (global or **local to this repo**):

```bash
git config user.name "Your Name"
git config user.email "your.email@example.com"
```

Then:

```bash
git init
git add .
git status   # confirm no .env or outputs/
git commit -m "Initial commit: Generative Football Society V13"
```

This checkout may already have an initial commit on `main` with a placeholder author. To fix the last commit author:

```bash
git commit --amend --reset-author --no-edit
```

Create an empty repository on GitHub, then:

```bash
git branch -M main
git remote add origin https://github.com/<you>/<repo>.git
git push -u origin main
```

## Optional: verify nothing sensitive is staged

```bash
git grep -i "sk-" --cached || true
git grep -i "api_key=" --cached || true
```
