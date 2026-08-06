# GFS Studio Product Surface

GFS Studio turns the simulator, world model, cognition layer, and evidence
reports into one persistent user workflow. The lower-level research scripts
remain available, but product users work through a studio session.

## Product loop

```text
initialize studio -> check readiness -> run matches -> inspect dashboard
       -> accumulate session history -> change mode only with evidence
```

Three explicit modes prevent research flags from leaking into stable use:

| Mode | Stable simulator | Calibrated world model | Real LLM |
|---|---:|---:|---:|
| `stable` | yes | no | no |
| `research` | yes | yes | no |
| `cognitive` | yes | yes | yes, credentials required |

The stable mode is the default. Research mode requires the accepted candidate
checkpoint. Cognitive mode additionally refuses to run without `API_KEY`.

## Quick start

```bash
python gfs.py studio init --name "My World Cup" --mode stable --seed 42
python gfs.py studio status
python gfs.py studio match --home Brazil --away Argentina --fast
```

The session is persisted under `data/persistence/product_session.json`.
Every match creates both an auditable JSON report and a human-readable HTML
dashboard under `outputs/studio/<studio>/matches/`. The report composes score,
xG, possession, passing, shooting, psychology, tactical drift, world-model
calibration, decision adoption, cognitive plans, timeline, artifacts, and an
evidence snapshot in one envelope.

GFS Studio does not promote a research checkpoint. It reads the frozen release
and evaluation artifacts and exposes their status; the release manifest remains
the only production authority.
