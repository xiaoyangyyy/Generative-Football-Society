# Generative Football Society (GFS) V13

Generative Football Society is a World Cup-scale multi-agent social simulation framework.

It combines:
- continuous state dynamics
- affective appraisal and coping
- structured memory and belief fields
- constrained LLM reflection
- multi-expert fusion for match-day decisions

---

## Core Pipeline

`Event -> Appraisal -> Emotion -> Coping -> Memory Weight -> Belief Field -> State/Tactics -> Match Outcome`

Decision integration:
- subsystem experts run in parallel (`phys`, `affect`, `social`, `tactic`, `governance`)
- `FusionController` performs a single soft-gated fusion step
- fused outputs drive effective status, volatility, and chaos terms

---

## System 1 / System 2

- **System 1 (continuous math engine)**  
  Canonical state is latent `z_state`, projected via smooth bounded functions.

- **System 2 (LLM cognition)**  
  LLM provides bounded, confidence-weighted, evidence-linked suggestions.
  LLM does not directly decide match outcomes; the math engine remains the source of score generation.

---

## Affective Appraisal Model

Appraisal (6D):
- `impact` in `[-1, 1]`
- `novelty`, `control`, `certainty`, `norm_violation` in `[0, 1]`
- `agency` in `[-1, 1]`

Emotion (5D, softmax):
- `pride`, `anger`, `shame`, `fear`, `determination`

Coping (4D):
- softmax: `planning`, `self_correction`, `external_blame`
- bounded shift: `risk_shift = tanh(...)`

---

## Memory and Belief

Structured memory includes appraisal/emotion/coping fields and `memory_weight`.

Key behavior:
- all events are written to memory (no hard write gate)
- influence decays continuously via exponential weighting
- retrieval uses all episodic memories with softmax weighting
- top-k is only for display/export convenience

Belief field:
- tracks continuous `support` and `contradiction`
- confidence is continuous (`sigmoid(alpha * (support - contradiction))`)
- policy effects are softly activated and bounded

---

## Canonical State Notice

- canonical affective model: `appraisal + emotion_profile + coping`
- canonical state model: `z_state`
- `hidden_state` is retained only as a compatibility read view

---

## Key Modules

- `src/simulation/agent.py`
- `src/simulation/fusion_controller.py`
- `src/simulation/tournament_2026.py`
- `src/simulation/social_dialogue.py`
- `src/simulation/llm_engine.py`

---

## Install

```bash
pip install -r requirements.txt
```

Recommended Python: 3.10+

---

## Configuration

Environment variables:
- `API_KEY` or `OPENAI_API_KEY`
- `BASE_URL` (optional)
- `MODEL_NAME` (optional)
- `GFS_SEED` (optional)

Example:

```env
API_KEY=your_real_key
BASE_URL=https://api.openai.com/v1
MODEL_NAME=gpt-4-turbo-preview
GFS_SEED=42
```

---

## Quick Run

Tactical quick smoke:

```bash
python run_world_cup_2026_tactical.py --quick --no-interactive --seed 42
```

Full tournament:

```bash
python run_world_cup_2026_full.py
```

Monte Carlo:

```bash
python main_monte_carlo.py -n 1000 --seed 42
```

---

## Outputs

Generated artifacts are written under `outputs/` during runs.

---

## Clean Release Packaging

Use this to avoid shipping `outputs/`, caches, and local env files:

```bash
python package_release_zip.py --out release.zip
```

---

## License

MIT License. See `LICENSE`.
