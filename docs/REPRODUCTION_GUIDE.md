# GFS Registered-Report Reproduction Guide

This guide separates read-only package review from compute-authorized
confirmatory execution. Completing the first three sections does not run a
match, train a model, or call an external provider.

## 1. Current evidence boundary

- Manuscript stage: registered-report draft.
- Confirmatory experiment: not executed.
- Independent reproduction: not performed.
- Target-user validation: preregistered, zero participants observed.
- Independent-reproduction handoff: registered, waiting for container and
  confirmatory-result prerequisites.
- Stable release: 7.0.0.
- Sealed M1: research-only and default-off.
- Environment: two verified 64-package reference locks—Python 3.12 Linux CPU
  deployment and Python 3.13 Windows CPU development—with current CycloneDX
  SBOMs and immutable container-image digests; no built-container claim.

The authoritative inventory is
`data/evaluation/reproduction_manifest_v1.json`. Do not infer that an absent
confirmatory result is a null or negative result.

## 2. Read-only package audit

From the repository root:

```bash
python scripts/verify_paper_package.py
python scripts/verify_reproduction_release.py
python scripts/build_supply_chain_sbom.py --check
python scripts/verify_reproducibility_matrix.py
python scripts/verify_local_runtime.py
python scripts/verify_container_images.py
python scripts/verify_data_release.py
python scripts/product_validation_study.py
python scripts/verify_independent_reproduction.py
python scripts/audit_research_evidence.py --check
python scripts/run_formal_experiment.py
```

Expected state:

- every paper claim has exactly one manuscript marker and valid evidence;
- all eight direct dependencies match both 64-package reference locks, every
  locked package has SHA-256 evidence, and Docker enforces `--require-hashes`;
- both CycloneDX 1.6 SBOMs exactly match their locked runtime closures;
- all eight direct dependencies import in the observed Windows runtime;
- Python and Caddy retain readable tags while resolving through recorded,
  immutable OCI index digests;
- all five known external source families have a source-specific, default-deny
  decision, with only the IDSSE/Sportec-derived supplement approved;
- the 37-file deterministic data-supplement manifest resolves to archive
  SHA-256 `df25a33dea93902bca965ba725ee72c67901015b5ed8808585a2e639d8dc7d79`;
- the frozen research evidence report is current;
- protocol and sealed checkpoint identities pass;
- confirmatory state is `not_started`, with 60 runs remaining;
- no progress or decision file exists.
- product-validation and independent-reproduction protocol audits pass while
  retaining zero-execution and not-performed states.

Any different state means this pre-execution package is stale and must be
reviewed before using the manuscript.

## 3. Code regression

Install the project in an isolated environment. The project declares Python
3.11 through 3.13; the deployment image currently uses Python 3.12.11.

```bash
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .
python -m pytest -q
```

For the deployment target, install inside Python 3.12 x86_64 Linux with:

```bash
python -m pip install --require-hashes -r requirements-linux-py312.lock
python -m pip install --require-hashes -r requirements-ci-linux-py312.lock
python -m pip install --no-build-isolation --no-deps -e .
python -m ruff check src
python -m ruff check src/product/web.py src/match_engine/action_engine.py --select C901,PLR0912,PLR0915
python -m ruff check scripts/action_adoption_study.py scripts/run_formal_experiment.py
python -m ruff check scripts/action_adoption_study.py scripts/run_formal_experiment.py --select C901,PLR0912,PLR0915
python -m pytest -q
```

For the Windows development reference profile, use CPython 3.13 x86_64:

```powershell
python -m pip install --require-hashes -r requirements-windows-py313.lock
python -m pip install --no-build-isolation --no-deps -e .
```

The target runtime lock contains 64 exact packages, includes the CPU Torch wheel,
records all accepted distribution SHA-256 values, and has passed both uv and
pip dry-run resolution. `tzdata` and `colorama` are deliberate shims so pip can
also audit the Linux lock from a Windows host whose marker evaluation follows
the host. The separate CI-only Linux lock contains exact hashed pytest and Ruff
closures, so neither test execution nor the `src` static gate depends on tools
that happen to be preinstalled on the GitHub runner. The matrix intentionally
defines two runtime reference profiles; it does not
claim every Python/OS combination permitted by `pyproject.toml`. The observed
Windows import smoke is direct runtime evidence, but it is not a Linux import
test or an independent clean-room reproduction.

The image references are digest-pinned. On a Docker host, the remaining build
gate can be executed explicitly with:

```bash
python scripts/verify_container_runtime.py --execute \
  --out data/evaluation/container_runtime_verification_v1.json
```

That command builds and launches an ephemeral read-only container, verifies
the declared and actual non-root UID/GID, imports all direct dependencies with
networking disabled, inspects the runtime hardening options, validates both the
`/healthz` JSON contract and Docker health state, and removes both the
temporary container and image. It does not run a match or train a model.

The pinned GitHub Actions handoff in `.github/workflows/ci.yml` uses exact
official action commits, Ubuntu 24.04, exact CPython 3.12.11, the hashed Linux
runtime and CI-tool closures, a mandatory default source-health gate, a focused
complexity/branch/statement gate for the Web product and action-adoption core,
default plus focused complexity gates for both fixed-budget research runners,
and the same explicitly non-training test boundary. It
does not perform an unpinned pip self-upgrade. A successful push run uploads the
runtime and deployment reports and produces a GitHub artifact provenance
attestation. An uploaded artifact is evidence for its recorded commit only; it
does not make the current checkout pass automatically. GitHub-hosted runner
images still receive platform updates, so the runtime report records the
observed Docker server rather than claiming an immutable virtual machine.

## 4. Data review

Read `docs/RESEARCH_DATA_CARD.md` and
`data/evaluation/data_license_registry_v1.json`, then inspect each provider
manifest. Verify source terms independently before downloading or
redistributing data.
Repository-derived hashes establish file identity only when the referenced
source files are lawfully available; they do not grant a license.

The approved IDSSE/Sportec-only supplement is content-addressed but not stored
as a second 55 MiB repository blob. After the read-only verifier passes, an
authorized release operator can materialize it deterministically with:

```bash
python scripts/build_data_release_archive.py --check
python scripts/build_data_release_archive.py \
  --materialize build/gfs-idsse-data-supplement-v1.zip
```

The builder admits only the exact path families in the checked-in manifest,
uses fixed ZIP metadata, and includes `docs/IDSSE_ATTRIBUTION.md`. It excludes
all four unapproved source families. This supplement boundary does not audit
or rewrite repository history.

## 5. Compute-authorized confirmatory execution

This section is intentionally not part of package verification. It runs 60
full-length simulations and requires explicit authorization:

```bash
python scripts/run_formal_experiment.py --execute --authorization I_AUTHORIZE_GFS_FORMAL_EXPERIMENT_V2
```

The workflow is resumable only under identical protocol, checkpoint, and
critical-code identity. The identity also binds the observable calibration
contract, StatsBomb baselines, and joint baselines. The exact authorization
phrase is checked by the runner, not merely documented by this guide. It
prohibits interim analysis and optional stopping.
After all 60 runs are complete:

```bash
python scripts/run_formal_experiment.py --analyze --authorization I_AUTHORIZE_GFS_FORMAL_EXPERIMENT_V2
```

Analysis writes the decision only after exact pairing, sample-budget, validity,
and identity checks pass. Copy values into the manuscript solely from that
decision artifact.

## 6. Independent reviewer checklist

An independent reviewer should record:

1. repository commit and clean/dirty state;
2. operating system, Python version, package resolver output, CPU/GPU, and
   deterministic settings;
3. protocol, checkpoint, and critical-code hashes;
4. data sources, versions, licenses, and manifest verification;
5. commands executed and whether any deviation occurred;
6. completeness and exact M0/M1 pairing for all 60 confirmatory runs;
7. result-selected branch and exact M0/M1-predict-only/M1 pairing for all 72
   mechanism-replication runs;
8. both regenerated point estimates, intervals, mechanism quantities, validity
   gates, and decisions;
9. differences from either checked-in decision and their diagnosis;
10. reviewer relationship to the implementation team; and
11. a signed conclusion of reproduced, not reproduced, or inconclusive.

Until such a report exists, the project must retain
`INDEPENDENT_REPRODUCTION_STATUS: NOT_PERFORMED`.
