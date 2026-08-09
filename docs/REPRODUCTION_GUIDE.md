# GFS Registered-Report Reproduction Guide

This guide separates read-only package review from compute-authorized
confirmatory execution. Completing the first three sections does not run a
match, train a model, or call an external provider.

## 1. Current evidence boundary

- Manuscript stage: registered-report draft.
- Confirmatory experiment: not executed.
- Independent reproduction: not performed.
- Stable release: 7.0.0.
- Sealed M1: research-only and default-off.
- Environment: 64-package Python 3.12 Linux CPU transitive hash lock; no
  cross-platform lock matrix or built container claim.

The authoritative inventory is
`data/evaluation/reproduction_manifest_v1.json`. Do not infer that an absent
confirmatory result is a null or negative result.

## 2. Read-only package audit

From the repository root:

```bash
python scripts/verify_paper_package.py
python scripts/verify_reproduction_release.py
python scripts/build_supply_chain_sbom.py --check
python scripts/audit_research_evidence.py --check
python scripts/run_formal_experiment.py
```

Expected state:

- every paper claim has exactly one manuscript marker and valid evidence;
- all eight direct dependencies match the project contract and the 64-package
  target lock, every locked package has SHA-256 evidence, and Docker enforces
  `--require-hashes`;
- the CycloneDX 1.6 SBOM exactly matches the locked runtime closure;
- all five known external source families have a default-deny archive
  decision;
- the frozen research evidence report is current;
- protocol and sealed checkpoint identities pass;
- confirmatory state is `not_started`, with 60 runs remaining;
- no progress or decision file exists.

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
python -m pip install --no-build-isolation --no-deps -e .
```

The target lock contains 64 exact packages, includes the CPU Torch wheel,
records all accepted distribution SHA-256 values, and has passed both uv and
pip dry-run resolution. `tzdata` and `colorama` are deliberate shims so pip can
also audit the Linux lock from a Windows host whose marker evaluation follows
the host. The lock is not a cross-platform matrix, and the observed local
environment snapshot remains diagnostic rather than evidence of a Linux
runtime import test.

## 4. Data review

Read `docs/RESEARCH_DATA_CARD.md` and
`data/evaluation/data_license_registry_v1.json`, then inspect each provider
manifest. Verify source terms independently before downloading or
redistributing data.
Repository-derived hashes establish file identity only when the referenced
source files are lawfully available; they do not grant a license.

## 5. Compute-authorized confirmatory execution

This section is intentionally not part of package verification. It runs 60
full-length simulations and requires explicit authorization:

```bash
python scripts/run_formal_experiment.py --execute
```

The workflow is resumable only under identical protocol, checkpoint, and
critical-code identity. It prohibits interim analysis and optional stopping.
After all 60 runs are complete:

```bash
python scripts/run_formal_experiment.py --analyze
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
6. progress completeness and exact M0/M1 pairing;
7. regenerated point estimate, interval, changed-pair fraction, and decision;
8. differences from the checked-in decision and their diagnosis;
9. reviewer relationship to the implementation team; and
10. a signed conclusion of reproduced, not reproduced, or inconclusive.

Until such a report exists, the project must retain
`INDEPENDENT_REPRODUCTION_STATUS: NOT_PERFORMED`.
