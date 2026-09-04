# Excellence evidence kit

The evidence kit turns the remaining product and academic obligations into a
single privacy-bounded handoff for credential owners, study moderators,
reviewers, operators, independent reproducers, and paper authors.

Audit the kit entirely in memory:

```text
python gfs.py studio evidence-kit
```

Explicitly materialize the deterministic archive:

```text
python gfs.py studio evidence-kit --materialize build/evidence-kits/gfs-excellence-evidence-kit-v1.zip
```

Materialization is confined to `build/evidence-kits/`, refuses replacement
unless `--overwrite` is supplied, and never writes into the authoritative
`data/evaluation/` evidence tree.

The authenticated Studio Web release center also exposes
`GET /api/v1/excellence/evidence-kit.zip`. It builds the same deterministic ZIP
in memory, verifies its digest before responding, writes no server-side file,
and returns `X-GFS-Artifact-SHA256` plus `X-GFS-Template-Only: true`. Remote
access reaches this route only after the existing allowed-Host, trusted HTTPS,
and secure-session checks.

Every JSON template contains `template_only: true`, invalid placeholders, and
an extra template notice. The strict study and review validators therefore
reject it by default. Operators must copy a template, replace every
placeholder from contemporaneous evidence, remove only the two template-only
fields, retain failures and deviations, and then run the verifier named by
`python gfs.py studio excellence`.

Human-study records are not valid until the participant has been registered by
the matching `--register` command. The returned `registration_id` must replace
the template placeholder. Comparative-value conditions additionally use the
structured receipt template: remove its two template-only fields, fill the
three frozen-choice answers, hash the exact JSON bytes, and store those bytes
under that digest in the controlled evidence archive. The final analyzer takes
both `--registry` and `--evidence-root`; a syntactically plausible digest with
no matching file is rejected.

The participant case manifest is scoring-key-free. The separate moderator
scoring seal is deliberately excluded from the evidence kit and from every
generated participant packet; the kit verifier checks both properties. Use the
Studio `value-study packet` command or authenticated Web download for delivery,
and never give a participant repository access as a substitute for a blinded
packet. Final analysis receives the seal explicitly with `--scoring-seal`.

The archive contains no credential value or direct participant identifier and
must never be used to collect names, email addresses, access tokens, API keys,
raw free text, or host identifiers. Building or auditing it performs no study,
provider call, match, container execution, training, or formal experiment. A
kit report is preparation evidence only and cannot raise a maturity score.
