# GeoAI Mineral Labs — Publication Manifest

> **Classification: INTERNAL**
>
> This manifest belongs to the private canonical repository. It tracks publication decisions and must not be copied automatically into the public portfolio.

## Purpose

This document records the publication classification of relevant GeoAI Mineral Labs artifacts.

It acts as the operational companion to `PUBLICATION_POLICY.md`.

The manifest should evolve as new laboratories, datasets, models, figures, notebooks, and software components are created.

## Classification

Allowed values are:

* `PUBLIC`
* `INTERNAL`
* `RESERVED`
* `SENSITIVE`

The default classification for a new artifact is `INTERNAL`.

## Publication states

A classification is different from publication status.

Suggested publication states are:

* `NOT_REVIEWED`
* `HOLD`
* `APPROVED`
* `PUBLISHED`
* `REJECTED`

An artifact classified as PUBLIC is not published until its publication state becomes `APPROVED`.

## Current manifest

| Artifact                           | Classification | Publication status | Intended public form               | Notes                                                                |
| ---------------------------------- | -------------- | ------------------ | ---------------------------------- | -------------------------------------------------------------------- |
| Root README                        | PUBLIC         | HOLD               | Curated portfolio README           | Private repository version may differ from future public version     |
| LAB 00 overview                    | PUBLIC         | HOLD               | Simplified engineering summary     | Demonstrates reproducibility and software-engineering foundation     |
| `environment.yml` baseline         | PUBLIC         | HOLD               | Public reproducible environment    | Review dependencies before portfolio export                          |
| `pyproject.toml` baseline          | PUBLIC         | HOLD               | Public package metadata            | Adapt package name for showcase if required                          |
| General Git workflow               | PUBLIC         | HOLD               | Simplified documentation           | Remove unnecessary workstation-specific detail                       |
| Reproducibility methodology        | PUBLIC         | HOLD               | Portfolio documentation            | Suitable for demonstration                                           |
| General security practices         | PUBLIC         | HOLD               | Simplified security section        | Internal security configuration remains private                      |
| Workstation-specific configuration | INTERNAL       | HOLD               | Summary only                       | Public portfolio does not require detailed workstation configuration |
| Publication policy                 | INTERNAL       | REJECTED           | None                               | Must remain in canonical repository                                  |
| Publication manifest               | INTERNAL       | REJECTED           | None                               | Must remain in canonical repository                                  |
| Private `.env` files               | SENSITIVE      | REJECTED           | None                               | Never publish                                                        |
| Credentials and tokens             | SENSITIVE      | REJECTED           | None                               | Never commit or publish                                              |
| Private keys                       | SENSITIVE      | REJECTED           | None                               | Never commit or publish                                              |
| Private/proprietary datasets       | SENSITIVE      | REJECTED           | None                               | Use public or synthetic alternatives                                 |
| Standard geospatial foundations    | PUBLIC         | NOT_REVIEWED       | LAB 01 portfolio material          | Subject to LAB 01 completion                                         |
| Standard spatial EDA               | PUBLIC         | NOT_REVIEWED       | LAB 02 portfolio material          | Subject to LAB 02 completion                                         |
| Standard spatial cross-validation  | PUBLIC         | NOT_REVIEWED       | LAB 03 portfolio material          | Key portfolio capability                                             |
| Spatial leakage demonstration      | PUBLIC         | NOT_REVIEWED       | Simplified demonstration           | Show concept without advanced internal extensions                    |
| Generic ML baselines               | PUBLIC         | NOT_REVIEWED       | Portfolio benchmark                | Keep models interpretable and reproducible                           |
| Advanced feature engineering       | RESERVED       | HOLD               | High-level mention only            | Do not expose implementation by default                              |
| Advanced prospectivity ensembles   | RESERVED       | HOLD               | High-level mention only            | Potential competitive value                                          |
| Advanced Bayesian strategies       | RESERVED       | HOLD               | High-level mention only            | Evaluate case by case                                                |
| Advanced copula-based methods      | RESERVED       | HOLD               | High-level mention only            | Evaluate case by case                                                |
| Graph GeoAI research               | RESERVED       | HOLD               | Selected simplified demo only      | Full experimental architecture remains private                       |
| Multimodal GeoAI research          | RESERVED       | HOLD               | Selected simplified demo only      | Full implementation remains private                                  |
| Advanced uncertainty methods       | RESERVED       | HOLD               | Standard public approximation only | Preserve differentiating implementation                              |
| Internal optimization heuristics   | RESERVED       | REJECTED           | None                               | Private competitive knowledge                                        |
| Failed/abandoned experiments       | INTERNAL       | REJECTED           | None                               | Useful internally, not portfolio material                            |
| Debug notebooks                    | INTERNAL       | REJECTED           | None                               | Never export directly                                                |
| Research notebooks                 | INTERNAL       | HOLD               | Curated derivative only            | Public notebook must be recreated intentionally                      |

## Planned public portfolio threshold

The first public portfolio repository should not be created solely from LAB 00.

The preferred initial publication threshold is completion of:

* LAB 01 — Geospatial Python Foundations;
* LAB 02 — Spatial EDA;
* LAB 03 — Spatial Cross-Validation.

This provides enough technical evidence to demonstrate actual GeoAI capability rather than only repository engineering.

## Planned first public release

Tentative portfolio milestone:

`v0.1.0`

Expected public content may include:

* professional project README;
* project architecture;
* selected LAB 01 material;
* selected LAB 02 material;
* selected LAB 03 material;
* public/synthetic sample data;
* reproducible environment;
* reusable generic source code;
* tests;
* selected figures and maps;
* `CITATION.cff`;
* appropriate license.

The actual content must still pass the Publication Gate.

## Publication review template

When evaluating a new artifact, add or update a manifest row and record:

```text
Artifact:
LAB:
Classification:
Publication status:
Data source:
Dataset license:
Third-party dependencies:
Security review:
IP review:
Public simplification required:
Approved public destination:
Notes:
```

## Portfolio extraction rule

No artifact should be transferred to the public portfolio merely because it exists in the canonical repository.

The extraction process must be:

```text
identify
    ↓
classify
    ↓
review
    ↓
simplify if needed
    ↓
sanitize
    ↓
reproduce
    ↓
approve
    ↓
copy into clean public repository
```

## Review cadence

The manifest should be reviewed:

* when a laboratory is completed;
* before a GitHub release;
* before adding a new dataset;
* before publishing advanced methods;
* whenever licensing conditions change;
* whenever an artifact changes from generic demonstration to potentially differentiating technology.

## Current strategic position

The project currently follows this publication posture:

```text
canonical development     PRIVATE
advanced research         RESERVED
sensitive material        NEVER PUBLIC
generic technical skills  CANDIDATE FOR PUBLICATION
public portfolio          NOT YET CREATED
```

LAB 00 is complete.

The next development milestone is:

**LAB 01 — Geospatial Python Foundations**
