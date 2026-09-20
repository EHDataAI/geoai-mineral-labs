# GeoAI Mineral Labs — Publication Policy

> **Classification: INTERNAL**
>
> This document belongs to the private canonical repository and must not be copied automatically into any public portfolio repository.

## Purpose

This policy defines how knowledge, code, data, experiments, documentation, and results developed inside GeoAI Mineral Labs are classified before any material is published externally.

The project follows a deliberate separation between:

* private research and development;
* reusable public demonstrations;
* reserved technical knowledge;
* sensitive information that must never be published.

The objective is to demonstrate professional GeoAI, data science, machine learning, software engineering, and reproducibility skills without disclosing unnecessary technical depth, proprietary ideas, sensitive information, or potential competitive advantages.

## Repository model

GeoAI Mineral Labs uses two conceptual layers.

### Private canonical repository

`geoai-mineral-labs`

This repository is the source of truth for development.

It may contain:

* exploratory work;
* research notebooks;
* incomplete experiments;
* alternative implementations;
* advanced models;
* internal benchmarking;
* reserved feature engineering;
* private publication assessments;
* failed approaches;
* experimental architectures;
* technical notes;
* material that is not suitable for public release.

The canonical repository remains private.

### Public portfolio repository

A separate public showcase repository may be created when sufficient validated GeoAI content exists.

The public repository must contain only artifacts that have passed the Publication Gate defined in this document.

The public repository is not a mirror, clone, synchronization target, or complete representation of the private canonical repository.

## Core principle

The publication principle is:

> **Publish evidence of capability. Reserve competitive advantage.**

Public material should demonstrate that the project author understands and can correctly implement professional methods.

Public material does not need to disclose every implementation detail, experimental path, optimization, model combination, or advanced research extension used internally.

## Default classification rule

All new project material is considered:

`INTERNAL`

until it has been explicitly reviewed and assigned another classification.

Nothing becomes public merely because:

* it works;
* it is committed to Git;
* a laboratory is complete;
* it uses public data;
* similar ideas exist elsewhere.

Publication requires a separate decision.

## Classification levels

Every candidate artifact should be classified using one of the following categories.

### PUBLIC

Material intentionally approved for external publication.

Typical examples:

* standard geospatial workflows;
* CRS and projection examples;
* vector and raster processing;
* public-data ingestion examples;
* spatial exploratory data analysis;
* standard spatial statistics;
* standard machine-learning baselines;
* spatial cross-validation demonstrations;
* reproducibility examples;
* testing patterns;
* software-engineering practices;
* documentation intended for recruiters;
* small synthetic or redistribution-safe sample datasets;
* figures and maps approved for portfolio use.

PUBLIC means that the artifact may be considered for the external showcase.

It does not mean that publication is automatic.

### INTERNAL

Normal private development material.

Typical examples:

* exploratory notebooks;
* intermediate analyses;
* debugging outputs;
* alternative model configurations;
* parameter searches;
* incomplete documentation;
* temporary scripts;
* experimental branches;
* research notes;
* benchmarking results;
* implementation details not required by the public demonstration.

INTERNAL material remains in the canonical repository unless explicitly reclassified.

### RESERVED

Material that may represent technical differentiation, advanced research, future commercial value, or unnecessary disclosure of project know-how.

Examples may include:

* proprietary or original feature engineering;
* advanced prospectivity-ranking logic;
* sophisticated model ensembles;
* novel combinations of statistical and machine-learning techniques;
* internal Bayesian modelling strategies;
* advanced copula-based approaches;
* experimental Graph GeoAI methods;
* advanced multimodal fusion;
* proprietary uncertainty methods;
* optimization heuristics;
* advanced geological scoring logic;
* internal production architectures;
* technical methods that provide a meaningful competitive advantage.

RESERVED material must not be transferred into the public portfolio without a deliberate reclassification decision.

A public artifact may state that advanced research extensions exist without revealing their implementation.

### SENSITIVE

Material that must not be committed to public repositories and may require stronger controls even inside the private project.

Examples include:

* passwords;
* API keys;
* access tokens;
* private signing keys;
* confidential client information;
* proprietary exploration datasets;
* personal data;
* restricted geoscientific information;
* private contractual material;
* credentials;
* infrastructure secrets;
* protected intellectual property belonging to third parties.

SENSITIVE material must never enter the public portfolio.

Credentials and secrets should generally remain outside Git entirely.

## Publication Gate

Completion of a laboratory and approval for publication are separate events.

The workflow is:

```text
development
    ↓
technical validation
    ↓
LAB completed in private repository
    ↓
publication assessment
    ↓
classification review
    ↓
sanitization
    ↓
reproducibility test
    ↓
security and licensing review
    ↓
PUBLIC or RESERVED
```

A laboratory may therefore be complete while remaining entirely private.

## Public artifact requirements

Before publication, an artifact should satisfy the applicable criteria below.

### Technical quality

The artifact should:

* run as documented;
* use a reproducible environment;
* use understandable naming;
* contain sufficient explanation;
* avoid unnecessary experimental debris;
* distinguish exploratory analysis from validated conclusions;
* use appropriate evaluation methodology.

### Scientific quality

Where applicable, the artifact should demonstrate:

* correct handling of spatial structure;
* awareness of spatial autocorrelation;
* appropriate train/test separation;
* leakage prevention;
* appropriate baselines;
* transparent metrics;
* uncertainty where relevant;
* limitations and assumptions.

### Security

The artifact must not expose:

* credentials;
* access tokens;
* private keys;
* local secrets;
* private network information;
* confidential data;
* restricted system information.

Security scans should be performed before publication.

### Data governance

Any dataset included or referenced publicly must be reviewed for:

* license;
* redistribution rights;
* confidentiality;
* attribution requirements;
* personal or sensitive information;
* proprietary geological information.

Synthetic data should be preferred when it can demonstrate the same technical capability without exposing private information.

### Intellectual-property review

Before publication, determine whether the artifact contains:

* original methods with possible commercial value;
* potentially patentable concepts;
* unpublished research;
* proprietary implementation logic;
* third-party intellectual property;
* licensed content incompatible with redistribution.

When uncertain, keep the artifact INTERNAL or RESERVED until reviewed.

## Public simplification

Public demonstrations may intentionally use simpler implementations than their private research equivalents.

For example:

```text
PRIVATE
advanced ensemble
+ proprietary feature engineering
+ optimized ranking
+ experimental uncertainty model

PUBLIC
standard baseline models
+ documented spatial validation
+ generic features
+ interpretable metrics
```

The public version must remain technically correct.

It must not misrepresent a simplified demonstration as the complete internal implementation.

## Research notebook versus portfolio notebook

Private notebooks may contain:

* experimentation;
* dead ends;
* debugging;
* multiple competing approaches;
* advanced methods;
* internal observations.

Public notebooks should instead be:

* focused;
* readable;
* reproducible;
* pedagogical;
* intentionally scoped;
* free of unnecessary internal details.

Therefore:

`research notebook != portfolio notebook`

Public notebooks should normally be produced as curated artifacts rather than copied directly from research notebooks.

## No automatic repository mirroring

The private repository must never be automatically mirrored to the public repository.

Publication should use an explicit selection process.

The public repository should be constructed from approved artifacts only.

This prevents accidental disclosure through:

* Git history;
* deleted files;
* experimental branches;
* old credentials;
* abandoned notebooks;
* internal documentation;
* reserved implementations.

## Licensing strategy

The canonical private repository does not require an open-source license for internal development.

The future public portfolio may use a permissive license, such as MIT, only for material deliberately approved for public reuse.

Applying a public license is itself part of the Publication Gate.

Publishing a repository and licensing an artifact are related but separate decisions.

## Attribution strategy

For important public releases, the project may use:

* signed Git commits;
* GitHub releases;
* version tags;
* `CITATION.cff`;
* release notes;
* Zenodo DOI for stable research-oriented releases.

These mechanisms improve attribution, provenance, and reproducibility.

## Public contribution strategy

The public portfolio is primarily intended for professional review and demonstration.

At initial publication, contribution mechanisms may remain limited or disabled, including:

* Issues;
* Discussions;
* Pull Requests;
* Wiki;
* Projects.

The decision to enable community contributions can be reconsidered separately.

Open-source licensing does not require an open contribution workflow.

## Review responsibility

Before publication, each artifact should answer:

1. What capability does this demonstrate?
2. Does a recruiter need this level of technical detail?
3. Is the method generic or differentiating?
4. Does it expose private or restricted information?
5. Can the same capability be demonstrated with a simpler implementation?
6. Are all datasets legally publishable?
7. Is attribution sufficient?
8. Is the artifact reproducible?
9. Has it passed security scanning?
10. Would publication reduce a meaningful future competitive advantage?

If the answer to the final question is uncertain, the artifact remains private.

## Guiding principle

GeoAI Mineral Labs is designed to maximize learning and technical depth privately while publishing a carefully selected professional representation of that work.

The long-term objective is:

```text
learn deeply
    ↓
experiment privately
    ↓
validate rigorously
    ↓
classify deliberately
    ↓
publish selectively
    ↓
preserve advanced knowledge
```
