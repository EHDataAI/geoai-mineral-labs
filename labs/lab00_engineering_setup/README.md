# LAB 00 — Engineering & GeoAI Environment

## Overview

LAB 00 establishes the engineering foundation for the GeoAI Mineral Labs repository.

The objective of this laboratory is not to build a predictive mineral exploration model. Instead, it prepares a reproducible, secure, and maintainable environment in which the subsequent geospatial, machine-learning, and deep-learning laboratories can be developed.

The laboratory covers workstation validation, Python environment isolation, repository engineering, Git and GitHub workflow, secret protection, commit signing, packaging, and reproducibility testing.

## Objectives

The main objectives of LAB 00 are to:

* create an isolated Python environment for GeoAI development;
* establish a professional Git repository structure;
* separate source code, data, notebooks, configurations, reports, and tests;
* prevent credentials and private data from entering version control;
* introduce automated pre-commit quality and security controls;
* cryptographically sign Git commits;
* establish a branch and pull-request workflow;
* define a reproducible Conda environment;
* configure the Python project using `pyproject.toml`;
* validate that the project can be recreated and installed in a clean environment.

## Validated development environment

The LAB 00 environment was validated with:

| Component        | Validated configuration |
| ---------------- | ----------------------- |
| Operating system | Windows 11 x64          |
| Shell            | Windows PowerShell 5.1  |
| Git              | 2.53.0.windows.2        |
| GitHub CLI       | 2.101.0                 |
| Conda            | 26.1.1                  |
| Python           | 3.12.14                 |
| pre-commit       | 4.6.2                   |
| Gitleaks         | 8.30.0                  |
| IDE              | Visual Studio Code      |
| Git protocol     | HTTPS                   |
| Commit signing   | SSH / ED25519           |

Exact package versions may evolve over time. Reproducible project requirements are defined by the repository configuration rather than by this table alone.

## Repository architecture

The initial repository structure separates development concerns:

```text
geoai-mineral-labs/
├── configs/
├── data/
│   ├── raw/
│   ├── interim/
│   ├── processed/
│   └── sample/
├── labs/
│   └── lab00_engineering_setup/
├── notebooks/
├── references/
├── reports/
├── src/
│   └── geoai/
├── tests/
├── .env.example
├── .gitattributes
├── .gitignore
├── .pre-commit-config.yaml
├── environment.yml
├── pyproject.toml
├── README.md
└── SECURITY.md
```

Large, private, licensed, or sensitive data are intentionally kept outside version control.

## Environment strategy

The project uses an isolated Conda environment named:

```text
geoai-mineral
```

The reproducible environment definition is stored in:

```text
environment.yml
```

The environment intentionally uses:

```yaml
channels:
  - conda-forge
  - nodefaults
```

This keeps the project environment independent from the workstation's global Conda channel configuration and avoids unintentionally mixing package ecosystems.

Only the minimum environment required for the current development stage is declared.

Scientific and GeoAI dependencies are introduced incrementally when a laboratory has a concrete requirement for them.

## Python project configuration

Python packaging metadata is defined in:

```text
pyproject.toml
```

The project uses a `src` layout:

```text
src/
└── geoai/
    └── __init__.py
```

The distribution is named:

```text
geoai-mineral-labs
```

while the Python package is imported as:

```python
import geoai
```

The project was successfully installed in editable mode using:

```powershell
python -m pip install --editable . --no-deps --no-build-isolation
```

and validated by importing the package from:

```text
src/geoai/
```

## Reproducibility validation

The environment specification was tested beyond a simple syntax check.

A temporary clean Conda environment was created from `environment.yml`:

```powershell
conda env create `
  --name geoai-mineral-repro-test `
  --file environment.yml `
  -y
```

The validation confirmed:

* successful environment resolution;
* packages obtained from `conda-forge`;
* Python 3.12 availability;
* pre-commit availability;
* successful editable installation of the project;
* successful `import geoai`;
* correct resolution to `src/geoai/__init__.py`.

The temporary environment was deleted after validation.

This provides evidence that the repository can be recreated independently of the original development environment.

## Security controls

LAB 00 introduces several defensive controls before the repository is made public.

### Secret protection

Sensitive local configuration is excluded using `.gitignore`.

Examples include:

```text
.env
credentials/
secrets/
private keys
certificates
private datasets
```

A safe `.env.example` is versioned instead of real environment secrets.

### Gitleaks

Gitleaks is used to detect potentially exposed credentials and tokens.

A controlled test using a fictitious GitHub token pattern confirmed that the scanner:

1. detected the simulated secret;
2. blocked it through the pre-commit workflow;
3. returned to a clean state after the test file was removed.

### pre-commit

The repository uses pre-commit hooks for checks including:

* trailing whitespace;
* end-of-file normalization;
* YAML syntax;
* JSON syntax;
* TOML syntax;
* merge conflicts;
* filename case conflicts;
* private-key detection;
* oversized files;
* Gitleaks secret detection.

### Signed commits

Git commits are signed using a dedicated ED25519 SSH signing key.

Signatures were validated locally with:

```powershell
git verify-commit HEAD
```

and independently verified by GitHub.

## Git and GitHub workflow

Development follows a branch-based workflow:

```text
main
  │
  ├── working branch
  │      │
  │      ├── changes
  │      ├── validation
  │      ├── signed commit
  │      └── push
  │
  └── Pull Request
         │
         ├── review
         ├── verification
         └── merge into main
```

During LAB 00, this workflow was exercised using real pull requests rather than committing all changes directly to `main`.

The process included:

* local feature or maintenance branches;
* signed commits;
* remote branch publication;
* pull-request inspection;
* verification of changed files;
* GitHub signature validation;
* merge into `main`;
* deletion of merged branches;
* pruning of obsolete remote references.

## Quality gates

A change is considered ready for integration only when applicable controls pass.

The LAB 00 workflow includes:

```text
source change
    ↓
git diff review
    ↓
syntax validation
    ↓
pre-commit
    ↓
Gitleaks
    ↓
signed commit
    ↓
push
    ↓
Pull Request
    ↓
review
    ↓
merge
```

Failures are corrected rather than bypassed with options such as `--no-verify`.

## Engineering principles established

LAB 00 establishes the following principles for subsequent laboratories:

* reproducibility before model complexity;
* isolated environments;
* minimum necessary dependencies;
* version-controlled configuration;
* no credentials in source control;
* signed and auditable changes;
* branch-based development;
* validation before merge;
* reusable Python code under `src/`;
* notebooks for exploration rather than as the only implementation layer;
* incremental introduction of scientific dependencies.

## Completion criteria

LAB 00 is considered complete when the following evidence is available:

* [x] isolated Conda environment;
* [x] Python 3.12 validated;
* [x] repository architecture created;
* [x] `.gitignore` and `.gitattributes` configured;
* [x] secret detection tested;
* [x] pre-commit hooks operational;
* [x] SSH commit signing operational;
* [x] GitHub signature verification operational;
* [x] branch and pull-request workflow exercised;
* [x] `environment.yml` created;
* [x] `pyproject.toml` created;
* [x] editable package installation validated;
* [x] clean environment recreation validated;
- [x] detailed workstation documentation;
- [x] detailed security documentation;
- [x] reproducibility guide;
- [x] Git workflow guide;
- [x] final LAB 00 documentation review.

## Next steps

LAB 00 is complete.

The repository now has the engineering, security, reproducibility, packaging, and Git/GitHub workflow foundations required for the scientific laboratories.

Development will continue with:

**LAB 01 — Geospatial Python Foundations**

LAB 01 will introduce the first scientific and geospatial dependencies based on explicit analytical requirements.
