# Reproducibility Guide

## Purpose

This document describes how the initial GeoAI Mineral Labs development environment can be recreated and validated from the repository.

Reproducibility in LAB 00 means that a clean environment can be constructed from repository-controlled configuration and that the Python project can be installed and imported successfully.

## Prerequisites

The workstation should provide:

* Git;
* Conda or a compatible Conda distribution;
* access to the repository.

GitHub CLI, Visual Studio Code, and Gitleaks are development tools but are not required simply to recreate the Python environment.

## Clone the repository

Example:

```powershell
git clone https://github.com/ehtestinglab-hub/geoai-mineral-labs.git
cd geoai-mineral-labs
```

Repository visibility may affect whether authentication is required.

## Create the Conda environment

The environment definition is stored in:

```text
environment.yml
```

Create the environment with:

```powershell
conda env create `
  --file environment.yml `
  -y
```

The declared environment name is:

```text
geoai-mineral
```

For an independent validation environment, the name can be overridden:

```powershell
conda env create `
  --name geoai-mineral-repro-test `
  --file environment.yml `
  -y
```

This procedure was validated successfully during LAB 00.

## Activate the environment

```powershell
conda activate geoai-mineral
```

Verify Python:

```powershell
python --version
```

LAB 00 was validated with Python 3.12.

## Project installation

The repository uses a `src` layout.

Install the project in editable mode:

```powershell
python -m pip install `
  --editable . `
  --no-deps `
  --no-build-isolation
```

Editable installation allows development changes under:

```text
src/geoai/
```

to be reflected without repeatedly rebuilding and reinstalling the package.

## Validate package import

Run:

```powershell
python -c "import geoai; print(geoai.__file__)"
```

The module should resolve to the repository's source tree, conceptually:

```text
...\geoai-mineral-labs\src\geoai\__init__.py
```

Inspect installed package metadata with:

```powershell
python -m pip show geoai-mineral-labs
```

The distribution name is:

```text
geoai-mineral-labs
```

while the importable Python package is:

```python
import geoai
```

## Validate configuration files

YAML:

```powershell
pre-commit run check-yaml --files environment.yml
```

TOML:

```powershell
pre-commit run check-toml --files pyproject.toml
```

Python can also independently parse the TOML configuration:

```powershell
python -c "import tomllib, pathlib; tomllib.loads(pathlib.Path('pyproject.toml').read_text(encoding='utf-8')); print('pyproject.toml OK')"
```

## Environment resolution test

Before creating an environment, dependency resolution can be inspected using:

```powershell
conda env create `
  --file environment.yml `
  --dry-run
```

The LAB 00 environment is intentionally configured to use:

```text
conda-forge
```

with:

```text
nodefaults
```

to avoid implicit dependence on workstation-level Conda channel configuration.

## Clean-environment validation

LAB 00 performed an independent reconstruction test using:

```powershell
conda env create `
  --name geoai-mineral-repro-test `
  --file environment.yml `
  -y
```

The following checks were then performed without replacing the primary development environment:

```powershell
conda run -n geoai-mineral-repro-test python --version
```

```powershell
conda run -n geoai-mineral-repro-test pre-commit --version
```

The repository was installed into that clean environment:

```powershell
conda run -n geoai-mineral-repro-test `
  python -m pip install `
  --editable . `
  --no-deps `
  --no-build-isolation
```

Import validation:

```powershell
conda run -n geoai-mineral-repro-test `
  python -c "import geoai; print(geoai.__file__)"
```

The test successfully resolved the package from:

```text
src/geoai/__init__.py
```

## Remove validation environment

After successful validation:

```powershell
conda env remove `
  --name geoai-mineral-repro-test `
  -y
```

Confirm removal:

```powershell
conda env list
```

Temporary QA environments should not be retained unnecessarily.

## Quality controls before commit

Before integrating environment changes:

```powershell
git diff --check
```

```powershell
pre-commit run --all-files
```

```powershell
gitleaks dir --redact .
```

Changes are committed only after the applicable checks pass.

## Reproducibility boundaries

`environment.yml` provides a portable project environment specification.

It is not intended to reproduce:

* Windows itself;
* GPU drivers;
* Visual Studio Code installation;
* Git installation;
* GitHub CLI installation;
* workstation-specific credentials;
* private datasets;
* external services.

Those concerns are documented separately.

## Future evolution

Later GeoAI laboratories will introduce scientific dependencies only when required.

Potential future categories include:

* numerical computing;
* tabular data analysis;
* geospatial vector processing;
* raster processing;
* spatial statistics;
* machine learning;
* remote sensing;
* deep learning;
* graph learning.

Each addition should be justified by a concrete laboratory requirement and validated before integration.

## Reproducibility principle

The intended progression is:

```text
repository
    ↓
environment.yml
    ↓
clean Conda environment
    ↓
pyproject.toml
    ↓
editable project installation
    ↓
import validation
    ↓
quality and security checks
```

A successful model is not considered reproducible if its software environment cannot also be reconstructed.
