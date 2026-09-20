# Workstation and Development Environment

## Purpose

This document describes the workstation foundation used to develop GeoAI Mineral Labs during LAB 00.

The objective is not to reproduce every operating-system package or local application. Instead, it documents the tools and engineering decisions required to establish a stable development environment for subsequent GeoAI laboratories.

## Validated workstation

The initial development workstation was validated with the following configuration:

| Component        | Validated configuration |
| ---------------- | ----------------------- |
| Operating system | Windows 11 x64          |
| Architecture     | AMD64 / 64-bit          |
| Shell            | Windows PowerShell 5.1  |
| Git              | 2.53.0.windows.2        |
| GitHub CLI       | 2.101.0                 |
| Conda            | 26.1.1                  |
| Python           | 3.12.14                 |
| VS Code          | x64                     |
| pre-commit       | 4.6.2                   |
| Gitleaks         | 8.30.0                  |

Hardware-specific configuration is intentionally kept separate from project dependencies.

## Project workspace

The local project workspace is:

```text
C:\GeoAI\geoai-mineral-labs
```

The dedicated Conda environment is:

```text
geoai-mineral
```

The Python interpreter used by the project is isolated from the Miniconda base environment.

## Python isolation strategy

A dedicated Conda environment is used instead of installing GeoAI dependencies directly into `base`.

This provides:

* dependency isolation;
* reduced cross-project interference;
* easier troubleshooting;
* better reproducibility;
* clearer environment ownership.

The environment uses Python 3.12 because it provides broad compatibility with current scientific, geospatial, machine-learning, and deep-learning tooling while avoiding unnecessary dependence on newer Python versions before their ecosystem support is fully established.

## Conda configuration

The workstation-level Conda configuration is not treated as the source of truth for this repository.

The project instead declares its channels directly inside:

```text
environment.yml
```

using:

```yaml
channels:
  - conda-forge
  - nodefaults
```

This prevents the project's dependency resolution from depending implicitly on the developer's global channel configuration.

## System tools versus project dependencies

LAB 00 intentionally separates workstation tools from the Python environment.

### Workstation tools

The following tools are installed at operating-system level:

* Git;
* GitHub CLI;
* Visual Studio Code;
* Gitleaks.

These tools are not declared as Python dependencies.

### Conda environment

The environment initially contains only the minimum Python development requirements:

* Python 3.12;
* pip;
* pre-commit.

Scientific and GeoAI dependencies are added incrementally as later laboratories require them.

## Git configuration

Git uses:

```text
main
```

as the primary branch.

Repository operations use HTTPS authentication through GitHub CLI and the operating-system credential store.

Commit identity is separated from authentication credentials.

Development commits are cryptographically signed using a dedicated SSH ED25519 signing key.

## Line-ending strategy

Development occurs on Windows, while repository contents should remain portable across operating systems.

The repository therefore uses `.gitattributes` to normalize line endings.

Some Git commands may display informational messages such as:

```text
LF will be replaced by CRLF
```

or:

```text
CRLF will be replaced by LF
```

These messages do not necessarily indicate an error. Repository consistency is controlled through `.gitattributes` and pre-commit validation.

## Text encoding

Project text files should use UTF-8 without unnecessary byte-order marks.

During LAB 00, Windows PowerShell 5.1 was observed to introduce UTF-8 BOM characters in some file-writing workflows.

For this reason:

* source and configuration files are validated after automated writes;
* editors such as Visual Studio Code are preferred for normal editing;
* generated files are checked before commit;
* pre-commit hooks normalize end-of-file formatting.

## VS Code

Visual Studio Code is used as the primary development editor.

The repository does not version developer-specific `.vscode/` settings at this stage.

This avoids coupling the repository to workstation-specific configuration until shared editor settings provide a clear project benefit.

## Environment activation

From PowerShell:

```powershell
cd C:\GeoAI\geoai-mineral-labs
conda activate geoai-mineral
```

Basic validation:

```powershell
python --version
git status
```

The expected Python major/minor version is:

```text
Python 3.12
```

## Design principle

The workstation is infrastructure for the project, but it is not the project itself.

GeoAI Mineral Labs therefore aims to keep:

```text
workstation configuration
        ↓
project environment
        ↓
Python package
        ↓
laboratory-specific dependencies
```

as separate layers.

This separation reduces hidden dependencies and makes later laboratories easier to reproduce on another machine.
