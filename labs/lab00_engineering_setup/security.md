# Security Controls and Repository Hardening

## Purpose

This document describes the security controls introduced during LAB 00 for the GeoAI Mineral Labs development environment and repository.

The objective is to establish practical defensive controls before scientific datasets, external services, cloud resources, or more complex machine-learning workflows are introduced.

The controls described here focus on:

* workstation exposure reduction;
* credential and secret protection;
* repository hygiene;
* signed and auditable Git history;
* dependency and workflow security;
* preparation for public repository publication.

## Security approach

LAB 00 follows a defense-in-depth approach.

No individual control is assumed to provide complete protection.

Instead, multiple layers are combined:

```text
workstation security
        ↓
network exposure controls
        ↓
credential isolation
        ↓
.gitignore
        ↓
pre-commit
        ↓
Gitleaks
        ↓
signed Git commits
        ↓
Pull Request review
        ↓
GitHub security controls
```

## Workstation baseline

Before publishing repository content, the development workstation was reviewed for unnecessary remote exposure.

The validated baseline included:

* active endpoint protection;
* Windows Firewall enabled;
* inbound traffic blocked by default;
* Windows network profile configured conservatively;
* Remote Desktop disabled when not required;
* Remote Registry disabled;
* WinRM not actively exposed;
* no SSH server required for the development workflow;
* unnecessary remote-access software removed;
* system storage protected with BitLocker;
* SMB1 disabled.

Exact workstation settings may differ on another development machine and are not enforced by this repository.

## Local database exposure

A locally installed PostgreSQL service was identified during workstation review.

The service was hardened so that development access remained available locally while unnecessary network exposure was removed.

The final configuration allowed access through:

```text
127.0.0.1:5432
```

while access through the workstation's LAN address was not available.

The authentication configuration used SCRAM-based password authentication for the active local host rules.

Database credentials and configuration secrets are not stored in this repository.

## GitHub authentication

Git operations use HTTPS rather than SSH authentication.

GitHub CLI authentication is stored through the operating-system credential mechanism.

The workflow intentionally separates:

```text
GitHub authentication
```

from:

```text
Git commit signing
```

This means the SSH key used for commit signatures is not used as the repository authentication mechanism.

## Commit signing

Development commits are signed using a dedicated ED25519 SSH signing key.

Git is configured with:

```text
gpg.format = ssh
commit.gpgsign = true
```

This provides cryptographic evidence linking a commit to the configured signing identity.

Signatures are validated locally using:

```powershell
git verify-commit HEAD
```

and are also independently checked by GitHub.

Validated commits have returned:

```text
verified: true
reason: valid
```

through GitHub's commit metadata.

## Private-key handling

The private SSH signing key must never be:

* committed to Git;
* uploaded to the repository;
* stored in project configuration;
* copied into `.env`;
* included in documentation;
* pasted into issues or pull requests.

Only the public SSH key is registered with GitHub as a signing key.

The key passphrase is also outside repository scope.

## Environment secrets

Real secrets are stored outside Git.

The repository ignores:

```text
.env
.env.*
credentials/
secrets/
*.secret
*.key
*.pem
*.p12
*.pfx
*.ppk
*.jks
*.keystore
```

with an explicit exception for:

```text
.env.example
```

The purpose of `.env.example` is to document required variable names without publishing real values.

## Secret detection

LAB 00 uses Gitleaks as an additional secret-detection layer.

The repository was tested with a deliberately fictitious GitHub token pattern.

The validation demonstrated that Gitleaks:

1. detected the simulated secret;
2. identified the corresponding rule;
3. caused the pre-commit hook to fail;
4. prevented the test content from proceeding through the normal commit workflow;
5. returned to a clean state after the test file was removed.

No real credential was used for this test.

## Gitleaks workflow

Manual directory scan:

```powershell
gitleaks dir --redact .
```

Git-history scan:

```powershell
gitleaks git --redact --verbose
```

The `--redact` option is used so detected secrets are not printed unnecessarily in terminal output.

A successful repository scan should conclude with:

```text
no leaks found
```

## pre-commit security controls

The repository uses pre-commit hooks to reduce common repository risks before commits are accepted.

Relevant controls include:

* private-key detection;
* Gitleaks secret detection;
* oversized-file detection;
* merge-conflict detection;
* YAML validation;
* TOML validation;
* JSON validation;
* filename case-conflict detection;
* end-of-file normalization;
* trailing-whitespace normalization.

A failed hook is corrected rather than bypassed.

The development workflow intentionally avoids:

```text
--no-verify
```

as a routine workaround.

## Large and private data

Raw, intermediate, processed, proprietary, licensed, or private datasets are not committed by default.

The repository ignores:

```text
data/raw/*
data/private/*
data/interim/*
data/processed/*
```

while allowing controlled placeholders and small reproducible public samples where appropriate.

Future datasets should be reviewed for:

* licensing;
* redistribution rights;
* confidentiality;
* personal or sensitive information;
* proprietary geological information;
* file size;
* reproducibility requirements.

## Repository visibility strategy

Development started with the GitHub repository configured as private.

The intended progression is:

```text
local hardening
        ↓
private remote repository
        ↓
history and secret audit
        ↓
documentation review
        ↓
publication gate
        ↓
public repository
```

The repository should not become public simply because the code runs successfully.

Security and publication readiness are separate acceptance criteria.

## Pull Request security review

Changes intended for `main` are introduced through working branches and Pull Requests.

Before merge, applicable checks include:

* expected files only;
* clean working tree;
* pre-commit validation;
* Gitleaks scan;
* signed commit;
* GitHub signature verification;
* review of the branch-to-main diff;
* mergeability check.

This adds a review boundary between development work and the primary branch.

## GitHub security controls

As repository capabilities evolve, the intended GitHub-side controls include:

* signed commit requirements where supported;
* branch or repository rulesets;
* dependency monitoring;
* Dependabot alerts and security updates;
* secret scanning;
* push protection;
* CodeQL or equivalent static analysis;
* minimal GitHub Actions permissions.

Self-hosted GitHub Actions runners are intentionally not required for this laboratory.

Public pull-request code should not execute automatically on the personal workstation.

## Security boundaries

LAB 00 security controls improve the repository development posture but do not make the system immune to compromise.

They do not replace:

* operating-system security updates;
* endpoint protection;
* strong GitHub account authentication;
* credential rotation;
* dependency vulnerability management;
* secure cloud configuration;
* dataset governance;
* backup practices;
* incident response.

Security remains an ongoing engineering process.

## Security principle

The LAB 00 security model can be summarized as:

```text
do not commit the secret
        ↓
detect accidental exposure
        ↓
block unsafe commits
        ↓
sign legitimate changes
        ↓
review before merge
        ↓
verify again remotely
```

Prevention is preferred over attempting to remove secrets after publication.
