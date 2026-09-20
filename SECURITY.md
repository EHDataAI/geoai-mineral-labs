# Security Policy

## Scope

GeoAI Mineral Labs is an evolving educational, experimental, and technical portfolio focused on GeoAI and mineral exploration.

The repository is not currently intended to operate as a production mineral exploration platform.

Security reports related to the current `main` branch are nevertheless taken seriously, particularly when they concern:

* exposed credentials or secrets;
* insecure dependency configurations;
* unsafe automation or workflow permissions;
* accidental publication of private or sensitive data;
* vulnerabilities in repository code;
* unsafe handling of external data or services.

## Reporting a Vulnerability

Please do not disclose security vulnerabilities, credentials, tokens, private keys, or other sensitive information through a public GitHub issue, discussion, or pull request.

For security vulnerabilities, use GitHub's private vulnerability reporting or repository security advisory mechanisms when they are available for this repository.

A useful report should include, when possible:

* a concise description of the issue;
* the affected file, component, or laboratory;
* steps required to reproduce the issue;
* the potential security impact;
* relevant environment information;
* suggested remediation, if known.

Do not include real credentials or unrelated private information in a report.

## Secret Exposure

Real credentials, passwords, API keys, access tokens, signing keys, certificates, and private datasets must never be committed to this repository.

If a real secret is accidentally exposed, it should be considered compromised.

The appropriate response is to:

1. revoke or rotate the affected credential;
2. remove the secret from the repository;
3. investigate whether it exists elsewhere in Git history or artifacts;
4. review any relevant access or audit logs;
5. document the remediation without republishing the secret.

Removing a credential from the latest commit alone does not make that credential safe again.

## Repository Security Controls

The repository progressively applies security controls including:

* explicit `.gitignore` policies;
* isolated development environments;
* pre-commit validation;
* secret detection with Gitleaks;
* private-key detection;
* signed Git commits;
* dependency monitoring;
* restricted workflow permissions;
* branch and repository protection controls where available.

Security controls may evolve as additional laboratories and infrastructure are introduced.

## Data Security

Large, private, licensed, proprietary, or sensitive datasets should not be committed directly to the repository.

Only datasets or samples that are suitable for redistribution should be versioned publicly.

Future laboratories using external datasets should document their origin, licensing constraints, and handling requirements.

## Responsible Disclosure

Please allow reasonable time for investigation and remediation before publicly disclosing a reported security issue.

This policy may be updated as the repository evolves from laboratory work toward more advanced GeoAI and ML engineering workflows.
