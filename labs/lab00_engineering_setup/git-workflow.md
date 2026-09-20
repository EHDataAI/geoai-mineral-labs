# Git and GitHub Workflow

## Purpose

This document describes the Git and GitHub workflow established during LAB 00.

The objective is to move from ad-hoc local editing toward a traceable development process based on branches, validation, signed commits, Pull Requests, review, and controlled integration into `main`.

## Repository model

The primary branch is:

```text
main
```

Development changes are created in dedicated working branches.

Examples used during LAB 00 include:

```text
chore/publication-gate
chore/reproducible-environment
docs/lab00-closure
```

The branch name communicates the purpose of the change.

## Core workflow

The standard development flow is:

```text
main
  ↓
create branch
  ↓
modify files
  ↓
inspect diff
  ↓
quality/security checks
  ↓
stage selected files
  ↓
pre-commit
  ↓
signed commit
  ↓
push branch
  ↓
Pull Request
  ↓
review
  ↓
merge
  ↓
delete branch
  ↓
prune references
  ↓
synchronize main
```

## Start from synchronized `main`

Before starting new work:

```powershell
git switch main
git fetch --prune origin
git pull --ff-only
git status
```

The expected state is a clean working tree synchronized with `origin/main`.

A new branch should normally be created only after this state is confirmed.

## Create a working branch

Example:

```powershell
git switch -c docs/lab00-closure
```

Verify:

```powershell
git branch --show-current
git status
```

## Inspect changes before staging

Useful commands include:

```powershell
git status --short
```

```powershell
git --no-pager diff
```

```powershell
git diff --check
```

`git diff --check` is used to detect whitespace-related problems before commit.

For long terminal output, `--no-pager` avoids entering the Git pager unintentionally.

## Stage intentionally

Files are preferably added explicitly when the change set is small and well defined.

Example:

```powershell
git add environment.yml pyproject.toml
```

Verify:

```powershell
git status --short
```

Staged files appear in the left status column.

For example:

```text
A  environment.yml
M  README.md
```

A status such as:

```text
AM file
```

indicates that the staged version and working-tree version differ.

This situation should be resolved before committing.

## Review staged content

Before commit:

```powershell
git --no-pager diff --cached --stat
```

and:

```powershell
git --no-pager diff --cached
```

The purpose is to verify exactly what will enter Git history.

## Run pre-commit

Run:

```powershell
pre-commit run
```

If a hook modifies a file, the correct workflow is:

```text
hook modifies file
        ↓
review change
        ↓
git add file again
        ↓
pre-commit run again
```

This pattern was encountered during LAB 00 with the `end-of-file-fixer` hook.

Repeatedly attempting to commit without restaging the hook-modified version can produce staged versus unstaged conflicts.

## Secret scan

A manual repository scan can be performed with:

```powershell
gitleaks dir --redact .
```

A clean result should report:

```text
no leaks found
```

## Commit changes

Commits use concise messages describing the intent of the change.

Examples from LAB 00 include:

```text
chore: initialize secure GeoAI laboratory repository
docs: strengthen publication and security documentation
build: add reproducible Python environment
```

Commit:

```powershell
git commit -m "message"
```

Commit signing is enabled automatically.

The signing-key passphrase is entered locally and is never stored in the repository.

## Verify commit signature

After commit:

```powershell
git verify-commit HEAD
```

A valid local signature should report a good Git signature associated with the configured signing identity.

Inspect history:

```powershell
git log --oneline --decorate -5
```

## Push the working branch

First push:

```powershell
git push -u origin <branch-name>
```

The `-u` option configures the upstream tracking branch.

Verify:

```powershell
git branch -vv
```

## Compare branch with `main`

Before opening a Pull Request:

```powershell
git --no-pager diff main...<branch-name> --stat
```

and:

```powershell
git --no-pager log main..<branch-name> --oneline
```

This confirms both the files changed and the commits proposed for integration.

## Create a Pull Request

GitHub CLI can create the Pull Request from the terminal.

Example:

```powershell
gh pr create `
  --base main `
  --head <branch-name> `
  --title "Pull Request title" `
  --body "Pull Request description"
```

The PR description should summarize:

* purpose;
* files changed;
* validation performed;
* security checks;
* scope and limitations.

## Review Pull Request state

Useful commands:

```powershell
gh pr view --json number,title,state,mergeable,mergeStateStatus,changedFiles,additions,deletions,statusCheckRollup
```

```powershell
gh pr diff --name-only
```

```powershell
gh pr checks
```

A PR may legitimately report no CI checks before continuous integration is introduced.

## Verify remote signature

GitHub commit metadata can independently confirm the commit signature.

Conceptually, the expected state is:

```text
verified: true
reason: valid
```

This provides a second verification beyond local `git verify-commit`.

## Merge strategy

LAB 00 uses explicit Pull Request merge commits.

This preserves:

* the development commit;
* the branch relationship;
* the Pull Request integration point;
* an auditable history.

The resulting graph is conceptually:

```text
main ────────┐
             │
             ├── merge commit
             │
branch ──────┘
```

## Merge and delete branch

After review:

```powershell
gh pr merge <PR-number> `
  --merge `
  --delete-branch
```

Where appropriate, the expected head commit can also be matched before merge.

After successful integration, GitHub CLI can remove the merged local and remote branch.

## Synchronize after merge

Run:

```powershell
git fetch --prune origin
```

Verify:

```powershell
git status
git branch -vv
git branch -r
```

Then inspect the graph:

```powershell
git log --graph --oneline --decorate --all -8
```

The expected primary state is:

```text
HEAD -> main
origin/main
```

pointing to the same integrated history.

## Remote-reference pruning

Deleted GitHub branches may temporarily remain as stale local remote references.

Clean them with:

```powershell
git fetch --prune origin
```

This keeps the local repository representation aligned with the remote repository.

## Practices intentionally avoided

The LAB 00 workflow avoids routine use of:

```text
git push --force
git commit --no-verify
hardcoded credentials
direct development on main
unreviewed large commits
```

More advanced Git operations may be appropriate in specific circumstances, but they should be understood before use.

## Commit design

Changes should be grouped by coherent purpose.

LAB 00 intentionally separated:

```text
repository initialization
        ↓
publication/security documentation
        ↓
reproducible environment
        ↓
LAB 00 documentation
```

rather than placing all work into a single large commit.

This makes history easier to review and revert.

## Future CI workflow

Later laboratories can extend the branch workflow with automated checks:

```text
branch
  ↓
push
  ↓
Pull Request
  ↓
CI
  ├── formatting
  ├── linting
  ├── tests
  ├── security scanning
  └── package validation
  ↓
review
  ↓
merge
```

CI should supplement local controls rather than replace them.

## Git workflow principle

The central principle is:

```text
change
  ↓
understand
  ↓
validate
  ↓
sign
  ↓
review
  ↓
merge
```

The objective is not merely to preserve code, but to preserve a trustworthy and understandable history of how the project evolved.
