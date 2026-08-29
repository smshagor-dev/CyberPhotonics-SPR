# Security Policy

CyberPhotonics-SPR is research software that combines scientific simulation, machine-learning workflows, local desktop tooling, model export, and optional hardware/COMSOL integrations. Security reports are welcome and should be handled separately from ordinary bug reports.

## Supported versions

| Version | Security support |
| --- | --- |
| `main` | Supported |
| `1.0.0rc1` | Supported while it is the current release candidate |
| Older snapshots / unmaintained forks | Not supported |

Security fixes are developed against `main` first and are backported only when a maintained release requires it.

## Reporting a vulnerability

Please **do not publish exploitable details in a public issue, discussion, pull request, benchmark artifact, or dataset**.

Preferred reporting path:

1. Use GitHub Private Vulnerability Reporting / a private Security Advisory for this repository when that option is available.
2. If private reporting is not available, open a minimal public issue titled `Security contact request` without technical details, proof-of-concept code, credentials, or affected data. A maintainer can then establish an appropriate private channel.

Include, when possible:

- affected version or commit SHA;
- operating system and Python version;
- vulnerable component and attack preconditions;
- concise reproduction steps;
- expected security impact;
- whether credentials, private research data, hardware, or external services are involved;
- suggested remediation if known.

The project aims to acknowledge well-formed reports within 72 hours, then communicates severity, remediation status, and coordinated disclosure timing as the investigation progresses. Complex hardware or scientific-toolchain issues can require additional validation time.

## Security scope

Examples of issues that should be reported privately include:

- arbitrary command or code execution;
- path traversal or unintended file overwrite/read behavior;
- unsafe deserialization or malicious model/artifact loading;
- dependency vulnerabilities with a credible impact on this project;
- credential, token, environment, or research-data exposure;
- unsafe serial/socket/HIL input handling that crosses the documented trust boundary;
- artifact, evidence-registry, checksum, or provenance bypasses that could allow tampered evidence to be accepted;
- privilege escalation or sandbox/permission bypasses in supported workflows.

The following are normally **not** security vulnerabilities by themselves:

- missing optional scientific dependencies;
- numerical disagreement, model accuracy concerns, or research-methodology disputes;
- intentionally documented synthetic-vs-physical evidence boundaries;
- unsupported third-party forks or modified environments;
- denial of service that requires a user to intentionally launch an extremely large local research job with trusted input.

## Research and evidence integrity

Security fixes must not silently weaken scientific provenance. Changes that affect evidence hashing, dataset lineage, COMSOL qualification, model identity, experimental records, or device-benchmark provenance should preserve or strengthen the existing evidence boundary.

Synthetic or surrogate results must never be relabeled as experimental or physically validated evidence as part of a security workaround.

## Secrets and sensitive data

Do not commit API keys, access tokens, private COMSOL models, proprietary datasets, participant information, device credentials, or other confidential material. Use local environment variables, private storage, or repository secrets appropriate to the workflow.

If a secret is committed, rotate or revoke it immediately; deleting it from the latest commit is not sufficient because Git history may retain the value.

## Coordinated disclosure

Please allow maintainers a reasonable remediation window before public disclosure. After a fix is available, the project may publish a GitHub Security Advisory and release notes describing affected versions, impact, remediation, and upgrade guidance without exposing unnecessary sensitive details.
