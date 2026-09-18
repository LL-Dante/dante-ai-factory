# Security policy

## Private vulnerability reporting

GitHub Private Vulnerability Reporting is enabled on the official
[LL-Dante/dante-ai-factory repository](https://github.com/LL-Dante/dante-ai-factory).
Use [Report a vulnerability](https://github.com/LL-Dante/dante-ai-factory/security/advisories/new)
to submit privately. Do not disclose vulnerabilities, exploit details, credentials
or private evidence in public issues or discussions. No personal email is published.

## Scope and limitations

This is a pre-release foundation with no hardened OS sandbox, multi-tenant guarantee
or security SLA. Trusted Python tool handlers inherit user permissions. Thread timeout
cannot forcibly stop arbitrary code; uncertain effects require reconciliation.
Use restricted workspaces, explicit approvals, loopback services and a trusted account.
The baseline qualification in README.md does not guarantee future revisions or hardware.

Never distribute runtime data, populated environment files, cookies, access keys,
models or unreviewed third-party assets. Scan proposed release contents and history.
If real credentials are discovered, revoke them and separately authorize remediation.
Regex scans are heuristic; they do not prove complete absence of sensitive data.
