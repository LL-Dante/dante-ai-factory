# Security policy

## Private vulnerability reporting

GitHub Private Vulnerability Reporting will be enabled on the official DANTE GitHub
repository once that repository exists. No official repository has been created by
this preparation, so the mechanism is not currently active and no repository URL is
invented here. No personal email address is published.

After enablement, use the repository's Security / Advisories / Report a vulnerability
interface to submit privately. Before then, do not disclose vulnerabilities, exploit
details, credentials or private evidence in public issues or discussions. Keep the
report private until the official private reporting mechanism becomes available.

Before opening public intake, repository administrators must enable reporting and
verify notification handling. [GitHub configuration instructions](https://docs.github.com/en/code-security/how-tos/report-and-fix-vulnerabilities/configure-vulnerability-reporting/configure-for-a-repository).

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
