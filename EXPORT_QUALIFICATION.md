# Public export qualification

The private baseline 8ce432ba57b55ffb3bb41fe3afb8b4ad125986e4 passed 171/171 tests
and gates A-K. The independent public export does not have that commit hash or
inherit its history. It independently passed 171/171 in a fresh Python 3.12.13
22-package environment synced from public manifests (41.523 seconds), with cptr
and Open WebUI absent. Later replay: 171/171 in 42.361 seconds.

SOURCE_MANIFEST.json records 46 selected baseline files, with two EOF-only source
normalizations and license metadata added to pyproject.toml. Runtime architecture,
dependencies and tests are unchanged. Public configuration is neutral. Private
history, conversations, operational reports, credentials, models, deployment scripts
and copied upstream patches are excluded.

The owner confirmation resolves provenance for release preparation; original DANTE
material is Apache-2.0. Third-party components retain their licenses. No cptr assets
are redistributed. See LICENSE-DECISION.md and THIRD_PARTY_NOTICES.md.

The applicable test suite is foundation 16, P1 17, P2 22, P3 24, P4 21, P5 33,
P6 38. No real model, workstation hardware, llama.cpp runtime or vLLM is qualified.
Trusted Python tools are not an OS sandbox; timeout cannot forcibly stop arbitrary
threads. Gemini cost remains COST_UNVERIFIED. No P7, external AI calls or model
downloads are part of preparation.

Once final gates pass, the authorized local repository has one independent root
commit on main. No private history, refs, stash, reflog, alternates or object storage
is copied. Identical sanitized source naturally retains identical Git blob hashes;
this is not inherited private history. No remote or hosted repository is created.
GitHub Private Vulnerability Reporting will be enabled on the official repository
when it exists; it is not active yet.

Owner-confirmation safety replay: 171/171 PASS, zero failures/errors/skips, 42.096 seconds. Subsequent changes affect license metadata and documentation only; no executable source or dependency changes. Final complete-export secret, personal-data, development-path, size and whitespace checks pass. The remaining action is the authorized independent local initial commit.
