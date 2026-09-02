# Third-party dependency inventory and integration boundary

This inventory is not a replacement for third-party license texts. Dependencies
are obtained separately; no wheels, package sources, binaries or model assets are
redistributed in this export. Metadata is evidence, not a legal clearance.

## Direct public dependencies

| Package | Pin | Purpose | Upstream | Declared license | Distribution / concern |
|---|---|---|---|---|---|
| pydantic | 2.13.4 | Typed contracts/configuration | https://github.com/pydantic/pydantic | MIT | Dependency only; retain notices if later bundled |
| PyYAML | 6.0.3 | YAML configuration/registry | https://github.com/yaml/pyyaml | MIT | Dependency only; retain notices if later bundled |
| httpx | 0.28.1 | Local adapter/workspace HTTP transport | https://github.com/encode/httpx | BSD-3-Clause | Dependency only; preserve notices |
| ddgs | 9.14.4 | Search support in retained workspace wrapper | https://github.com/deedy5/ddgs | MIT | Dependency only; optional search can contact the network; tests do not |

The lock also records transitive packages. Installed metadata reviewed for these
versions reports MIT for annotated-types, anyio, brotli, h11, h2, hpack, hyperframe,
primp, pydantic_core and typing-inspection; BSD-3-Clause for click, httpcore, idna
and lxml; Apache-2.0 for fake-useragent; MPL-2.0 for certifi; PSF-2.0 for
typing_extensions. socksio has an MIT classifier but an UNKNOWN License field;
verify the distributed license before bundling. Native components and bundled data
may carry additional notices. Exact versions are in requirements-dante.lock.

## cptr / Open WebUI Computer: evidence and classification

Public source imports were inspected, including the transitive local entry points:
dante/cli.py imports tools.ai_cloud_workspace; that module imports httpx and ddgs,
not cptr. scripts/dante_e2e.py uses DANTE, the workspace wrapper and a local HTTP
fixture. Neither public dependency manifest lists cptr or Open WebUI. The public
171-test suite can run with only the public lock, with neither package installed.

Private AI-Cloud uses independently installed cptr with service/configuration
integration scripts (A: external integration). It also contains version-specific
patch scripts with copied/modified cptr source snippets (D in the private stack).
Those scripts are deliberately excluded. No cptr import/link dependency (B) is
present in the exported DANTE core. No cptr source, binary or asset redistribution
(C) and no copied cptr patch code (D) is included in this public export.
A historical Agent Host docstring mentioning cptr is descriptive, not an import.

The retained AICloudLiteLLMAdapter implements an HTTP boundary; it does not import
LiteLLM or cptr. The public package can be used without those integrations.
No architecture was rewritten to obtain this separation: existing independent
modules and deterministic fixtures were selected unchanged.

[Computer's upstream license](https://github.com/open-webui/computer/blob/main/LICENSE)
is Open Use License with additional conditions; independently installing it does
not license DANTE and does not waive its own obligations.
[Open WebUI](https://github.com/open-webui/open-webui/blob/main/LICENSE) has separate
branding/multi-license terms.
[LiteLLM](https://github.com/BerriAI/litellm/blob/main/LICENSE) distinguishes enterprise
content. These optional components and private patches are not bundled or required.
Review exact-version terms if adding any integration to a future distribution.

## Owner license decision

Following the owner declaration, original DANTE material is licensed Apache-2.0.
Third-party licenses remain unchanged. No package or cptr asset is bundled.
See LICENSE-DECISION.md for scope and the knowledge-qualified declaration.
