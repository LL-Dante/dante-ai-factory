from __future__ import annotations

from collections.abc import Mapping

from pydantic import Field

from dante.contracts import StrictModel


class AcceptanceContract(StrictModel):
    required_tools: frozenset[str] = frozenset()
    required_artifacts: frozenset[str] = frozenset()
    required_final_fields: frozenset[str] = frozenset()
    instructions: dict[str, str] = Field(default_factory=dict)


class ObservedAcceptance(StrictModel):
    tools: frozenset[str] = frozenset()
    artifacts: frozenset[str] = frozenset()
    final_fields: dict[str, str] = Field(default_factory=dict)

    def merged(self, other: "ObservedAcceptance") -> "ObservedAcceptance":
        return ObservedAcceptance(
            tools=self.tools | other.tools,
            artifacts=self.artifacts | other.artifacts,
            final_fields={**self.final_fields, **other.final_fields},
        )


class AcceptanceResult(StrictModel):
    acceptance_complete: bool
    missing_checks: tuple[str, ...]
    missing_artifacts: tuple[str, ...]
    missing_final_fields: tuple[str, ...]


class DeterministicAcceptanceVerifier:
    def verify(self, contract: AcceptanceContract, observed: ObservedAcceptance) -> AcceptanceResult:
        missing_tools = contract.required_tools - observed.tools
        missing_artifacts = contract.required_artifacts - observed.artifacts
        present_fields = frozenset(name for name, value in observed.final_fields.items() if value.strip())
        missing_fields = contract.required_final_fields - present_fields
        return AcceptanceResult(
            acceptance_complete=not (missing_tools or missing_artifacts or missing_fields),
            missing_checks=tuple(sorted(missing_tools)),
            missing_artifacts=tuple(sorted(missing_artifacts)),
            missing_final_fields=tuple(sorted(missing_fields)),
        )


def continuation_prompt(
    result: AcceptanceResult,
    known_final_fields: Mapping[str, str] | None = None,
    instructions: Mapping[str, str] | None = None,
) -> str:
    lines = ["The primary action succeeded.", "The task is NOT complete.", "", "Complete only these missing acceptance requirements:"]
    hints = instructions or {}
    lines.extend(f"- tool/action: {hints.get(name, name)}" for name in result.missing_checks)
    lines.extend(f"- artifact: {hints.get(name, name)}" for name in result.missing_artifacts)
    for name in result.missing_final_fields:
        known = (known_final_fields or {}).get(name)
        suffix = f" (use this verified value: {known})" if known else ""
        lines.append(f"- final field: {hints.get(name, name)}{suffix}")
    lines.extend(["", "Do not repeat completed work."])
    return "\n".join(lines)


__all__ = [
    "AcceptanceContract", "AcceptanceResult", "DeterministicAcceptanceVerifier",
    "ObservedAcceptance", "continuation_prompt",
]
