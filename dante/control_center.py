"""Read-only Control Center view of the model scout.

Turns a persisted `ScoutSnapshot` into a flat, display-ready structure: no Fact
internals, no mutable state, no control surface. Nothing here pulls, loads,
benchmarks, ranks or selects a model, so the view cannot be used to change the node.
"""
from __future__ import annotations

from typing import Any

from dante.contracts.models import ScoutSnapshot

# A view row is intentionally tiny so a panel can render it without knowing Fact.
Row = dict[str, Any]


def _cell(fact, *, unit: str | None = None) -> Row:
    """One display cell: the value when measured, else the reason it is unknown."""
    cell: Row = {'kind': fact.kind}
    if fact.kind == 'unknown':
        cell['value'] = None
        cell['reason'] = fact.reason
        return cell
    cell['value'] = fact.value
    if unit is not None:
        cell['unit'] = fact.unit or unit
    return cell


def _text(fact) -> str | None:
    return str(fact.value) if fact.kind == 'measured' else None


def model_card(record, fit=None) -> Row:
    """One model as the Control Center shows it."""
    if fit is None:
        fit_facts: Row = {}
    else:
        fit_facts = {
            'artifact_fits_total_vram': _cell(fit.artifact_fits_total_vram),
            'artifact_fits_free_vram': _cell(fit.artifact_fits_free_vram),
            'artifact_fits_free_storage': _cell(fit.artifact_fits_free_storage),
            'gpu_residency_plausible': _cell(fit.gpu_residency_plausible),
            'cpu_ram_offload_may_be_required': _cell(fit.cpu_ram_offload_may_be_required),
            'storage_headroom_bytes': _cell(fit.storage_headroom_bytes),
        }
    return {
        'model_id': record.model_id,
        'runtime': record.runtime,
        'endpoints': list(record.endpoints),
        'loaded_on': list(record.loaded_endpoints),
        'presence': list(record.presence),
        'installed': _cell(record.installed),
        'loaded': _cell(record.loaded),
        'node0_qualified': _cell(record.node0_qualified),
        'qualification_id': _cell(record.qualification_id),
        'architecture': _text(record.architecture),
        'parameter_count': _cell(record.parameter_count),
        'parameter_size_label': _text(record.parameter_size_label),
        'quantization': _text(record.quantization),
        'file_format': _text(record.file_format),
        'file_size_bytes': _cell(record.file_size_bytes, unit='B'),
        'native_context_tokens': _cell(record.native_context_tokens, unit='tokens'),
        'configured_context_tokens': _cell(record.configured_context_tokens, unit='tokens'),
        'loaded_vram_bytes': _cell(record.loaded_vram_bytes, unit='B'),
        'runtime_capabilities': list(record.declared_capabilities()),
        'parent_model': _text(record.parent_model),
        'digest': _text(record.digest),
        'unknown_fields': list(record.unknown_fields()),
        'fit': fit_facts,
    }


def model_overview(structured: dict) -> Row:
    """Control Center view of a persisted scout result.

    `structured` is the agent's `structured_result`. Only observed data is shown;
    absent data is reported as unknown rather than filled in.
    """
    payload = (structured or {}).get('scout')
    if not isinstance(payload, dict):
        raise ValueError('Scout result is unavailable')
    snapshot = ScoutSnapshot.model_validate(payload)
    registry = snapshot.registry
    fit = {entry.model_id: entry for entry in registry.fit}
    models = [model_card(record, fit.get(record.model_id)) for record in registry.records]
    return {
        'observed_at': snapshot.observed_at.isoformat(),
        'read_only': True,
        'runtimes': list(registry.runtimes_observed),
        'runtimes_unreachable': list(registry.runtimes_unreachable),
        'qualified_endpoint': registry.qualified_endpoint,
        'qualified_model': registry.qualified_model,
        'model_count': snapshot.model_count,
        'loaded_count': snapshot.loaded_count,
        'qualified_count': snapshot.qualified_count,
        'models': models,
        'benchmark_plan': {
            'plan_version': snapshot.benchmark_plan.plan_version,
            'execution_state': snapshot.benchmark_plan.execution_state,
            'metric_count': len(snapshot.benchmark_plan.metrics),
            'classes': [{'class_id': item.class_id, 'name': item.name,
                         'repetitions': item.repetitions,
                         'metrics': list(item.metrics)} for item in snapshot.benchmark_plan.classes],
            'roles': [{'role_id': item.role_id, 'name': item.name,
                       'assigned_models': list(item.assigned_models)} for item in snapshot.benchmark_plan.roles],
        },
        'rankings_withheld': registry.rankings_withheld,
        'winners_declared': snapshot.winners_declared,
        'inference_performed': snapshot.inference_performed,
        'models_downloaded': snapshot.models_downloaded,
        'models_deleted': snapshot.models_deleted,
    }


def render_text(overview: Row) -> str:
    """One bounded, dependency-free text panel for the Control Center."""
    lines = [
        f"MODEL SCOUT {overview['observed_at']} read_only={str(overview['read_only']).lower()}",
        f"runtimes={len(overview['runtimes'])} models={overview['model_count']} "
        f"loaded={overview['loaded_count']} qualified={overview['qualified_count']}",
        f"rankings_withheld={str(overview['rankings_withheld']).lower()} "
        f"winners_declared={str(bool(overview['winners_declared'])).lower()}",
    ]
    for card in overview['models']:
        size = card['file_size_bytes']['value']
        size_text = f'{size}B' if isinstance(size, int) else 'unknown'
        loaded = 'resident' if card['loaded_on'] else 'not-resident'
        lines.append(
            f"- {card['model_id']} [{','.join(card['presence'])}] {loaded} "
            f"{card['parameter_size_label'] or '?'} {card['quantization'] or '?'} {size_text} "
            f"resident_on={','.join(card['loaded_on']) or 'none'}")
    plan = overview['benchmark_plan']
    lines.append(f"benchmark {plan['plan_version']} {plan['execution_state']} "
                 f"classes={len(plan['classes'])} roles={len(plan['roles'])} "
                 f"metrics={plan['metric_count']} (no winners declared)")
    return '\n'.join(lines)


__all__ = ['model_card', 'model_overview', 'render_text']
