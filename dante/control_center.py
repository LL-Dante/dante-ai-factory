"""Read-only Control Center views of the model scout and the Stage B2 benchmark.

Turns a persisted `ScoutSnapshot` or `BenchmarkResult` into a flat, display-ready
structure: no Fact internals, no mutable state, no control surface. Nothing here
pulls, loads, benchmarks, ranks or selects a model, so neither view can be used to
change the node.
"""
from __future__ import annotations

from typing import Any

from dante.contracts.benchmarks import BenchmarkResult
from dante.contracts.models import ScoutSnapshot

# A view row is intentionally tiny so a panel can render it without knowing Fact.
Row = dict[str, Any]


def _cell(fact, *, unit: str | None = None) -> Row:
    """One display cell: the value when measured, else the reason it is unknown.

    Accepts a `Fact` or the plain dict a persisted result carries, because a
    panel is fed stored JSON rather than live objects.
    """
    if isinstance(fact, dict):
        kind = fact.get('kind')
        value = fact.get('value')
        cell_unit = fact.get('unit')
        reason = fact.get('reason')
    else:
        kind, value, cell_unit, reason = fact.kind, fact.value, fact.unit, fact.reason
    cell: Row = {'kind': kind}
    if kind == 'unknown':
        cell['value'] = None
        cell['reason'] = reason
        return cell
    cell['value'] = value
    if unit is not None:
        cell['unit'] = cell_unit or unit
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


def benchmark_model_card(run: dict) -> Row:
    """One benchmarked model as the Control Center shows it.

    Speed, memory and reliability stay in separate fields. A withheld dimension
    carries its blocking reason, so an absent number is never read as a zero.
    """
    profile = run.get('profile') or {}
    budget = profile.get('vram_budget') or {}
    samples = run.get('samples') or []
    return {
        'model_id': profile.get('model_id'),
        'digest': profile.get('digest_sha256'),
        'quantization': profile.get('quantization'),
        'parameter_size_label': profile.get('parameter_size_label'),
        'artifact_bytes': profile.get('artifact_bytes'),
        'tested': profile.get('tested'),
        'skip_reason': profile.get('skip_reason'),
        'load_succeeded': profile.get('load_succeeded'),
        'load_failure': profile.get('load_failure'),
        'gpu_execution': profile.get('gpu_execution'),
        'sample_count': profile.get('sample_count'),
        'completed_samples': profile.get('completed_samples'),
        'failed_samples': profile.get('failed_samples'),
        'timed_out_samples': profile.get('timed_out_samples'),
        'malformed_outputs': profile.get('malformed_outputs'),
        'median_generation_tokens_per_second': profile.get('median_generation_tokens_per_second'),
        'median_prompt_tokens_per_second': profile.get('median_prompt_tokens_per_second'),
        'median_cold_load_s': profile.get('median_cold_load_s'),
        'median_time_to_first_token_s': profile.get('median_time_to_first_token_s'),
        'observed_vram_resident_bytes': profile.get('observed_vram_resident_bytes'),
        'observed_ram_working_set_bytes': profile.get('observed_ram_working_set_bytes'),
        'vram_budget': {
            'num_gpu_layers': budget.get('num_gpu_layers'),
            'block_count': budget.get('block_count'),
            'full_gpu_possible': budget.get('full_gpu_possible'),
            'budget_bytes': budget.get('budget_bytes'),
            'reserve_bytes': budget.get('reserve_bytes'),
            'free_vram_bytes': budget.get('free_vram_bytes'),
            'basis': budget.get('basis'),
        },
        'failures': [sample.get('failure') for sample in samples if sample.get('failure')],
        'samples': [{
            'prompt_id': sample.get('prompt_id'),
            'prompt_kind': sample.get('prompt_kind'),
            'repetition': sample.get('repetition'),
            'phase': sample.get('phase'),
            'disposition': sample.get('disposition'),
            'done_reason': sample.get('done_reason'),
            'output_parsed': sample.get('output_parsed'),
            'measurements': {name: _cell(fact) for name, fact in (sample.get('measurements') or {}).items()},
        } for sample in samples],
    }


def benchmark_overview(structured: dict) -> Row:
    """Control Center view of a persisted Stage B2 benchmark result.

    `structured` is the benchmark agent's `structured_result`. The panel shows
    isolation evidence, per-model measurements and per-dimension comparisons. It
    exposes no overall winner and no role assignment, because neither exists.
    """
    payload = (structured or {}).get('benchmark')
    if not isinstance(payload, dict):
        raise ValueError('Benchmark result is unavailable')
    result = BenchmarkResult.model_validate(payload)
    isolation = result.isolation
    excluded = [dict(item) for item in (structured or {}).get('excluded_models') or []]
    return {
        'observed_at': result.observed_at.isoformat(),
        'probe_version': result.probe_version,
        'smoke_only': result.smoke_only,
        'inference_performed': bool((structured or {}).get('inference_performed')),
        'isolation': {
            'production_endpoint': isolation.production_endpoint,
            'benchmark_endpoint': isolation.benchmark_endpoint,
            'production_unchanged': isolation.production_unchanged,
            'production_checked_batches': isolation.production_checked_batches,
            'production_disturbances': list(isolation.production_disturbances),
            'benchmark_restored': isolation.benchmark_restored,
            'models_loaded': list(isolation.models_loaded),
            'leaked_residency': list(isolation.leaked_residency),
            'violations': list(isolation.violations),
            'verified': isolation.verified(),
            'policy': isolation.policy,
        },
        'models_tested': list(result.tested_models()),
        'models': [benchmark_model_card(run.model_dump(mode='json')) for run in result.runs],
        'excluded_models': excluded,
        'comparison': {
            'models': list(result.comparison.models),
            'shared_parameters_identical': result.comparison.shared_parameters_identical,
            'recorded_differences': list(result.comparison.recorded_differences),
            'universal_score': result.comparison.universal_score,
            'overall_winner': result.comparison.overall_winner,
            'verdict': result.comparison.verdict,
            'dimensions': [{
                'dimension': item.dimension,
                'unit': item.unit,
                'comparable': item.comparable,
                'best_model_id': item.best_model_id,
                'basis': item.basis,
                'blocking_differences': list(item.blocking_differences),
                'values': {model: _cell(fact) for model, fact in item.values.items()},
            } for item in result.comparison.dimensions],
        },
        'models_downloaded': result.models_downloaded,
        'models_deleted': result.models_deleted,
        'qualification_changed': result.qualification_changed,
        'role_assignments_made': result.role_assignments_made,
    }


def render_benchmark_text(overview: Row) -> str:
    """One bounded, dependency-free benchmark panel for the Control Center."""
    isolation = overview['isolation']
    lines = [
        f"BENCHMARK {overview['observed_at']} probe={overview['probe_version']} "
        f"smoke_only={str(overview['smoke_only']).lower()}",
        f"production={isolation['production_endpoint']} benchmark={isolation['benchmark_endpoint']} "
        f"production_unchanged={str(isolation['production_unchanged']).lower()} "
        f"checked_batches={isolation['production_checked_batches']} "
        f"benchmark_restored={str(isolation['benchmark_restored']).lower()}",
    ]
    for card in overview['models']:
        budget = card['vram_budget']
        speed = card['median_generation_tokens_per_second']
        lines.append(
            f"- {card['model_id']} {card['parameter_size_label'] or '?'} {card['quantization'] or '?'} "
            f"tested={str(card['tested']).lower()} gpu={card['gpu_execution']} "
            f"gen_tok_s={speed if speed is not None else 'unknown'} "
            f"vram={card['observed_vram_resident_bytes'] or 'unknown'} "
            f"ram={card['observed_ram_working_set_bytes'] or 'unknown'} "
            f"num_gpu={budget['num_gpu_layers']}/{budget['block_count']}")
        for failure in card['failures']:
            lines.append(f"    failure: {failure}")
    for item in overview['comparison']['dimensions']:
        values = ' '.join(f"{model}={cell['value'] if cell['value'] is not None else 'unknown'}"
                          for model, cell in item['values'].items())
        lines.append(f"  {item['dimension']} comparable={str(item['comparable']).lower()} {values}")
    for difference in overview['comparison']['recorded_differences']:
        lines.append(f"  difference: {difference}")
    lines.append(f"universal_score={overview['comparison']['universal_score']} "
                 f"overall_winner={overview['comparison']['overall_winner']} "
                 f"role_assignments={overview['role_assignments_made']}")
    for item in overview['excluded_models']:
        lines.append(f"  excluded: {item['model_id']} ({item['reason']})")
    return '\n'.join(lines)


__all__ = ['benchmark_model_card', 'benchmark_overview', 'model_card', 'model_overview',
           'render_benchmark_text', 'render_text']
