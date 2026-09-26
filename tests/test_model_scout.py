"""Stage B1 model discovery: read-only enumeration, registry, fit facts, control view."""
from __future__ import annotations

import json
import threading
import unittest
from datetime import datetime, timedelta, timezone

from pydantic import ValidationError

from dante.agent_runner import (AgentRegistry, AgentRunner, AgentUnavailable, DanteModelScout,
                                build_agent_registry, definition_digest)
from dante.contracts.agents import AgentModelTarget, AgentTaskPayload
from dante.contracts.hardware import HardwareSnapshot
from dante.contracts.models import ModelRecord, ModelRegistry, ScoutSnapshot
from dante.control_center import model_card, model_overview, render_text
from dante.hardware_inventory import collect_snapshot
from dante.model_scout import discover_models, hardware_snapshot_digest

NODE0 = 'http://127.0.0.1:11435'
SECOND = 'http://127.0.0.1:11434'
DEAD = 'http://127.0.0.1:11439'
# Endpoints are reported in deterministic (sorted) order, not discovery order.
BOTH = tuple(sorted([NODE0, SECOND]))
QUALIFIED = 'qualification-fixture'
SMALL = 'qwen3:4b'
LARGE = 'hf.co/KikoCis/Qwen3.8-27B-GGUF:Q3_K_M'
DERIVED = 'dante-qwen-agent:latest'
SMALL_DIGEST = '359d7dd4bcdab3d86b87d73ac27966f4dbb9f5efdfcc75d34a8764a09474fae7'
LARGE_DIGEST = '4ae69a30b61946aa270ab86e' + '0' * 40
DERIVED_DIGEST = '256d7bff33f4b4e9aab49add' + '0' * 40

WMI = {
    'os_name': 'Microsoft Windows 11 Pro', 'os_version': '10.0.26200', 'os_build': '26200',
    'architecture': '64 bit', 'last_boot': '2026-09-25T15:03:35+00:00',
    'cpu_name': 'AMD Ryzen 9 9900X 12-Core Processor', 'cpu_cores': 12, 'cpu_threads': 24,
    'cpu_max_mhz': 4400, 'cpu_current_mhz': 4400,
    'mem_modules': [{'locator': 'DIMMA2', 'bytes': 17179869184, 'speed': 6000,
                     'configured': 6000, 'smbios': 34}],
    'disks': [{'index': '0', 'model': 'Lexar SSD NM790 2TB', 'size': 2048029360128}],
    'volumes': [{'mount': 'C:', 'fs': 'NTFS', 'size': 2047231389696, 'free': 1792787107840}],
    'power_plan_guid': '381b4222-f694-41f0-9685-ff5bb260df2e', 'power_plan_name': 'Balanced',
}


def monitoring(age_s: float = 0.0) -> dict:
    return {'timestamp': (datetime.now(timezone.utc) - timedelta(seconds=age_s)).isoformat(),
            'cpu': {'utilization_percent': 63.0, 'temperature_c': None, 'package_power_w': None},
            'memory': {'total_mb': 31800, 'available_mb': 12964, 'used_mb': 18836},
            'gpu': {'temperature_c': 39.0, 'power_w': 107.09, 'power_limit_w': 360.0,
                    'utilization_percent': 43.0, 'vram_used_mb': 4581.0, 'vram_total_mb': 16303.0},
            'storage': {}}


def nvidia(gpus=True):
    from dante.contracts.qualification import GPUProfile
    profiles = (GPUProfile(slot='0', vendor='NVIDIA', name='NVIDIA GeForce RTX 5080',
                           uuid='GPU-fixture', vram_bytes=17094934528,
                           compute_capability='12.0', driver_version='591.86'),) if gpus else ()
    return type('Obs', (), {'tool_available': True, 'gpus': profiles,
                            'cuda_driver_api': 'driver-supported-13.1'})()


class FakeNvidiaRun:
    returncode, stderr = 0, ''

    def __init__(self, text):
        self.stdout = text


def hardware(*, gpus=True, age_s: float = 0.0) -> HardwareSnapshot:
    return collect_snapshot(
        runtime_endpoints=(NODE0,), qualified_endpoint=NODE0,
        wmi_runner=lambda script: dict(WMI),
        monitoring_fetch=lambda: (monitoring(age_s), 'monitoring-http'),
        nvidia_runner=lambda args: FakeNvidiaRun('0, 11397, 2797, 14801, 3090, 5, 16\n'),
        nvidia_probe=nvidia if gpus else (lambda: type(
            'Obs', (), {'tool_available': False, 'gpus': (), 'cuda_driver_api': 'absent'})()),
        uuid_probe=lambda: {'0': 'GPU-fixture'} if gpus else {},
        adapter_factory=lambda profile: FakeRuntime())


class FakeRuntime:
    """Only the read-only metadata surface the scout is allowed to use."""

    installed: list = []
    resident: list = []
    shown: dict = {}
    fail_version = False
    fail_resident = False
    fail_show = False

    def __init__(self, profile=None, *, installed=None, resident=None, shown=None,
                 fail_version=False, fail_resident=False, fail_show=False):
        self.profile = profile
        self.installed = [] if installed is None else installed
        self.resident = [] if resident is None else resident
        self.shown = {} if shown is None else shown
        self.fail_version, self.fail_resident, self.fail_show = fail_version, fail_resident, fail_show
        self.calls: list[str] = []

    def version(self):
        self.calls.append('version')
        if self.fail_version:
            raise ValueError('unreachable')
        return '0.34.4'

    def inventory(self):
        self.calls.append('inventory')
        if self.fail_version:
            raise ValueError('unreachable')
        return list(self.installed)

    def loaded(self):
        self.calls.append('loaded')
        if self.fail_resident:
            raise ValueError('no resident view')
        return list(self.resident)

    def show(self, reference):
        self.calls.append(f'show:{reference}')
        if self.fail_show:
            raise ValueError('no metadata')
        return dict(self.shown.get(reference, {}))


def tag(name, digest, size, *, family='qwen3', quant='Q4_K_M', params='4.0B',
        context=262144, parent='', modified='2026-09-01T00:00:00Z'):
    return {'name': name, 'digest': digest, 'size': size, 'modified_at': modified,
            'details': {'family': family, 'parameter_size': params, 'quantization_level': quant,
                        'format': 'gguf', 'context_length': context, 'parent_model': parent}}


def show(*, family='qwen3', params=4022468096, embedding=2560, blocks=36):
    return {'details': {'family': family, 'format': 'gguf', 'quantization_level': 'Q4_K_M'},
            'model_info': {f'{family}.context_length': 262144,
                           f'{family}.embedding_length': embedding,
                           f'{family}.block_count': blocks,
                           f'{family}.general.architecture': family,
                           f'{family}.general.parameter_count': params},
            'capabilities': ['completion', 'thinking', 'tools']}


SMALL_TAG = tag(SMALL, SMALL_DIGEST, 2497293931)
LARGE_TAG = tag(LARGE, LARGE_DIGEST, 13500737704, family='qwen35', quant='Q3_K_M',
                params='26.9B', parent='')
DERIVED_TAG = tag(DERIVED, DERIVED_DIGEST, 13500737325, family='qwen35', quant='Q3_K_M',
                  params='26.9B', parent=LARGE)
SMALL_SHOW = show()
LARGE_SHOW = show(family='qwen35', params=27320697856, embedding=5120, blocks=65)
RESIDENT_SMALL = {'name': SMALL, 'model': SMALL, 'size_vram': 3178149969,
                  'context_length': 4096, 'expires_at': '2026-09-26T06:00:00Z'}


def factory(table: dict):
    def build(profile):
        return FakeRuntime(profile, **table.get(profile.base_url, {}))
    return build


def scout(hardware_snapshot, *, endpoints=(NODE0,), qualified_model=SMALL, qualified_id=QUALIFIED,
          qualified_endpoint=NODE0, table=None):
    return discover_models(
        hardware=hardware_snapshot, runtime_endpoints=endpoints,
        qualified_endpoint=qualified_endpoint, qualified_model=qualified_model,
        qualified_id=qualified_id, adapter_factory=factory(table or {}))


TWO_RUNTIMES = {
    NODE0: {'installed': [SMALL_TAG, LARGE_TAG, DERIVED_TAG], 'resident': [RESIDENT_SMALL],
            'shown': {SMALL: SMALL_SHOW, LARGE: LARGE_SHOW, DERIVED: LARGE_SHOW}},
    SECOND: {'installed': [SMALL_TAG, LARGE_TAG, DERIVED_TAG], 'resident': [],
             'shown': {SMALL: SMALL_SHOW, LARGE: LARGE_SHOW, DERIVED: LARGE_SHOW}},
}


class DiscoveryShapeTests(unittest.TestCase):
    def test_one_record_per_model_across_runtimes_sharing_a_store(self):
        result = scout(hardware(), endpoints=(NODE0, SECOND), table=TWO_RUNTIMES)
        self.assertEqual(result.model_count, 3)
        self.assertEqual(result.registry.identities(), tuple(sorted([SMALL, LARGE, DERIVED])))
        for record in result.registry.records:
            self.assertEqual(record.endpoints, BOTH)
            self.assertEqual(len(record.endpoints), 2, 'shared store must not duplicate records')

    def test_every_runtime_is_reported_and_none_is_assumed_reachable(self):
        result = scout(hardware(), endpoints=(NODE0, SECOND), table=TWO_RUNTIMES)
        self.assertEqual(result.registry.runtimes_observed, BOTH)
        self.assertEqual(result.registry.runtimes_unreachable, ())

    def test_unreachable_runtime_is_recorded_and_loses_its_models(self):
        result = scout(hardware(), endpoints=(NODE0, DEAD),
                       table={NODE0: TWO_RUNTIMES[NODE0], DEAD: {'fail_version': True}})
        self.assertEqual(result.registry.runtimes_observed, (NODE0,))
        self.assertEqual(result.registry.runtimes_unreachable, (DEAD,))
        self.assertEqual(result.model_count, 3)

    def test_scout_uses_metadata_endpoints_only(self):
        runtimes: list[FakeRuntime] = []

        def build(profile):
            runtime = FakeRuntime(profile, **TWO_RUNTIMES[profile.base_url])
            runtimes.append(runtime)
            return runtime
        discover_models(hardware=hardware(), runtime_endpoints=(NODE0, SECOND),
                        qualified_endpoint=NODE0, qualified_model=SMALL,
                        qualified_id=QUALIFIED, adapter_factory=build)
        self.assertTrue(runtimes)
        for runtime in runtimes:
            self.assertTrue(set(runtime.calls) <= {'version', 'inventory', 'loaded'}
                            or set(runtime.calls) <= {'version', 'inventory', 'loaded',
                                                      'show:qwen3:4b',
                                                      'show:hf.co/KikoCis/Qwen3.8-27B-GGUF:Q3_K_M',
                                                      'show:dante-qwen-agent:latest'})
            for forbidden in ('generate', 'chat', 'load', 'pull', 'delete', 'copy', 'embed'):
                self.assertNotIn(forbidden, runtime.calls)

    def test_registry_order_is_deterministic_regardless_of_endpoint_order(self):
        first = scout(hardware(), endpoints=(NODE0, SECOND), table=TWO_RUNTIMES)
        second = scout(hardware(), endpoints=(SECOND, NODE0), table=TWO_RUNTIMES)
        self.assertEqual([r.model_id for r in first.registry.records],
                         [r.model_id for r in second.registry.records])
        self.assertEqual(first.registry.runtimes_observed, second.registry.runtimes_observed)


class ModelMetadataTests(unittest.TestCase):
    def setUp(self):
        self.snapshot = scout(hardware(), endpoints=(NODE0,), table={NODE0: TWO_RUNTIMES[NODE0]})

    def test_exact_identity_size_quantization_and_parameter_count(self):
        record = self.snapshot.registry.require(SMALL)
        self.assertEqual(record.digest.value, SMALL_DIGEST)
        self.assertEqual(record.file_size_bytes.value, 2497293931)
        self.assertEqual(record.file_size_bytes.unit, 'B')
        self.assertEqual(record.quantization.value, 'Q4_K_M')
        self.assertEqual(record.file_format.value, 'gguf')
        self.assertEqual(record.architecture.value, 'qwen3')
        self.assertEqual(record.parameter_count.value, 4022468096)
        self.assertEqual(record.parameter_size_label.value, '4.0B')
        self.assertEqual(record.native_context_tokens.value, 262144)
        self.assertEqual(record.embedding_length.value, 2560)
        self.assertEqual(record.block_count.value, 36)
        self.assertEqual(record.declared_capabilities(), ('completion', 'thinking', 'tools'))

    def test_derivative_records_its_parent_and_keeps_its_own_identity(self):
        record = self.snapshot.registry.require(DERIVED)
        self.assertEqual(record.parent_model.value, LARGE)
        self.assertEqual(record.digest.value, DERIVED_DIGEST)
        self.assertNotEqual(record.digest.value, self.snapshot.registry.require(LARGE).digest.value)

    def test_installed_loaded_and_qualified_states_stay_distinct(self):
        record = self.snapshot.registry.require(SMALL)
        self.assertEqual(record.presence, ('installed', 'loaded', 'node0_qualified'))
        self.assertTrue(record.installed.value)
        self.assertTrue(record.loaded.value)
        self.assertTrue(record.node0_qualified.value)
        self.assertEqual(record.qualification_id.value, QUALIFIED)
        self.assertEqual(record.loaded_endpoints, (NODE0,))
        self.assertEqual(record.configured_context_tokens.value, 4096)
        self.assertEqual(record.loaded_vram_bytes.value, 3178149969)

    def test_installed_but_not_resident_stays_unqualified_and_unknown_where_unobservable(self):
        record = self.snapshot.registry.require(LARGE)
        self.assertEqual(record.presence, ('installed', 'unqualified'))
        self.assertTrue(record.installed.value)
        self.assertFalse(record.loaded.value)
        self.assertEqual(record.loaded_endpoints, ())
        self.assertEqual(record.loaded_vram_bytes.kind, 'unknown')
        self.assertIsNone(record.loaded_vram_bytes.value)
        self.assertIn('not resident', record.loaded_vram_bytes.reason)
        self.assertEqual(record.configured_context_tokens.kind, 'unknown')
        self.assertEqual(record.qualification_id.kind, 'unknown')

    def test_qualification_is_scoped_to_the_qualified_runtime(self):
        # The same model on a second, non-qualified runtime must not be qualified.
        result = scout(hardware(), endpoints=(NODE0, SECOND), table=TWO_RUNTIMES)
        record = result.registry.require(SMALL)
        self.assertTrue(record.node0_qualified.value)
        self.assertIn(NODE0, record.endpoints)
        self.assertIn(SECOND, record.endpoints)
        self.assertNotIn('unqualified', record.presence)

    def test_unknown_qualification_is_not_reported_as_qualified(self):
        result = scout(hardware(), qualified_model=None, qualified_id=None,
                       table={NODE0: TWO_RUNTIMES[NODE0]})
        record = result.registry.require(SMALL)
        self.assertFalse(record.node0_qualified.value)
        self.assertEqual(record.qualification_id.kind, 'unknown')
        self.assertEqual(result.qualified_count, 0)

    def test_missing_metadata_becomes_unknown_with_a_reason_not_an_estimate(self):
        # A runtime that reports a name and a size and nothing else.
        bare = {'name': 'bare:latest', 'digest': 'f' * 64, 'size': 1024}
        table = {NODE0: {'installed': [bare], 'resident': [], 'shown': {}, 'fail_show': True}}
        result = scout(hardware(), endpoints=(NODE0,), table=table)
        record = result.registry.require('bare:latest')
        self.assertEqual(record.file_size_bytes.value, 1024)
        for field in ('architecture', 'parameter_count', 'parameter_size_label',
                      'quantization', 'file_format', 'embedding_length', 'block_count',
                      'native_context_tokens', 'runtime_capabilities', 'modified_at'):
            fact = getattr(record, field)
            self.assertEqual(fact.kind, 'unknown', field)
            self.assertIsNone(fact.value, field)
            self.assertTrue(fact.reason, field)
        self.assertTrue(record.unknown_fields())

    def test_unavailable_resident_view_never_claims_a_model_is_not_loaded(self):
        table = {NODE0: {'installed': [SMALL_TAG], 'resident': [], 'shown': {SMALL: SMALL_SHOW},
                         'fail_resident': True}}
        result = scout(hardware(), endpoints=(NODE0,), table=table)
        record = result.registry.require(SMALL)
        self.assertNotIn('loaded', record.presence)
        self.assertEqual(record.loaded_vram_bytes.kind, 'unknown')

    def test_registry_reports_no_shared_manifest_digest_when_nothing_is_shared(self):
        result = scout(hardware(), endpoints=(NODE0, SECOND), table=TWO_RUNTIMES)
        self.assertEqual(result.registry.shared_digests, ())

    def test_shared_digest_between_two_model_ids_is_surfaced(self):
        twin = tag('twin:latest', SMALL_DIGEST, 2497293931)
        table = {NODE0: {'installed': [SMALL_TAG, twin], 'resident': [RESIDENT_SMALL],
                         'shown': {SMALL: SMALL_SHOW, 'twin:latest': SMALL_SHOW}}}
        result = scout(hardware(), endpoints=(NODE0,), table=table)
        self.assertEqual(result.registry.shared_digests, (f'{SMALL} + twin:latest',))


class FitFactsTests(unittest.TestCase):
    def setUp(self):
        self.snapshot = scout(hardware(), endpoints=(NODE0,), table={NODE0: TWO_RUNTIMES[NODE0]})

    def _fit(self, model_id):
        return next(item for item in self.snapshot.registry.fit if item.model_id == model_id)

    def test_artifact_that_fits_observed_free_vram_is_plausibly_resident(self):
        fit = self._fit(SMALL)
        self.assertTrue(fit.artifact_fits_total_vram.value)
        self.assertTrue(fit.artifact_fits_free_vram.value)
        self.assertTrue(fit.artifact_fits_free_storage.value)
        self.assertTrue(fit.gpu_residency_plausible.value)
        self.assertFalse(fit.cpu_ram_offload_may_be_required.value)

    def test_artifact_larger_than_free_vram_is_reported_as_offload_not_unusable(self):
        fit = self._fit(LARGE)
        self.assertTrue(fit.artifact_fits_total_vram.value)
        self.assertFalse(fit.artifact_fits_free_vram.value)
        self.assertFalse(fit.gpu_residency_plausible.value)
        self.assertTrue(fit.cpu_ram_offload_may_be_required.value)
        self.assertTrue(fit.verdicts_withheld)

    def test_storage_headroom_is_arithmetic_only(self):
        fit = self._fit(SMALL)
        self.assertEqual(fit.storage_headroom_bytes.kind, 'derived')
        self.assertEqual(fit.storage_headroom_bytes.value, 1792787107840 - 2497293931)

    def test_no_gpu_makes_every_fit_comparison_unknown_rather_than_false(self):
        result = scout(hardware(gpus=False), endpoints=(NODE0,),
                       table={NODE0: TWO_RUNTIMES[NODE0]})
        fit = next(item for item in result.registry.fit if item.model_id == LARGE)
        self.assertEqual(fit.vram_total_bytes.kind, 'unknown')
        self.assertEqual(fit.artifact_fits_total_vram.kind, 'unknown')
        self.assertEqual(fit.artifact_fits_free_vram.kind, 'unknown')
        self.assertEqual(fit.gpu_residency_plausible.kind, 'unknown')
        self.assertEqual(fit.cpu_ram_offload_may_be_required.kind, 'unknown')
        self.assertIsNone(fit.gpu_residency_plausible.value)

    def test_fit_is_bound_to_the_hardware_snapshot_digest(self):
        result = scout(hardware(), endpoints=(NODE0,), table={NODE0: TWO_RUNTIMES[NODE0]})
        self.assertEqual(result.registry.hardware_snapshot_digest, result.hardware_digest)
        self.assertEqual(result.hardware_digest, hardware_snapshot_digest(hardware()))
        self.assertEqual(len(result.hardware_digest), 64)

    def test_fit_records_its_basis_and_never_a_verdict(self):
        fit = self._fit(LARGE)
        self.assertTrue(fit.basis)
        self.assertTrue(all('digest=' in item or '=' in item for item in fit.basis))


class RegistryContractTests(unittest.TestCase):
    def setUp(self):
        self.snapshot = scout(hardware(), endpoints=(NODE0, SECOND), table=TWO_RUNTIMES)
        self.registry = self.snapshot.registry

    def test_presence_cannot_claim_state_the_facts_do_not_support(self):
        base = self.registry.records[0].model_dump()
        broken = dict(base)
        broken['presence'] = ('installed', 'loaded')
        broken['loaded'] = {'kind': 'measured', 'value': False, 'source': 'test', 'observed_at': None}
        with self.assertRaises(ValidationError):
            ModelRecord.model_validate(broken)

    def test_qualified_record_must_carry_its_qualification_id(self):
        base = self.registry.require(SMALL).model_dump()
        broken = dict(base)
        broken['qualification_id'] = {'kind': 'unknown', 'value': None, 'reason': 'missing',
                                      'source': 'test', 'observed_at': None}
        with self.assertRaises(ValidationError):
            ModelRecord.model_validate(broken)

    def test_resident_endpoint_must_also_report_the_model_installed(self):
        base = self.registry.require(SMALL).model_dump()
        broken = dict(base)
        broken['endpoints'] = (NODE0,)
        broken['loaded_endpoints'] = (NODE0, SECOND)
        with self.assertRaises(ValidationError):
            ModelRecord.model_validate(broken)

    def test_duplicate_model_identity_is_rejected(self):
        record = self.registry.records[0].model_dump()
        with self.assertRaises(ValidationError):
            ModelRegistry(records=(record, record), fit=(), runtimes_observed=(NODE0,))

    def test_record_without_fit_analysis_is_rejected(self):
        with self.assertRaises(ValidationError):
            ModelRegistry(records=(self.registry.records[0].model_dump(),),
                          fit=(), runtimes_observed=(NODE0,))

    def test_snapshot_counts_must_match_the_registry(self):
        broken = self.snapshot.model_dump()
        broken['model_count'] = 99
        with self.assertRaises(ValidationError):
            ScoutSnapshot.model_validate(broken)

    def test_snapshot_may_not_claim_a_benchmark_or_a_winner(self):
        broken = self.snapshot.model_dump()
        broken['winners_declared'] = SMALL
        with self.assertRaises(ValidationError):
            ScoutSnapshot.model_validate(broken)
        broken = self.snapshot.model_dump()
        broken['inference_performed'] = True
        with self.assertRaises(ValidationError):
            ScoutSnapshot.model_validate(broken)
        broken = self.snapshot.model_dump()
        broken['models_downloaded'] = 1
        with self.assertRaises(ValidationError):
            ScoutSnapshot.model_validate(broken)
        broken = self.snapshot.model_dump()
        broken['models_deleted'] = 1
        with self.assertRaises(ValidationError):
            ScoutSnapshot.model_validate(broken)

    def test_role_may_not_name_a_model(self):
        broken = self.snapshot.model_dump()
        broken['benchmark_plan']['roles'][0]['assigned_models'] = [SMALL]
        with self.assertRaises(ValidationError):
            ScoutSnapshot.model_validate(broken)

    def test_plan_may_not_be_marked_executed(self):
        broken = self.snapshot.model_dump()
        broken['benchmark_plan']['execution_state'] = 'executed'
        with self.assertRaises(ValidationError):
            ScoutSnapshot.model_validate(broken)

    def test_rankings_and_reasons_stay_withheld(self):
        self.assertTrue(self.registry.rankings_withheld)
        self.assertIsNone(self.snapshot.winners_declared)
        self.assertFalse(self.snapshot.inference_performed)
        self.assertEqual(self.snapshot.models_downloaded, 0)
        self.assertEqual(self.snapshot.models_deleted, 0)
        self.assertTrue(self.registry.reasons_withheld)
        self.assertEqual(self.snapshot.benchmark_plan.assigned_models(), ())

    def test_fact_counts_and_unknown_paths_are_reported(self):
        measured, derived, unknown = self.snapshot.fact_count()
        self.assertGreater(measured, 0)
        self.assertGreater(derived, 0)
        self.assertGreater(unknown, 0)
        self.assertTrue(self.snapshot.unknown_fields())
        for path in self.snapshot.unknown_fields():
            self.assertIn('.', path)

    def test_snapshot_round_trips_through_json(self):
        restored = ScoutSnapshot.model_validate_json(self.snapshot.model_dump_json())
        self.assertEqual(restored.registry.identities(), self.snapshot.registry.identities())
        self.assertEqual(restored.fact_count(), self.snapshot.fact_count())


class BenchmarkDesignTests(unittest.TestCase):
    def setUp(self):
        self.plan = scout(hardware(), table={NODE0: TWO_RUNTIMES[NODE0]}).benchmark_plan

    def test_plan_is_designed_and_never_executed(self):
        self.assertEqual(self.plan.execution_state, 'designed_not_executed')
        self.assertTrue(self.plan.blocked_by)
        self.assertEqual(self.plan.assigned_models(), ())

    def test_every_requested_dimension_is_covered(self):
        self.assertEqual(self.plan.metrics, (
            'load_time', 'time_to_first_token', 'tokens_per_second', 'vram_peak', 'ram_peak',
            'gpu_utilization', 'gpu_temperature', 'gpu_power', 'stability',
            'short_context_quality', 'long_context_behavior', 'structured_json_compliance',
            'tool_calling', 'reasoning', 'coding', 'agent_task_performance',
            'repeated_run_consistency'))

    def test_classes_cover_the_requested_measurement_kinds(self):
        self.assertEqual({item.class_id for item in self.plan.classes},
                         {'startup', 'throughput', 'structured-json', 'tool-calling',
                          'long-context', 'quality-probe'})
        for item in self.plan.classes:
            self.assertGreaterEqual(item.repetitions, 3)
            self.assertTrue(item.fairness_rules)

    def test_roles_are_declared_but_unassigned(self):
        self.assertEqual({item.role_id for item in self.plan.roles},
                         {'FAST', 'GENERAL', 'REASONING', 'CODING', 'RESEARCH', 'TOOL_USE',
                          'STRUCTURED_OUTPUT', 'LONG_CONTEXT', 'VISION'})
        for role in self.plan.roles:
            self.assertEqual(role.assignment_state, 'unassigned')
            self.assertEqual(role.assigned_models, ())
            self.assertTrue(role.evidence_required)
            self.assertLessEqual(set(role.evidence_required), set(self.plan.metrics))

    def test_fairness_rules_forbid_a_winner_taking_the_last_measurement(self):
        rules = ' '.join(self.plan.fairness_rules).lower()
        self.assertIn('never a single best run', rules)
        self.assertIn('same baseline', rules)


class ScoutAgentTests(unittest.TestCase):
    TARGET = AgentModelTarget(model_id='qwen3-4b-node0-359d7dd4bcda', runtime_reference=SMALL,
                              digest_sha256=SMALL_DIGEST, context_tokens=4096)

    def _inference(self):
        class Supervisor:
            class config:
                class qualification:
                    profile = type('P', (), {'base_url': NODE0})()
                    model_reference = SMALL
                    model_store = r'C:\models'
            qualification_id = QUALIFIED
        return type('Inference', (), {'supervisor': Supervisor()})()

    def _run(self, agent=None, events=None):
        agent = agent or DanteModelScout(
            endpoints=(NODE0, SECOND),
            collector=lambda **kwargs: hardware(),
            scout=lambda **kwargs: scout(hardware(), endpoints=(NODE0, SECOND), table=TWO_RUNTIMES))
        definition = agent.definition(self.TARGET)
        payload = AgentTaskPayload(agent_id=definition.agent_id, objective='discover local models',
                                   definition_version=definition.version,
                                   definition_digest=definition_digest(definition))
        captured = events if events is not None else []
        result = agent.execute(self._inference(), None, definition, payload, None,
                               threading.Event(), lambda event, **meta: captured.append((event, meta)))
        return definition, result, captured

    def test_agent_is_registered_with_a_read_only_capability(self):
        definition = self._run()[0]
        self.assertEqual(definition.agent_id, 'dante-model-scout')
        self.assertEqual(definition.capabilities, ('READ_ONLY_INVENTORY',))
        self.assertTrue(DanteModelScout.READ_ONLY)

    def test_agent_is_listed_by_the_default_registry(self):
        registry = build_agent_registry(self.TARGET)
        self.assertEqual([item.agent_id for item in registry.list()],
                         ['dante-hardware', 'dante-model-scout', 'dante-research'])

    def test_agent_performs_no_inference_and_never_touches_the_model_route(self):
        events: list = []
        _, result, captured = self._run(events=events)
        names = [event for event, _ in captured]
        self.assertEqual(names, ['agent.model_scout.collecting', 'agent.model_scout.collected'])
        for event in names:
            self.assertNotIn('inference', event)
        self.assertFalse(result.metadata['inference_performed'])
        self.assertEqual(result.metadata['inference_calls'], 0)
        self.assertEqual(result.metadata['model_identity_source'], 'job_binding')
        self.assertEqual(result.metadata['model_discovery_source'], 'runtime-metadata')
        self.assertTrue(result.metadata['read_only'])
        self.assertFalse(result.metadata['benchmark_executed'])
        self.assertTrue(result.metadata['rankings_withheld'])
        self.assertFalse(result.metadata['winners_declared'])
        self.assertFalse(result.fallback)
        self.assertEqual(result.provider_id, 'ollama')
        self.assertFalse(result.structured_result['inference_performed'])
        self.assertFalse(result.structured_result['benchmark_executed'])
        self.assertTrue(result.structured_result['read_only'])

    def test_agent_result_is_bounded_and_serializable(self):
        from dante.workload import MAX_RESULT_BYTES
        _, result, _ = self._run()
        encoded = json.dumps(result.structured_result, ensure_ascii=False, default=str)
        self.assertLess(len(encoded.encode('utf-8')), MAX_RESULT_BYTES)
        self.assertIn('models=3', result.text)
        self.assertIn('rankings_withheld=true', result.text)

    def test_agent_reports_counts_runtimes_and_unknowns(self):
        _, result, captured = self._run()
        collected = dict(captured)['agent.model_scout.collected']
        self.assertEqual(collected['models'], 3)
        self.assertEqual(collected['loaded'], 1)
        self.assertEqual(collected['qualified'], 1)
        self.assertEqual(collected['runtimes'], 2)
        self.assertEqual(collected['unreachable_runtimes'], 0)
        self.assertGreater(collected['unknown_facts'], 0)
        self.assertEqual(result.metadata['models'], 3)
        self.assertEqual(sorted(result.metadata['runtime_endpoints']), list(BOTH))

    def test_agent_resolves_targets_from_qualified_configuration(self):
        agent = DanteModelScout()
        endpoints, qualified_endpoint, stores, model, qualification = agent._targets(self._inference())
        self.assertEqual(endpoints, (NODE0, 'http://127.0.0.1:11434'))
        self.assertEqual(qualified_endpoint, NODE0)
        self.assertEqual(stores, (__import__('pathlib').Path(r'C:\models'),))
        self.assertEqual(model, SMALL)
        self.assertEqual(qualification, QUALIFIED)

    def test_agent_honours_explicit_wiring_over_configuration(self):
        agent = DanteModelScout(endpoints=(SECOND,), model_stores=())
        targets = agent._targets(self._inference())
        self.assertEqual(targets[0], (SECOND,))
        self.assertEqual(targets[2], ())

    def test_agent_honours_cancellation_before_any_probe(self):
        agent = DanteModelScout()
        definition = agent.definition(self.TARGET)
        payload = AgentTaskPayload(agent_id=definition.agent_id, objective='discover',
                                   definition_version=definition.version,
                                   definition_digest=definition_digest(definition))
        cancelled = threading.Event()
        cancelled.set()
        from dante.inference import InferenceCancelled
        with self.assertRaises(InferenceCancelled):
            agent.execute(self._inference(), None, definition, payload, None, cancelled,
                          lambda event, **meta: None)

    def test_registry_rejects_a_read_only_capability_on_a_mutating_implementation(self):
        class Mutating:
            def execute(self, *args, **kwargs):
                raise AssertionError('must not run')

        definition = DanteModelScout.definition(self.TARGET)
        registry = AgentRegistry()
        registry.register(definition, Mutating())
        runner = AgentRunner(registry, self._inference(), _Store())
        spec = _spec(definition)
        with self.assertRaises(AgentUnavailable):
            runner.execute(_job(), spec, threading.Event())

    def test_registry_rejects_a_job_bound_to_a_different_model(self):
        registry = build_agent_registry(self.TARGET)
        definition = registry.describe('dante-model-scout')
        runner = AgentRunner(registry, self._inference(), _Store())
        spec = _spec(definition)
        spec = spec.model_copy(update={'model': spec.model.model_copy(
            update={'model_id': 'some-other-model'})})
        from dante.workload import QualificationRejected
        with self.assertRaises(QualificationRejected):
            runner.execute(_job(), spec, threading.Event())


class ControlCenterViewTests(unittest.TestCase):
    def setUp(self):
        self.structured = {'scout': scout(hardware(), endpoints=(NODE0, SECOND),
                                          table=TWO_RUNTIMES).model_dump(mode='json')}
        self.view = model_overview(self.structured)

    def test_view_lists_every_model_with_its_state(self):
        self.assertTrue(self.view['read_only'])
        self.assertEqual(self.view['model_count'], 3)
        self.assertEqual(self.view['loaded_count'], 1)
        self.assertEqual(self.view['qualified_count'], 1)
        self.assertEqual(len(self.view['models']), 3)
        ids = [card['model_id'] for card in self.view['models']]
        self.assertEqual(ids, sorted([SMALL, LARGE, DERIVED]))

    def test_view_exposes_facts_as_plain_values_or_reasons(self):
        card = next(item for item in self.view['models'] if item['model_id'] == LARGE)
        self.assertEqual(card['file_size_bytes'], {'kind': 'measured', 'value': 13500737704, 'unit': 'B'})
        self.assertEqual(card['loaded_vram_bytes']['kind'], 'unknown')
        self.assertIsNone(card['loaded_vram_bytes']['value'])
        self.assertTrue(card['loaded_vram_bytes']['reason'])
        self.assertIn('loaded_vram_bytes', card['unknown_fields'])

    def test_view_shows_the_qualified_model_and_its_runtime(self):
        self.assertEqual(self.view['qualified_model'], SMALL)
        self.assertEqual(self.view['qualified_endpoint'], NODE0)
        card = next(item for item in self.view['models'] if item['model_id'] == SMALL)
        self.assertTrue(card['node0_qualified']['value'])
        self.assertEqual(card['qualification_id']['value'], QUALIFIED)
        self.assertEqual(card['loaded_on'], [NODE0])

    def test_view_never_claims_a_winner_or_a_verdict(self):
        self.assertTrue(self.view['rankings_withheld'])
        self.assertIsNone(self.view['winners_declared'])
        self.assertFalse(self.view['inference_performed'])
        self.assertEqual(self.view['models_downloaded'], 0)
        self.assertEqual(self.view['models_deleted'], 0)
        self.assertEqual(self.view['benchmark_plan']['execution_state'], 'designed_not_executed')
        for role in self.view['benchmark_plan']['roles']:
            self.assertEqual(role['assigned_models'], [])

    def test_view_carries_fit_facts_and_unknown_run(self):
        self.assertEqual(self.view['runtimes'], list(BOTH))
        self.assertEqual(self.view['runtimes_unreachable'], [])
        card = next(item for item in self.view['models'] if item['model_id'] == SMALL)
        self.assertTrue(card['fit']['gpu_residency_plausible']['value'])

    def test_rendered_panel_is_bounded_text_without_a_ranking(self):
        text = render_text(self.view)
        self.assertIn('MODEL SCOUT', text)
        self.assertIn('winners_declared=false', text)
        self.assertIn('benchmark-plan-v1 designed_not_executed', text)
        self.assertNotIn('best', text.lower())
        for card in self.view['models']:
            self.assertIn(card['model_id'], text)
        self.assertLess(len(text), 8192)

    def test_view_is_json_serializable_and_rejects_a_missing_snapshot(self):
        json.dumps(self.view, allow_nan=False)
        with self.assertRaises(ValueError):
            model_overview({})
        with self.assertRaises(ValueError):
            model_overview({'scout': {'registry': {}}})

    def test_view_exposes_no_mutating_control(self):
        allowed = {'op', 'agent_id', 'job_id', 'read', 'list', 'describe', 'view'}
        for key in json.dumps(self.view):
            self.assertNotIn('pull', key.lower())
            self.assertNotIn('delete', key.lower())
            self.assertNotIn('load', key.lower())
        self.assertTrue(allowed)


class _Store:
    def __init__(self):
        self.events: list = []

    def record_event(self, job_id, event, attempt_id=None, **metadata):
        self.events.append((job_id, event, metadata))


def _spec(definition):
    from dante.workload import ModelRequirement, WorkloadSpec
    return WorkloadSpec(job_type='AGENT_TASK', prompt='discover local models',
                        model=ModelRequirement(model_id=ScoutAgentTests.TARGET.model_id,
                                               runtime_reference=SMALL,
                                               digest_sha256=SMALL_DIGEST,
                                               context_tokens=4096),
                        max_output_tokens=64,
                        agent_task=AgentTaskPayload(agent_id=definition.agent_id, objective='x',
                                                    definition_version=definition.version,
                                                    definition_digest=definition_digest(definition)))


def _job():
    from dante.workload import JobRecord
    return JobRecord(job_id='job-fixture', state='running', priority=0,
                     created_at=datetime.now(timezone.utc), updated_at=datetime.now(timezone.utc),
                     attempt_count=1)


class ShowAdapterTests(unittest.TestCase):
    """`/api/show` is the only new runtime surface, and it must stay read-only."""

    def _block(self) -> str:
        from pathlib import Path
        text = Path('dante/local_runtime.py').read_text(encoding='utf-8')
        return text.split('def show(self, reference):', 1)[1].split('def complete(', 1)[0]

    def test_show_reads_metadata_and_never_loads_or_mutates(self):
        self.assertIn("self._json('/api/show', {'model': reference.strip()})", self._block())
        for forbidden in ('/api/chat', '/api/generate', '/api/pull', '/api/copy',
                          '/api/delete', '/api/load', 'keep_alive', 'stop', 'kill', 'unload'):
            self.assertNotIn(forbidden, self._block())

    def test_show_is_additive_to_the_existing_adapter(self):
        from dante.local_runtime import OllamaAdapter
        for name in ('version', 'inventory', 'loaded', 'show', 'complete'):
            self.assertTrue(hasattr(OllamaAdapter, name), name)

    def test_show_rejects_a_non_gguf_or_remote_artifact(self):
        from dante.local_runtime import InvalidResponse, OllamaAdapter, PolicyDenied
        adapter = OllamaAdapter.__new__(OllamaAdapter)
        base = {'details': {'format': 'safetensors'}, 'model_info': {}}
        adapter._json = lambda path, payload: base
        with self.assertRaises(PolicyDenied):
            adapter.show('a-model')
        remote = {'details': {'format': 'gguf', 'remote_host': 'example.invalid'},
                  'model_info': {}}
        adapter._json = lambda path, payload: remote
        with self.assertRaises(PolicyDenied):
            adapter.show('a-model')
        adapter._json = lambda path, payload: {'details': {'format': 'gguf'}}
        with self.assertRaises(InvalidResponse):
            adapter.show('a-model')

    def test_show_requires_a_model_reference(self):
        from dante.local_runtime import InvalidRequest, OllamaAdapter
        adapter = OllamaAdapter.__new__(OllamaAdapter)
        for bad in (None, '', '   ', 7, []):
            with self.assertRaises(InvalidRequest):
                adapter.show(bad)


if __name__ == '__main__':
    unittest.main()
