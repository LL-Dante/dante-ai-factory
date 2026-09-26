"""Stage A hardware intelligence: normalized inventory, provenance, read-only agent."""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
import tempfile
import threading
import unittest

from pydantic import ValidationError

from dante.agent_runner import (AgentRegistry, AgentRunner, AgentUnavailable, DanteHardwareAgent,
                                Node0WorkloadExecutor, build_agent_registry, definition_digest)
from dante.contracts.agents import AgentDefinition, AgentModelTarget, AgentTaskPayload
from dante.contracts.hardware import (CpuFacts, Fact, GpuFacts, HardwareCapabilityProfile,
                                      HardwareSnapshot, MemoryFacts, ModelCapacityFact,
                                      RuntimeFacts, SensorCoverage, SystemFacts, VolumeFacts)
from dante.contracts.qualification import GPUProfile
from dante.contracts.runtime import RuntimeProfile
from dante.hardware_inventory import MAX_SENSOR_AGE_S, capability_profile, collect_snapshot
from dante.nvidia_probe import NvidiaObservation
from dante.workload import (ExecutionResult, JobState, MAX_RESULT_BYTES, ModelRequirement,
                            WorkloadOrchestrator, WorkloadSpec, WorkloadStore)

RUNTIME = 'qwen3:4b'
MODEL_ID = 'qwen3-4b-node0-359d7dd4bcda'
DIGEST = '359d7dd4bcdab3d86b87d73ac27966f4dbb9f5efdfcc75d34a8764a09474fae7'
QUALIFICATION = 'qualification-fixture'
GPU = 'GPU-fixture'
NODE0 = 'http://127.0.0.1:11435'
SECOND = 'http://127.0.0.1:11434'

WMI = {
    'os_name': 'Microsoft Windows 11 Pro', 'os_version': '10.0.26200', 'os_build': '26200',
    'architecture': '64 bit', 'last_boot': '2026-09-25T15:03:35+00:00',
    'cpu_name': 'AMD Ryzen 9 9900X 12-Core Processor', 'cpu_cores': 12, 'cpu_threads': 24,
    'cpu_max_mhz': 4400, 'cpu_current_mhz': 4400,
    'mem_modules': [
        {'locator': 'DIMMA2', 'bank': 'P0 CHANNEL A', 'bytes': 17179869184, 'speed': 6000,
         'configured': 6000, 'smbios': 34},
        {'locator': 'DIMMB2', 'bank': 'P0 CHANNEL B', 'bytes': 17179869184, 'speed': 6000,
         'configured': 6000, 'smbios': 34}],
    'disks': [{'index': '0', 'model': 'Lexar SSD NM790 2TB', 'interface': 'SCSI',
               'media': 'Fixed hard disk media', 'size': 2048029360128}],
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
    profiles = (GPUProfile(slot='0', vendor='NVIDIA', name='NVIDIA GeForce RTX 5080',
                           uuid=GPU, vram_bytes=17094934528, compute_capability='12.0',
                           driver_version='591.86'),) if gpus else ()
    return NvidiaObservation(tool_available=True, gpus=profiles, cuda_driver_api='driver-supported-13.1')


class FakeAdapter:
    """Read-only Ollama surface: version, installed inventory, resident models."""
    def __init__(self, profile, *, version='0.34.4', installed=None, resident=None, fail=False):
        self.profile, self.version_value = profile, version
        self.installed, self.resident, self.fail = installed, resident, fail
        self.calls = []
    def version(self):
        self.calls.append('version')
        if self.fail:
            raise ValueError('unreachable')
        return self.version_value
    def inventory(self):
        self.calls.append('inventory')
        if self.fail:
            raise ValueError('unreachable')
        return list(self.installed or [])
    def loaded(self):
        self.calls.append('loaded')
        if self.fail:
            raise ValueError('unreachable')
        return list(self.resident or [])


class FakeNvidiaRun:
    returncode = 0
    stderr = ''
    def __init__(self, text):
        self.stdout = text


def snapshot(**changes) -> HardwareSnapshot:
    base = {'runtime_endpoints': (NODE0,), 'qualified_endpoint': NODE0,
            'wmi_runner': lambda script: dict(WMI),
            'monitoring_fetch': lambda: (monitoring(), 'monitoring-http'),
            'nvidia_runner': lambda args: FakeNvidiaRun(
                '0, 11397, 2797, 14801, 3090, 5, 16\n'),
            'nvidia_probe': nvidia, 'uuid_probe': lambda: {'0': GPU},
            'adapter_factory': lambda profile: FakeAdapter(profile)}
    base.update(changes)
    return collect_snapshot(**base)


class FactContractTests(unittest.TestCase):
    def test_measured_and_derived_carry_a_value_and_no_reason(self):
        for kind in ('measured', 'derived'):
            fact = Fact(kind=kind, value=1, source='test', unit='B')
            self.assertEqual(fact.value, 1)
            self.assertIsNone(fact.reason)
            self.assertIsInstance(fact.observed_at, datetime)

    def test_unknown_must_carry_a_reason_and_never_a_value(self):
        fact = Fact(kind='unknown', source='test', reason='sensor absent')
        self.assertIsNone(fact.value)
        self.assertEqual(fact.reason, 'sensor absent')
        with self.assertRaises(ValidationError):
            Fact(kind='unknown', source='test')
        with self.assertRaises(ValidationError):
            Fact(kind='unknown', source='test', value=3, reason='absent')
        with self.assertRaises(ValidationError):
            Fact(kind='unknown', source='test', value=3, unit='C', reason='absent')

    def test_known_facts_reject_a_reason_and_require_a_value(self):
        with self.assertRaises(ValidationError):
            Fact(kind='measured', source='test')
        with self.assertRaises(ValidationError):
            Fact(kind='derived', source='test', value=1, reason='because')

    def test_fact_constructors_enforce_the_same_rules(self):
        self.assertEqual(Fact.measured(5, source='s', unit='B').kind, 'measured')
        self.assertEqual(Fact.derived(5, source='s').kind, 'derived')
        self.assertEqual(Fact.unknown('gone', source='s').kind, 'unknown')
        with self.assertRaises(ValidationError):
            Fact.unknown('', source='s')


class SystemCpuMemoryDiscoveryTests(unittest.TestCase):
    def test_system_facts_are_measured_with_provenance(self):
        system = snapshot().system
        for name, expected in (('os_name', 'Microsoft Windows 11 Pro'),
                               ('os_version', '10.0.26200'), ('os_build', '26200'),
                               ('architecture', '64 bit'),
                               ('power_plan_guid', '381b4222-f694-41f0-9685-ff5bb260df2e'),
                               ('power_plan_name', 'Balanced')):
            fact = getattr(system, name)
            self.assertEqual(fact.kind, 'measured', name)
            self.assertEqual(fact.value, expected, name)
            self.assertTrue(fact.source)
        self.assertEqual(system.uptime_seconds.kind, 'measured')
        self.assertGreater(system.uptime_seconds.value, 0)
        self.assertEqual(system.uptime_seconds.unit, 's')

    def test_cpu_identity_cores_threads_and_clocks(self):
        cpu = snapshot().cpu
        self.assertEqual(cpu.model.value, 'AMD Ryzen 9 9900X 12-Core Processor')
        self.assertEqual(cpu.physical_cores.value, 12)
        self.assertEqual(cpu.logical_threads.value, 24)
        self.assertEqual(cpu.base_clock_mhz.value, 4400)
        self.assertEqual(cpu.base_clock_mhz.unit, 'MHz')
        self.assertEqual(cpu.current_clock_mhz.value, 4400)
        self.assertLessEqual(cpu.physical_cores.value, cpu.logical_threads.value)

    def test_cpu_utilization_comes_from_monitoring_and_temperature_stays_unknown(self):
        cpu = snapshot().cpu
        self.assertEqual(cpu.utilization_percent.kind, 'measured')
        self.assertEqual(cpu.utilization_percent.value, 63.0)
        self.assertEqual(cpu.temperature_c.kind, 'unknown')
        self.assertIn('temperature_c', cpu.temperature_c.reason)
        self.assertIsNone(cpu.temperature_c.value)

    def test_ram_installed_speed_configuration_and_layout(self):
        memory = snapshot().memory
        self.assertEqual(memory.installed_bytes.value, 34359738368)
        self.assertEqual(memory.installed_bytes.unit, 'B')
        self.assertEqual(memory.module_count.value, 2)
        self.assertEqual(memory.speed_mhz.value, 6000)
        self.assertEqual(memory.configured_speed_mhz.value, 6000)
        self.assertEqual(memory.memory_type.value, 'ddr5')
        self.assertIn('CHANNEL A', memory.channel_layout.value)
        self.assertIn('CHANNEL B', memory.channel_layout.value)
        self.assertEqual(memory.available_bytes.value, 12964 * 1024 * 1024)

    def test_storage_disks_and_volumes_report_capacity_and_free_space(self):
        result = snapshot()
        self.assertEqual(len(result.disks), 1)
        self.assertEqual(result.disks[0].model.value, 'Lexar SSD NM790 2TB')
        self.assertEqual(result.disks[0].size_bytes.value, 2048029360128)
        self.assertEqual(len(result.volumes), 1)
        self.assertEqual(result.volumes[0].mount_point, 'C:')
        self.assertEqual(result.volumes[0].filesystem.value, 'NTFS')
        self.assertEqual(result.volumes[0].free_bytes.value, 1792787107840)
        self.assertLess(result.volumes[0].free_bytes.value, result.volumes[0].size_bytes.value)

    def test_integer_typed_disk_index_is_kept_not_silently_dropped(self):
        # Windows CIM reports Index as an int, while nvidia-smi reports slot as text.
        wmi = dict(WMI, disks=[{'index': 0, 'model': 'Lexar SSD NM790 2TB', 'interface': 'SCSI',
                                'media': 'Fixed hard disk media', 'size': 2048407280640}])
        result = snapshot(wmi_runner=lambda script: wmi)
        self.assertEqual(len(result.disks), 1)
        self.assertEqual(result.disks[0].index, '0')
        self.assertEqual(result.disks[0].size_bytes.value, 2048407280640)
        self.assertEqual(result.unknown_facts(), ('cpu.temperature_c',))

    def test_absent_windows_inventory_yields_unknown_not_invented_values(self):
        result = snapshot(wmi_runner=lambda script: None)
        for fact in result.system.os_name, result.cpu.model, result.memory.installed_bytes:
            self.assertEqual(fact.kind, 'unknown')
            self.assertIsNone(fact.value)
            self.assertTrue(fact.reason)
        self.assertEqual(result.disks, ())
        self.assertEqual(result.volumes, ())


class GpuDiscoveryTests(unittest.TestCase):
    def test_gpu_identity_vram_and_capability(self):
        gpu = snapshot().gpus[0]
        self.assertEqual((gpu.slot, gpu.vendor, gpu.name), ('0', 'NVIDIA', 'NVIDIA GeForce RTX 5080'))
        self.assertEqual(gpu.uuid.value, GPU)
        self.assertEqual(gpu.driver_version.value, '591.86')
        self.assertEqual(gpu.compute_capability.value, '12.0')
        self.assertEqual(gpu.cuda_driver_api.value, 'driver-supported-13.1')
        self.assertEqual(gpu.vram_total_bytes.value, 17094934528)
        self.assertEqual(gpu.vram_total_bytes.unit, 'B')

    def test_vram_free_clocks_and_pcie_state(self):
        gpu = snapshot().gpus[0]
        self.assertEqual(gpu.vram_free_bytes.value, 11397 * 1024 * 1024)
        self.assertEqual(gpu.sm_clock_mhz.value, 2797)
        self.assertEqual(gpu.memory_clock_mhz.value, 14801)
        self.assertEqual(gpu.max_sm_clock_mhz.value, 3090)
        self.assertEqual(gpu.pcie_generation.value, 5)
        self.assertEqual(gpu.pcie_link_width.value, 16)
        self.assertLessEqual(gpu.vram_free_bytes.value, gpu.vram_total_bytes.value)

    def test_gpu_dynamic_sensors_come_from_monitoring(self):
        gpu = snapshot().gpus[0]
        self.assertEqual(gpu.temperature_c.value, 39.0)
        self.assertEqual(gpu.power_draw_w.value, 107.09)
        self.assertEqual(gpu.power_limit_w.value, 360.0)
        self.assertEqual(gpu.utilization_percent.value, 43.0)
        self.assertEqual(gpu.vram_used_bytes.value, 4581 * 1024 * 1024)

    def test_cpu_only_host_reports_no_gpus_and_an_unknown_capacity_dimension(self):
        result = snapshot(nvidia_probe=lambda: NvidiaObservation(tool_available=True, gpus=()))
        self.assertEqual(result.gpus, ())
        profile = capability_profile(result)
        self.assertIn('gpu.vram_total_bytes', [fact.dimension for fact in profile.unknown()])

    def test_unavailable_gpu_tool_yields_no_fabricated_gpu(self):
        result = snapshot(nvidia_probe=lambda: NvidiaObservation(tool_available=False))
        self.assertEqual(result.gpus, ())

    def test_malformed_dynamic_query_degrades_to_unknown_fields(self):
        result = snapshot(nvidia_runner=lambda args: FakeNvidiaRun('0, not-a-number\n'))
        gpu = result.gpus[0]
        for name in ('sm_clock_mhz', 'memory_clock_mhz', 'pcie_generation', 'pcie_link_width'):
            self.assertEqual(getattr(gpu, name).kind, 'unknown', name)
        self.assertEqual(gpu.vram_total_bytes.kind, 'measured')


class MissingSensorTests(unittest.TestCase):
    def test_stale_monitoring_sample_never_presents_itself_as_current(self):
        result = snapshot(monitoring_fetch=lambda: (monitoring(age_s=MAX_SENSOR_AGE_S + 600), 'monitoring-state-file'))
        gpu = result.gpus[0]
        self.assertEqual(gpu.temperature_c.kind, 'unknown')
        self.assertIn('old', gpu.temperature_c.reason)
        self.assertEqual(result.cpu.utilization_percent.kind, 'unknown')
        self.assertEqual(result.memory.available_bytes.kind, 'unknown')
        # Static identity collected elsewhere is unaffected by sensor staleness.
        self.assertEqual(gpu.vram_total_bytes.kind, 'measured')
        self.assertEqual(result.coverage.monitoring_reachable, True)
        self.assertEqual(result.coverage.monitoring_source, 'monitoring-state-file')

    def test_absent_monitoring_leaves_every_sensor_unknown(self):
        result = snapshot(monitoring_fetch=lambda: ({}, 'monitoring-state-absent'))
        self.assertEqual(result.coverage.monitoring_reachable, False)
        self.assertIsNone(result.coverage.monitoring_observed_at)
        for fact in (result.cpu.utilization_percent, result.cpu.temperature_c,
                     result.memory.available_bytes, result.gpus[0].vram_used_bytes,
                     result.gpus[0].temperature_c, result.gpus[0].power_draw_w,
                     result.gpus[0].power_limit_w, result.gpus[0].utilization_percent):
            self.assertEqual(fact.kind, 'unknown')
            self.assertIsNone(fact.value)
            self.assertTrue(fact.reason)

    def test_unreachable_runtime_is_unknown_not_a_fabricated_version(self):
        result = snapshot(adapter_factory=lambda profile: FakeAdapter(profile, fail=True))
        runtime = result.runtimes[0]
        self.assertEqual(runtime.reachable.value, False)
        self.assertEqual(runtime.version.kind, 'unknown')
        self.assertEqual(runtime.installed_models, ())
        self.assertEqual(runtime.loaded_models, ())
        self.assertEqual(runtime.vram_allocated_bytes.kind, 'unknown')

    def test_absent_model_store_is_unknown(self):
        with tempfile.TemporaryDirectory() as temp:
            store = Path(temp) / 'absent-store'
            facts = snapshot(model_stores=(store,)).model_stores[0]
            self.assertEqual(facts.name, 'absent-store')
            self.assertEqual(facts.exists.kind, 'unknown')
            self.assertEqual(facts.file_count.kind, 'unknown')
            self.assertEqual(facts.size_bytes.kind, 'unknown')


class MeasuredDerivedUnknownTests(unittest.TestCase):
    def test_counts_separate_measured_from_unknown(self):
        result = snapshot()
        measured, derived, unknown = result.fact_count()
        self.assertGreater(measured, 20)
        self.assertEqual(derived, 1)
        self.assertGreater(unknown, 0)
        self.assertEqual(measured + derived + unknown, len(list(result.facts())))

    def test_every_fact_declares_a_kind_and_a_source(self):
        for fact in snapshot().facts():
            self.assertIn(fact.kind, {'measured', 'derived', 'unknown'})
            self.assertTrue(fact.source)
            if fact.kind == 'unknown':
                self.assertIsNone(fact.value)
                self.assertTrue(fact.reason)
            else:
                self.assertIsNotNone(fact.value)
                self.assertIsNone(fact.reason)

    def test_unknown_facts_are_addressable_by_path(self):
        stale = snapshot(monitoring_fetch=lambda: (monitoring(age_s=MAX_SENSOR_AGE_S + 600),
                                                   'monitoring-state-file'))
        paths = stale.unknown_facts()
        self.assertIn('cpu.temperature_c', paths)
        self.assertIn('gpus[0].temperature_c', paths)
        self.assertIn('gpus[0].vram_used_bytes', paths)
        self.assertFalse(any(path.startswith('system.') for path in paths))
        for path in paths:
            self.assertNotIn(' ', path)

    def test_derived_facts_are_labelled_not_guessed(self):
        result = snapshot()
        runtime = result.runtimes[0]
        self.assertEqual(runtime.qualified_node0.kind, 'derived')
        self.assertEqual(runtime.qualified_node0.value, True)
        self.assertNotEqual(runtime.qualified_node0.source, runtime.version.source)

    def test_snapshot_is_always_marked_read_only(self):
        self.assertIs(snapshot().read_only, True)


class ModelCapacityViewTests(unittest.TestCase):
    def test_profile_exposes_deterministic_dimensions_only(self):
        profile = capability_profile(snapshot())
        self.assertIsInstance(profile, HardwareCapabilityProfile)
        self.assertIs(profile.recommendations_withheld, True)
        self.assertTrue(profile.reasons_withheld)
        dimensions = {fact.dimension for fact in profile.facts}
        self.assertIn('logical_threads', dimensions)
        self.assertIn('system_memory_bytes', dimensions)
        self.assertIn('gpu[0].vram_total_bytes', dimensions)
        self.assertIn('volume[C:].free_bytes', dimensions)

    def test_profile_carries_no_fit_verdict_or_recommendation(self):
        profile = capability_profile(snapshot())
        for fact in profile.facts:
            self.assertIn(fact.kind, {'measured', 'derived', 'unknown'})
            self.assertNotIn('recommend', fact.dimension)
            self.assertNotIn('fit', fact.dimension)
            self.assertNotIn('suitable', fact.dimension)
        rendered = json.dumps(profile.model_dump(mode='json')).lower()
        for forbidden in ('should_run', 'recommended_model', 'will_fit', 'capable_model'):
            self.assertNotIn(forbidden, rendered)

    def test_profile_records_the_basis_of_every_dimension(self):
        for fact in capability_profile(snapshot()).facts:
            self.assertTrue(fact.basis)
            if fact.kind == 'unknown':
                self.assertTrue(fact.reason)
                self.assertIsNone(fact.value)
            else:
                self.assertIsNotNone(fact.value)

    def test_profile_is_bound_to_its_source_snapshot(self):
        first = capability_profile(snapshot())
        second = capability_profile(snapshot())
        self.assertEqual(first.source_snapshot_digest, second.source_snapshot_digest)
        self.assertEqual(len(first.source_snapshot_digest), 64)

    def test_capacity_fact_constructor_rules_match_fact_rules(self):
        with self.assertRaises(ValidationError):
            ModelCapacityFact(dimension='x', kind='unknown', value=1, reason='r')
        with self.assertRaises(ValidationError):
            ModelCapacityFact(dimension='x', kind='measured')
        self.assertEqual(ModelCapacityFact(dimension='x', kind='unknown', reason='r').value, None)

    def test_dynamic_readings_do_not_change_the_snapshot_digest(self):
        fresh = capability_profile(snapshot()).source_snapshot_digest
        stale = capability_profile(snapshot(
            monitoring_fetch=lambda: (monitoring(age_s=MAX_SENSOR_AGE_S + 600), 'x'))).source_snapshot_digest
        self.assertEqual(fresh, stale)


class RuntimeDiscoveryTests(unittest.TestCase):
    def test_qualified_runtime_reports_version_models_and_vram(self):
        adapter = FakeAdapter(
            RuntimeProfile(runtime='ollama', base_url=NODE0),
            installed=[{'id': 'qwen3:4b', 'digest': 'a' * 64, 'size': 2495613440,
                        'details': {'quantization_level': 'Q4_K_M'}},
                       {'id': 'other:latest', 'digest': 'b' * 64, 'size': 12884901888,
                        'details': {}}],
            resident=[{'id': 'qwen3:4b', 'digest': 'a' * 64, 'size_vram': 3178149969}])
        result = snapshot(adapter_factory=lambda profile: adapter)
        runtime = result.runtimes[0]
        self.assertEqual(runtime.version.value, '0.34.4')
        self.assertEqual(runtime.reachable.value, True)
        self.assertEqual(runtime.qualified_node0.value, True)
        self.assertEqual([m.runtime_reference for m in runtime.installed_models],
                         ['other:latest', 'qwen3:4b'])
        self.assertEqual(runtime.installed_models[1].quantization.value, 'Q4_K_M')
        self.assertEqual(runtime.installed_models[1].size_bytes.value, 2495613440)
        self.assertEqual(runtime.installed_models[0].quantization.kind, 'unknown')
        self.assertEqual([m.runtime_reference for m in runtime.loaded_models], ['qwen3:4b'])
        self.assertEqual(runtime.vram_allocated_bytes.value, 3178149969)
        self.assertEqual(sorted(adapter.calls), ['inventory', 'loaded', 'version'])

    def test_both_runtimes_are_observed_and_only_one_is_qualified(self):
        result = snapshot(runtime_endpoints=(NODE0, SECOND))
        self.assertEqual([r.endpoint for r in result.runtimes], [NODE0, SECOND])
        self.assertEqual([r.qualified_node0.value for r in result.runtimes], [True, False])
        self.assertEqual([r.version.value for r in result.runtimes], ['0.34.4', '0.34.4'])

    def test_runtime_observation_uses_only_read_only_adapter_calls(self):
        adapter = FakeAdapter(RuntimeProfile(runtime='ollama', base_url=NODE0))
        snapshot(adapter_factory=lambda profile: adapter)
        self.assertEqual(adapter.calls, ['version', 'inventory', 'loaded'])
        for forbidden in ('complete', 'complete_cancellable', '_complete', 'stop', 'start'):
            self.assertNotIn(forbidden, adapter.calls)


class HardwareAgentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.db = Path(self.temp.name) / 'workloads.db'
        self.store = WorkloadStore(self.db)
        self.local = FakeInference()
        self.registry = AgentRegistry()
        self.hardware = DanteHardwareAgent(endpoints=(NODE0,), model_stores=(),
                                           collector=lambda **kw: snapshot())
        self.definition = self.hardware.definition(model_target())
        self.registry.register(self.definition, self.hardware)
        self.runner = AgentRunner(self.registry, self.local, self.store)
        self.executor = Node0WorkloadExecutor(self.local, self.runner)

    def spec(self, **changes):
        payload = AgentTaskPayload(
            agent_id='dante-hardware', objective='Inventory the local node.',
            requested_output_tokens=64, definition_version=self.definition.version,
            definition_digest=definition_digest(self.definition))
        data = dict(job_type='AGENT_TASK', model=self.local.target,
                    prompt='registered-agent-task:dante-hardware', agent_task=payload,
                    max_output_tokens=64, thinking=self.definition.thinking,
                    timeout_s=self.definition.timeout_s,
                    maximum_attempts=self.definition.retry_policy.maximum_attempts,
                    retry_base_s=self.definition.retry_policy.retry_base_s,
                    retry_max_s=self.definition.retry_policy.retry_max_s)
        data.update(changes)
        return WorkloadSpec(**data)

    def run_job(self, **changes):
        record = self.store.submit(self.spec(**changes))
        orchestrator = WorkloadOrchestrator(self.store, self.executor, poll_s=0.05,
                                            max_gpu_jobs=1, worker_id='hardware-test')
        try:
            orchestrator.start()
            self.assertTrue(orchestrator.run_once())
        finally:
            orchestrator.stop()
        return self.store.get(str(record.job_id))

    def test_agent_declares_a_read_only_capability_and_never_inference(self):
        definition = self.registry.describe('dante-hardware')
        self.assertEqual(definition.agent_id, 'dante-hardware')
        self.assertEqual(definition.capabilities, ('READ_ONLY_INVENTORY',))
        self.assertNotIn('LOCAL_INFERENCE', definition.capabilities)
        self.assertIs(definition.model_target.local_only, True)
        self.assertEqual(definition.output_token_budget, 64)
        self.assertIs(DanteHardwareAgent.READ_ONLY, True)

    def test_read_only_definition_rejects_a_mutating_implementation(self):
        class Mutating:
            READ_ONLY = False

            def execute(self, *args, **kwargs):
                raise AssertionError('must never run')

        registry = AgentRegistry()
        registry.register(self.definition, Mutating())
        runner = AgentRunner(registry, self.local, self.store)
        spec = self.spec()
        record = self.store.submit(spec)
        with self.assertRaises(AgentUnavailable):
            runner.execute(record, spec, threading.Event())

    def test_agent_performs_no_inference(self):
        saved = self.run_job()
        self.assertEqual(saved.state, JobState.SUCCEEDED)
        self.assertEqual(self.local.calls, [])
        self.assertEqual(saved.result['metadata']['inference_performed'], False)
        self.assertEqual(saved.result['metadata']['inference_calls'], 0)
        self.assertEqual(saved.result['metadata']['model_identity_source'], 'job_binding')

    def test_structured_result_persists_snapshot_and_capability_profile(self):
        saved = self.run_job()
        structured = saved.result['structured_result']
        self.assertIs(structured['read_only'], True)
        self.assertIs(structured['inference_performed'], False)
        snapshot_payload = structured['snapshot']
        self.assertEqual(snapshot_payload['read_only'], True)
        self.assertEqual(snapshot_payload['cpu']['model']['value'], 'AMD Ryzen 9 9900X 12-Core Processor')
        self.assertEqual(snapshot_payload['gpus'][0]['name'], 'NVIDIA GeForce RTX 5080')
        self.assertEqual(snapshot_payload['runtimes'][0]['endpoint'], NODE0)
        profile = structured['capability_profile']
        self.assertIs(profile['recommendations_withheld'], True)
        self.assertEqual(len(profile['source_snapshot_digest']), 64)
        self.assertTrue(profile['facts'])

    def test_persisted_result_respects_the_store_size_bound(self):
        saved = self.run_job()
        rendered = json.dumps(saved.result, separators=(',', ':')).encode('utf-8')
        self.assertLessEqual(len(rendered), MAX_RESULT_BYTES)

    def test_agent_audit_chain_records_collection_without_inference(self):
        saved = self.run_job()
        events = [event['event'] for event in self.store.events(str(saved.job_id))]
        self.assertIn('agent.hardware.collecting', events)
        self.assertIn('agent.hardware.collected', events)
        self.assertNotIn('agent.inference.started', events)
        self.assertNotIn('agent.inference.completed', events)
        collected = [event for event in self.store.events(str(saved.job_id))
                     if event['event'] == 'agent.hardware.collected'][0]
        self.assertEqual(collected['metadata']['probe_version'], 'hardware-v1')
        self.assertGreater(collected['metadata']['measured_facts'], 0)
        self.assertEqual(collected['metadata']['gpus'], 1)

    def test_result_text_is_bounded_and_names_unknowns(self):
        saved = self.run_job()
        text = saved.result['text']
        self.assertIn('read_only=true', text)
        self.assertIn('recommendations_withheld=true', text)
        self.assertIn('unknown_paths=', text)
        self.assertLess(len(text.encode('utf-8')), 4096)

    def test_cancellation_before_collection_produces_no_snapshot(self):
        from dante.inference import InferenceCancelled
        spec = self.spec()
        record = self.store.submit(spec)
        cancelled = threading.Event()
        cancelled.set()
        with self.assertRaises(InferenceCancelled):
            self.runner.execute(record, spec, cancelled)
        self.assertEqual(self.local.calls, [])

    def test_agent_never_writes_to_the_model_store_or_configuration(self):
        store = Path(self.temp.name) / 'store'
        store.mkdir()
        (store / 'a.bin').write_bytes(b'x' * 32)
        before = sorted(path.name for path in store.iterdir())
        self.run_job()
        self.assertEqual(sorted(path.name for path in store.iterdir()), before)

    def test_module_exposes_no_hardware_mutation_helper(self):
        import dante.hardware_inventory as module
        forbidden = ('set_power', 'set_clock', 'set_limit', 'restart', 'shutdown', 'install',
                     'download', 'pull', 'delete', 'unload', 'kill', 'set_active',
                     'powercfg_set', 'write_registry', 'set_qualification')
        exported = [name for name in dir(module) if not name.startswith('_')]
        for name in forbidden:
            self.assertFalse([item for item in exported if name in item.lower()], name)


def model_target() -> AgentModelTarget:
    return AgentModelTarget(model_id=MODEL_ID, runtime_reference=RUNTIME, digest_sha256=DIGEST,
                            context_tokens=4096)


class FakeInference:
    def __init__(self):
        self.calls = []
        self.started = threading.Event()
        from dante.workload import ModelRequirement
        self.target = ModelRequirement(model_id=MODEL_ID, runtime_reference=RUNTIME,
                                       digest_sha256=DIGEST, context_tokens=4096, local_only=True)
    def execute(self, spec, cancellation):
        self.calls.append(spec)
        raise AssertionError('Hardware inventory must not run inference')


class Node0RegressionTests(unittest.TestCase):
    def test_research_agent_still_registers_and_keeps_its_draft_contract(self):
        registry = build_agent_registry(model_target())
        self.assertEqual([item.agent_id for item in registry.list()],
                         ['dante-hardware', 'dante-research'])
        research = registry.describe('dante-research')
        self.assertEqual(research.output_token_budget, 384)
        self.assertEqual(research.timeout_s, 120.0)
        self.assertEqual(research.version, '1.0.0')

    def test_hardware_registration_does_not_change_the_research_definition_digest(self):
        from dante.agent_runner import definition_digest
        target = model_target()
        research = DanteResearchAgentDefinition()
        self.assertEqual(definition_digest(research),
                         definition_digest(DanteResearchAgentDefinition()))

    def test_ollama_adapter_loaded_is_read_only_and_additive(self):
        from dante.local_runtime import OllamaAdapter
        self.assertTrue(hasattr(OllamaAdapter, 'loaded'))
        text = Path('dante/local_runtime.py').read_text(encoding='utf-8')
        block = text.split('def loaded(self):', 1)[1].split('def complete(', 1)[0]
        self.assertIn("self._json('/api/ps')", block)
        for forbidden in ('/api/chat', '/api/generate', 'delete', 'stop', 'kill'):
            self.assertNotIn(forbidden, block)


def DanteResearchAgentDefinition():
    from dante.agent_runner import DanteResearchAgent
    return DanteResearchAgent.definition(model_target())


if __name__ == '__main__':
    unittest.main()
