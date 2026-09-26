"""P7 contract fixtures only. No hardware/runtime/backend is qualified by these tests."""
from contextlib import closing
from datetime import timedelta
import io
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from uuid import UUID, uuid4

from pydantic import ValidationError

from dante.cli import main
from dante.contracts import Task, utc_now
from dante.contracts.qualification import (
    Check, CheckEvidence, GPUProfile, MachineProfile, ModelQualification, PerformanceEvidence,
    QualificationIdentity, QualificationState as State, RuntimeObservation,
)
from dante.ledger import TaskLedger
from dante.node_probe import RuntimeDiscovery, adapter_inventory, probe_machine
from dante.qualification import QualificationRunner, QualificationStore, NodeQualificationGate, assess, changed_inputs
from dante.recovery import digest
from dante.registry import ModelRegistry
from test_local_runtime import model, Wire


NODE = UUID('00000000-0000-4000-8000-000000000001')


def identity():
    return QualificationIdentity(machine=MachineProfile(node_id=NODE, os='FixtureOS', os_version='1',
        architecture='fixture64', cpu='fixture-cpu', logical_cpus=4, physical_cores=2, ram_bytes=8192,
        gpus=(GPUProfile(slot='0', vendor='fixture', name='fixture-gpu', vram_bytes=4096, driver_version='1'),),
        cuda_runtime='fixture-1', cuda_toolkit='fixture-1'),
        runtime=RuntimeObservation(runtime_id='ollama', runtime='ollama', version='fixture-1', backend='fixture',
            configuration_sha256=digest({'fixture': 1}), probe_version='fixture-v1'),
        model_id='local-fixture', artifact_sha256='a' * 64, artifact_kind='runtime_manifest',
        quantization='fixture', context_tokens=4096, configuration_sha256=digest({'fixture': 2}))


def changed(value, **updates):
    return type(value).model_validate({**value.model_dump(), **updates})


def record(current=None, source='synthetic', **updates):
    now = utc_now()
    values = dict(qualification_id=uuid4(), identity=current or identity(), source=source,
        checks=tuple(CheckEvidence(check=check, outcome='passed', evidence_sha256=digest(['fixture', check]))
                     for check in Check), started_at=now, completed_at=now)
    values.update(updates)
    return ModelQualification(**values)


class FakeProbe:
    source = 'synthetic'

    def __init__(self):
        self.current = identity()
        self.calls = []

    def observe(self):
        return self.current

    def run(self, check):
        self.calls.append(check)
        return CheckEvidence(check=check, outcome='passed', evidence_sha256=digest(['fixture', check]))

    def performance(self):
        return PerformanceEvidence()


class QualificationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / 'tasks.db'
        self.ledger = TaskLedger(self.path)
        self.store = QualificationStore(self.ledger)

    def test_profiles_roundtrip_and_unknown_measurements(self):
        original = record()
        self.assertEqual(ModelQualification.model_validate_json(original.model_dump_json()), original)
        self.assertIsNone(original.performance.peak_vram_bytes)
        self.assertIsNone(original.performance.generation_tokens_per_s)
        self.assertIsNone(original.identity.machine.storage_free_bytes)

    def test_invalid_profiles(self):
        for update in ({'ram_bytes': -1}, {'logical_cpus': True}, {'physical_cores': 5},
                       {'node_id': 'hostname'}, {'username': 'fixture'}, {'os': '../private'},
                       {'storage_capacity_bytes': 10, 'storage_free_bytes': 11}, {'observed_at': '2026-01-01'}):
            with self.subTest(update=update), self.assertRaises(ValidationError):
                changed(identity().machine, **update)

    def test_multiple_gpus_order_irrelevant_duplicate_slots_denied(self):
        first = identity()
        gpu = changed(first.machine.gpus[0], slot='1')
        machine = changed(first.machine, gpus=first.machine.gpus + (gpu,))
        before = changed(first, machine=machine)
        after = changed(before, machine=changed(machine, gpus=tuple(reversed(machine.gpus))))
        self.assertEqual(before.fingerprint, after.fingerprint)
        with self.assertRaises(ValidationError):
            changed(machine, gpus=(gpu, gpu))

    def test_synthetic_pass_never_qualifies_hardware(self):
        result = assess(record(), identity())
        self.assertEqual(result.state, State.UNKNOWN)
        self.assertIn('synthetic_evidence_only', result.reasons)

    def test_hardware_evidence_contract_fixture(self):
        # Tests verdict logic with constructed data, not an actual hardware observation.
        result = assess(record(source='hardware'), identity())
        self.assertEqual(result.state, State.QUALIFIED)
        self.assertIn(Check.TOOL_CALLING, result.measured_capabilities)

    def test_missing_required_checks_and_unfinished_run(self):
        for update in ({'checks': ()}, {'completed_at': None}):
            self.assertEqual(assess(record(source='hardware', **update), identity()).state, State.UNKNOWN)

    def test_failure_has_persistent_reason(self):
        failure = CheckEvidence(check=Check.LOAD, outcome='failed', failure='unavailable')
        item = record(checks=(failure,))
        self.store.append(item)
        restored = QualificationStore(TaskLedger(self.path)).get(item.qualification_id)
        self.assertEqual(assess(restored, identity()).state, State.FAILED)
        self.assertEqual(restored.checks[0].failure, 'unavailable')

    def test_unknown_identity_cannot_qualify(self):
        initial = identity()
        variants = [changed(initial, artifact_sha256=None), changed(initial, quantization=None),
                    changed(initial, context_tokens=None), changed(initial, configuration_sha256=None),
                    changed(initial, machine=changed(initial.machine, gpus=None)),
                    changed(initial, runtime=changed(initial.runtime, version=None))]
        for current in variants:
            with self.subTest(current=current):
                self.assertEqual(assess(record(current, source='hardware'), current).state, State.UNKNOWN)

    def test_relevant_identity_changes_are_stale(self):
        initial = identity()
        variants = [changed(initial, artifact_sha256='b' * 64), changed(initial, quantization='other'),
            changed(initial, context_tokens=2048), changed(initial, configuration_sha256='c' * 64),
            changed(initial, suite_version='node0-v2'),
            changed(initial, machine=changed(initial.machine, node_id=uuid4())),
            changed(initial, machine=changed(initial.machine, cuda_runtime='fixture-2')),
            changed(initial, machine=changed(initial.machine, cuda_toolkit='fixture-2')),
            changed(initial, machine=changed(initial.machine, gpus=(changed(initial.machine.gpus[0], name='other'),))),
            changed(initial, machine=changed(initial.machine, gpus=(changed(initial.machine.gpus[0], driver_version='2'),))),
            changed(initial, runtime=changed(initial.runtime, version='fixture-2')),
            changed(initial, runtime=changed(initial.runtime, backend='vulkan')),
            changed(initial, runtime=changed(initial.runtime, configuration_sha256='d' * 64))]
        for current in variants:
            with self.subTest(changes=changed_inputs(initial, current)):
                result = assess(record(initial, source='hardware'), current)
                self.assertEqual(result.state, State.STALE)
                self.assertTrue(result.reasons)
                self.assertNotEqual(initial.fingerprint, current.fingerprint)

    def test_irrelevant_observations_do_not_invalidate(self):
        initial = identity()
        current = changed(initial,
            machine=changed(initial.machine, observed_at=utc_now() + timedelta(days=1), storage_free_bytes=1),
            runtime=changed(initial.runtime, observed_at=utc_now(), model_inventory=('another-model',), capabilities=('new',)))
        self.assertEqual(current.fingerprint, initial.fingerprint)
        self.assertEqual(assess(record(initial, source='hardware'), current).state, State.QUALIFIED)

    def test_backend_matrix_has_no_ranking(self):
        for backend in ('cuda', 'vulkan'):
            current = changed(identity(), runtime=changed(identity().runtime, runtime='llamacpp', backend=backend))
            self.store.append(record(current))
        self.assertEqual(len(self.store.history(NODE)), 2)
        self.assertEqual(len({r.identity.scope for r in self.store.history(NODE)}), 2)

    def test_new_suite_and_schema_fail_closed(self):
        current = changed(identity(), suite_version='node0-v2')
        self.assertIn('unsupported_suite', assess(record(current, source='hardware'), current).reasons)
        with self.assertRaises(ValidationError):
            changed(identity(), schema_version=2)

    def test_evidence_validation(self):
        for update in ({'outcome': 'passed', 'evidence_sha256': None}, {'outcome': 'failed'},
                       {'duration_s': float('nan')}, {'evidence_sha256': 'not-a-hash'}):
            with self.subTest(update=update), self.assertRaises(ValidationError):
                CheckEvidence.model_validate({'check': 'load', 'outcome': 'passed', 'evidence_sha256': 'a' * 64, **update})
        with self.assertRaises(ValidationError):
            record(checks=(record().checks[0], record().checks[0]))
        with self.assertRaises(ValidationError):
            record(completed_at=utc_now() - timedelta(days=1))
        with self.assertRaises(ValidationError):
            PerformanceEvidence(generation_tokens_per_s=float('inf'))

    def test_persistence_across_process_restart(self):
        item = record()
        self.store.append(item)
        code = ('from pathlib import Path; from uuid import UUID; from dante.ledger import TaskLedger; '
                'from dante.qualification import QualificationStore; import sys; '
                'print(QualificationStore(TaskLedger(Path(sys.argv[1]))).get(UUID(sys.argv[2])).model_dump_json())')
        result = subprocess.run([sys.executable, '-c', code, str(self.path), str(item.qualification_id)],
                                capture_output=True, text=True, check=True)
        self.assertEqual(ModelQualification.model_validate_json(result.stdout), item)

    def test_latest_failure_blocks_older_pass(self):
        self.store.append(record(source='hardware'))
        self.store.append(record(interrupted=True))
        self.assertEqual(self.store.status(identity()).state, State.FAILED)
        with self.assertRaises(ValueError):
            self.store.require_qualified(identity())

    def test_get_history_unknown_and_append_only(self):
        self.assertEqual(self.store.status(identity()).state, State.UNKNOWN)
        item = record()
        self.store.append(item)
        with self.assertRaises(sqlite3.IntegrityError):
            self.store.append(item)
        self.assertEqual(self.store.history(NODE), [item])
        self.assertEqual(self.store.history(uuid4()), [])
        with self.assertRaises(KeyError):
            self.store.get(uuid4())

    def test_tampered_evidence_denied(self):
        item = record()
        self.store.append(item)
        with closing(self.ledger._connect()) as db, db:
            db.execute("UPDATE model_qualifications SET payload_digest=?", ('0' * 64,))
        with self.assertRaises(ValueError):
            self.store.status(identity())

    def test_machine_snapshot_deduplication_and_persistence(self):
        machine = identity().machine
        self.assertEqual(self.store.record_machine(machine), self.store.record_machine(machine))
        self.assertEqual(QualificationStore(TaskLedger(self.path)).machines(NODE), [machine])

    def test_runner_synthetic_checks_and_unknown_performance(self):
        probe = FakeProbe()
        item = QualificationRunner(self.store).run(probe)
        self.assertEqual(probe.calls, list(Check))
        self.assertEqual(self.store.status(probe.current).state, State.UNKNOWN)
        self.assertIsNone(item.performance.peak_ram_bytes)
        self.assertEqual(len(self.store.history(NODE)), 2)

    def test_runner_exception_does_not_store_message(self):
        probe = FakeProbe()
        with patch.object(probe, 'run', side_effect=RuntimeError('private exception marker')):
            item = QualificationRunner(self.store).run(probe)
        self.assertTrue(item.interrupted)
        self.assertNotIn('private exception marker', item.model_dump_json())
        self.assertEqual(self.store.status(probe.current).state, State.FAILED)

    def test_runner_crash_leaves_unknown_instead_of_previous_pass(self):
        self.store.append(record(source='hardware'))
        probe = FakeProbe()
        with patch.object(probe, 'run', side_effect=KeyboardInterrupt), self.assertRaises(KeyboardInterrupt):
            QualificationRunner(self.store).run(probe)
        self.assertEqual(QualificationStore(TaskLedger(self.path)).status(probe.current).state, State.UNKNOWN)

    def test_changed_identity_during_run_fails(self):
        probe = FakeProbe()
        with patch.object(probe, 'observe', side_effect=[probe.current, changed(probe.current, artifact_sha256='e' * 64)]):
            item = QualificationRunner(self.store).run(probe)
        self.assertEqual(assess(item, probe.current).state, State.FAILED)

    def test_runtime_discovery_and_unsupported(self):
        discovery = RuntimeDiscovery({'ollama': lambda: identity().runtime})
        self.assertEqual(discovery.discover('ollama').state, 'discovered')
        self.assertEqual(discovery.discover('vllm').state, 'unsupported')
        self.assertEqual(RuntimeDiscovery().discover('ollama').state, 'unsupported')
        discovery = RuntimeDiscovery({'llamacpp': lambda: identity().runtime})
        self.assertEqual(discovery.discover('llamacpp').state, 'invalid')
        def fail():
            raise RuntimeError('private marker')
        self.assertEqual(RuntimeDiscovery({'ollama': fail}).discover('ollama').state, 'unavailable')

    def test_adapter_discovery_uses_fixture_transports_only(self):
        for runtime in ('ollama', 'llamacpp'):
            wire = Wire(runtime)
            observation = changed(identity().runtime, runtime=runtime, version=None, backend=None)
            result = adapter_inventory(wire.adapter(), observation, {'fixture:1': 'public-alias'})
            self.assertEqual(result.model_inventory, ('public-alias',))
            self.assertIsNone(result.version)
            self.assertIsNone(result.backend)
            self.assertNotIn('fixture:1', result.model_dump_json())
            self.assertNotIn('/api/chat', [path for path, _ in wire.calls])

    def test_registry_gate_rejects_synthetic_and_stale(self):
        current = identity()
        gate = NodeQualificationGate(self.store, lambda _: current)
        registry = ModelRegistry([model()], machine_profile='old-pc', qualification_gate=gate)
        self.store.append(record(current))
        with self.assertRaises(ValueError):
            registry.require_automatic(model())
        self.store.append(record(current, source='hardware'))
        registry.require_automatic(model(), tool_use=True)
        current = changed(current, runtime=changed(current.runtime, version='other'))
        with self.assertRaises(ValueError):
            registry.require_automatic(model())

    def test_registry_cannot_reuse_another_models_qualification(self):
        current = identity()
        self.store.append(record(current, source='hardware'))
        gate = NodeQualificationGate(self.store, lambda _: current)
        with self.assertRaises(ValueError):
            gate(model(model_id='other'))
        changed_model = model()
        changed_model.local_metadata.runtime_digest = 'b' * 64
        with self.assertRaises(ValueError):
            gate(changed_model)

    def test_tool_capability_needs_its_own_check(self):
        checks = tuple(item for item in record().checks if item.check != Check.TOOL_CALLING)
        self.store.append(record(source='hardware', checks=checks))
        self.store.require_qualified(identity())
        with self.assertRaises(ValueError):
            self.store.require_qualified(identity(), tool_use=True)

    def test_unknown_markers_and_missing_cuda_runtime_do_not_qualify(self):
        for current in (changed(identity(), quantization='unknown'),
                        changed(identity(), runtime=changed(identity().runtime, version='unverified')),
                        changed(identity(), runtime=changed(identity().runtime, backend='cuda'),
                                machine=changed(identity().machine, cuda_runtime=None))):
            self.assertEqual(assess(record(current, source='hardware'), current).state, State.UNKNOWN)

    def test_runner_cannot_change_evidence_source_mid_run(self):
        probe = FakeProbe()
        def measurements():
            probe.source = 'hardware'
            return PerformanceEvidence()
        with patch.object(probe, 'performance', side_effect=measurements):
            item = QualificationRunner(self.store).run(probe)
        self.assertEqual(item.source, 'synthetic')

    def test_gateway_denies_stale_before_adapter_call(self):
        from dante.inference import InferenceGateway, QualificationDenied
        from dante.contracts import RouteDecision
        from dante.contracts.inference import InferenceRequest
        current = identity()
        self.store.append(record(current, source='hardware'))
        stale = changed(current, runtime=changed(current.runtime, version='fixture-2'))
        registry = ModelRegistry([model()], machine_profile='old-pc',
            qualification_gate=NodeQualificationGate(self.store, lambda _: stale))
        wire = Wire()
        decision = RouteDecision(task_id='fixture-task', trace_id='fixture-trace', selected_model=model(),
            candidates=('local-fixture',), reasons=(), policy_version='fixture')
        with self.assertRaises(QualificationDenied):
            InferenceGateway(registry, [wire.adapter()]).infer(decision,
                InferenceRequest(model=model(), messages=[{'role': 'user', 'content': 'fixture'}]))
        self.assertEqual(wire.calls, [])

    def test_probe_avoids_host_and_environment_collection(self):
        from dante.nvidia_probe import NvidiaObservation
        with patch('dante.node_probe.platform.node', side_effect=AssertionError('must not read hostname')):
            machine = probe_machine(NODE, nvidia_probe=lambda: NvidiaObservation(tool_available=False))
        self.assertEqual(machine.node_id, NODE)
        self.assertIsNone(machine.gpus)
        self.assertIsNone(machine.cuda_runtime)
        for key in ('hostname', 'username', 'environment', 'endpoint', 'path', 'serial'):
            self.assertNotIn(key, machine.model_dump())

    def test_cli_inspect_and_history_and_status(self):
        with patch('dante.node_probe.probe_machine', return_value=identity().machine), patch('sys.stdout', new_callable=io.StringIO) as output:
            self.assertEqual(main(['--db', str(self.path), 'node-inspect', '--node-id', str(NODE)]), 0)
            self.assertEqual(json.loads(output.getvalue())['qualification'], 'unknown')
        self.store.append(record())
        with patch('sys.stdout', new_callable=io.StringIO) as output:
            self.assertEqual(main(['--db', str(self.path), 'node-history', '--node-id', str(NODE)]), 0)
            self.assertEqual(len(json.loads(output.getvalue())), 1)
        current = self.path.parent / 'identity.json'
        current.write_text(identity().model_dump_json(), encoding='utf-8')
        with patch('sys.stdout', new_callable=io.StringIO) as output:
            self.assertEqual(main(['--db', str(self.path), 'node-status', '--identity', str(current)]), 0)
            self.assertEqual(json.loads(output.getvalue())['state'], 'unknown')

    def test_v3_migration_preserves_all_existing_tables(self):
        task = self.ledger.create_task(Task(goal='historical', workspace=str(self.path.parent)))
        with closing(self.ledger._connect()) as db, db:
            db.execute('DROP TABLE node_profiles')
            db.execute('DROP TABLE model_qualifications')
            db.execute('PRAGMA user_version=3')
            names = [r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")]
            before = {name: db.execute('SELECT * FROM "' + name + '"').fetchall() for name in names}
        migrated = TaskLedger(self.path)
        self.assertEqual(migrated.get_task(task.task_id), task)
        with closing(migrated._connect()) as db:
            self.assertEqual(db.execute('PRAGMA user_version').fetchone()[0], 5)
            for name, rows in before.items():
                self.assertEqual(db.execute('SELECT * FROM "' + name + '"').fetchall(), rows)
        QualificationStore(TaskLedger(self.path)).append(record())

    def test_future_database_version_rejected(self):
        with closing(self.ledger._connect()) as db, db:
            db.execute('PRAGMA user_version=99')
        with self.assertRaises(RuntimeError):
            TaskLedger(self.path)


if __name__ == '__main__':
    unittest.main()
