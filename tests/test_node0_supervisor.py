"""Supervisor state/lifecycle contracts; no Ollama or physical hardware."""
from pathlib import Path
import json
import os
import tempfile
import unittest
from unittest.mock import patch
from uuid import uuid4

from dante.contracts.qualification import QualificationAssessment, QualificationState
from dante.contracts.runtime import RuntimeProfile
from dante.node0_supervisor import Node0Supervisor, RotatingAudit, SupervisorConfig, WindowsMutex
from dante.ollama_qualification import OllamaQualificationConfig, ProbeFailure
from test_local_runtime import model
from test_qualification import identity


class FakeLock:
    def __init__(self, allowed=True): self.allowed, self.acquired, self.releases = allowed, False, 0
    def acquire(self):
        if self.acquired or not self.allowed: return False
        self.acquired = True
        return True
    def release(self):
        if self.acquired: self.releases += 1
        self.acquired = False


class FakeProcess:
    pid = 12345
    alive = True
    def poll(self): return None if self.alive else 1


class FakeRuntime:
    owned = True
    def __init__(self): self.process = FakeProcess(); self.starts = 0; self.stops = 0
    def start(self): self.starts += 1; self.process = FakeProcess()
    def stop(self): self.stops += 1; self.process.alive = False; return True
    def pids(self): return frozenset({self.process.pid}) if self.process.alive else frozenset()


class FakeHTTP:
    def __init__(self, *, digest='a' * 64, version='fixture-1'):
        self.digest, self.version = digest, version
        self.fail = False
        self.malformed_tags = False
    def get(self, path):
        if self.fail: raise ProbeFailure('timeout')
        if path == '/api/version': return {'version': self.version}
        if path == '/api/tags' and self.malformed_tags: return {'models': 'malformed'}
        if path == '/api/tags': return {'models': [{'name': 'fixture:1', 'digest': self.digest,
            'details': {'format': 'gguf', 'quantization_level': 'fixture'}}]}
        raise AssertionError(path)
    def post(self, path, body):
        if self.fail: raise ProbeFailure('timeout')
        return {'done': True}


class FakeObserver:
    def __init__(self, identity): self.identity = identity
    def __call__(self): return self.identity


class FakeStore:
    def __init__(self, identity, state=QualificationState.QUALIFIED):
        self.identity, self.state = identity, state
        self.record = type('Record', (), {'qualification_id': uuid4()})()
    def status(self, identity):
        if identity.fingerprint != self.identity.fingerprint:
            return QualificationAssessment(state=QualificationState.STALE, reasons=('identity_changed',))
        return QualificationAssessment(state=self.state, reasons=())
    def latest(self, identity): return self.record


class FakeNode:
    def __init__(self, identity, state=QualificationState.QUALIFIED):
        self.store = FakeStore(identity, state)
        self.gate = type('Gate', (), {'current': None})()
        self.qualifications = 0
        self.force_qualification_failure = False
        self.continuity = type('Continuity', (), {'observe': lambda *a: None})()
        self.host = type('Host', (), {
            'start': lambda *a: type('Task', (), {'task_id':'task'})(),
            'infer': lambda *a: 'fixture response'})()
    def qualify(self):
        self.qualifications += 1
        if self.force_qualification_failure:
            self.store.state = QualificationState.FAILED
            return self.store.record
        self.store.state = QualificationState.QUALIFIED
        return self.store.record


def build(root: Path, *, state=QualificationState.QUALIFIED, lock=None):
    base = identity()
    gpu = base.machine.gpus[0].model_copy(update={'vendor':'NVIDIA', 'uuid':'GPU-fixture', 'name':'fixture-gpu'})
    machine = base.machine.model_copy(update={'gpus':(gpu,)})
    runtime_obs = base.runtime.model_copy(update={'version':'fixture-1','backend':'cuda'})
    current = base.model_copy(update={'machine':machine,'runtime':runtime_obs,'artifact_sha256':'a'*64,
        'quantization':'fixture','context_tokens':4096})
    qual = OllamaQualificationConfig(profile=RuntimeProfile(runtime='ollama',
        base_url='http://127.0.0.1:11435'), model_reference='fixture:1', model_digest='a'*64,
        quantization='fixture', context_tokens=4096, gpu_uuid='GPU-fixture')
    config = SupervisorConfig(node_uuid=current.machine.node_id, machine_profile='node0',
        ledger_path=root/'ledger.db', evidence_path=root/'evidence', audit_path=root/'audit.jsonl',
        state_path=root/'state.json', config_path=root/'config.json', model=model(), qualification=qual,
        health_interval_s=5, restart_window_s=60, max_restarts=2, restart_backoff_s=1,
        max_backoff_s=2, stable_reset_s=30, audit_max_bytes=65536)
    node = FakeNode(current, state)
    runtime, observer, http = FakeRuntime(), FakeObserver(current), FakeHTTP()
    audit = RotatingAudit(config.audit_path, config.audit_max_bytes)
    instance = Node0Supervisor(config,node,runtime,observer,http,audit,lock=lock or FakeLock())
    instance.stop_event.wait = lambda _delay: False
    return instance, current


class SupervisorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_start_reuses_exact_verified_evidence_and_single_composition_gate(self):
        supervisor, _ = build(self.root)
        self.assertTrue(supervisor.start())
        self.assertEqual(supervisor.state, 'QUALIFIED')
        self.assertTrue(supervisor.snapshot().ready_for_local_routing)
        self.assertIs(supervisor.node.gate.current.__self__, supervisor)
        self.assertEqual(supervisor.node.qualifications, 0)
        self.assertEqual(supervisor.snapshot().runtime_pid, 12345)

    def test_unknown_stale_or_failed_qualification_is_requalified_before_ready(self):
        for state in (QualificationState.UNKNOWN, QualificationState.STALE, QualificationState.FAILED):
            with self.subTest(state=state):
                supervisor, _ = build(self.root / state.value, state=state)
                self.assertTrue(supervisor.start())
                self.assertEqual(supervisor.node.qualifications, 1)
                self.assertTrue(supervisor.snapshot().ready_for_local_routing)

    def test_wrong_digest_fails_closed_without_requalification(self):
        supervisor, _ = build(self.root)
        supervisor.http.digest = 'b' * 64
        self.assertFalse(supervisor.start())
        self.assertFalse(supervisor.snapshot().ready_for_local_routing)
        self.assertEqual(supervisor.state, 'FAILED')
        self.assertEqual(supervisor.node.qualifications, 0)

    def test_malformed_runtime_inventory_fails_closed(self):
        supervisor, _ = build(self.root)
        supervisor.http.malformed_tags = True
        self.assertFalse(supervisor.start())
        self.assertFalse(supervisor.snapshot().ready_for_local_routing)
        self.assertEqual(supervisor.state, 'FAILED')
        self.assertEqual(supervisor.node.qualifications, 0)

    def test_occupied_unowned_endpoint_is_not_stopped_or_adopted(self):
        supervisor, _ = build(self.root)
        supervisor.runtime.process = None
        supervisor.runtime.start = lambda: (_ for _ in ()).throw(ProbeFailure('unavailable'))
        self.assertFalse(supervisor.start())
        self.assertEqual(supervisor.runtime.stops, 0)
        self.assertFalse(supervisor.snapshot().ready_for_local_routing)

    def test_runtime_version_and_context_drift_do_not_reuse_evidence(self):
        supervisor, current = build(self.root)
        self.assertTrue(supervisor.start())
        for identity in (
            current.model_copy(update={'runtime': current.runtime.model_copy(update={'version':'fixture-2'})}),
            current.model_copy(update={'context_tokens':8192}),
        ):
            with self.subTest(identity=identity.runtime.version, context=identity.context_tokens):
                supervisor.observer.identity = identity
                self.assertFalse(supervisor._trust_or_requalify(identity))
                self.assertFalse(supervisor.snapshot().ready_for_local_routing)
        self.assertEqual(supervisor.node.qualifications, 2)

    def test_failed_requalification_never_restores_readiness(self):
        supervisor, _ = build(self.root, state=QualificationState.STALE)
        supervisor.node.force_qualification_failure = True
        self.assertFalse(supervisor.start())
        self.assertEqual(supervisor.node.qualifications, 1)
        self.assertFalse(supervisor.snapshot().ready_for_local_routing)
        self.assertEqual(supervisor.state, 'FAILED')

    def test_endpoint_hang_denies_and_runs_bounded_recovery(self):
        supervisor, _ = build(self.root)
        self.assertTrue(supervisor.start())
        supervisor.http.fail = True
        # A health exception is what the monitor hands to its recovery path.
        with self.assertRaises(ProbeFailure): supervisor.health_check()
        supervisor.http.fail = False
        self.assertTrue(supervisor.recover_once())
        self.assertTrue(supervisor.snapshot().ready_for_local_routing)

    def test_driver_identity_drift_is_not_reused(self):
        supervisor, current = build(self.root)
        self.assertTrue(supervisor.start())
        new_gpu = current.machine.gpus[0].model_copy(update={'driver_version':'new-driver'})
        new_machine = current.machine.model_copy(update={'gpus':(new_gpu,)})
        supervisor.observer.identity = current.model_copy(update={'machine':new_machine})
        # A fresh but materially changed identity cannot use the old record.
        self.assertFalse(supervisor._trust_or_requalify(supervisor.observer.identity))
        self.assertEqual(supervisor.node.qualifications, 1)
        self.assertFalse(supervisor.snapshot().ready_for_local_routing)

    def test_gate_rechecks_supervisor_readiness_to_block_toctou(self):
        supervisor, _ = build(self.root)
        self.assertTrue(supervisor.start())
        self.assertTrue(supervisor.node.gate.current(None).runtime.backend == 'cuda')
        supervisor._ready = False
        with self.assertRaisesRegex(ValueError, 'not ready'):
            supervisor.node.gate.current(None)

    def test_identity_observation_failure_immediately_revokes_readiness(self):
        supervisor, _ = build(self.root)
        self.assertTrue(supervisor.start())
        supervisor.observer = lambda: (_ for _ in ()).throw(ProbeFailure('timeout'))
        with self.assertRaisesRegex(ValueError, 'identity or evidence'):
            supervisor.node.gate.current(None)
        self.assertFalse(supervisor.snapshot().ready_for_local_routing)
        self.assertEqual(supervisor.state, 'DEGRADED')

    def test_readiness_change_during_gate_observation_denies_route(self):
        supervisor, current = build(self.root)
        self.assertTrue(supervisor.start())
        def race_observer():
            supervisor._ready = False
            supervisor.state = 'REQUALIFYING'
            return current
        supervisor.observer = race_observer
        with self.assertRaisesRegex(ValueError, 'changed during'):
            supervisor.node.gate.current(None)

    def test_inference_bridge_uses_shared_host_and_readiness_gate(self):
        supervisor, _ = build(self.root)
        self.assertTrue(supervisor.start())
        self.assertEqual(supervisor.infer('private test prompt'), 'fixture response')
        supervisor._ready = False
        with self.assertRaisesRegex(ValueError, 'not ready'):
            supervisor.infer('must not execute')

    def test_duplicate_instance_exits_without_touching_runtime(self):
        class HeldLock(FakeLock):
            def acquire(self): return False
        supervisor, _ = build(self.root, lock=HeldLock())
        self.assertFalse(supervisor.start())
        self.assertEqual(supervisor.runtime.starts, 0)
        self.assertEqual(supervisor.runtime.stops, 0)

    def test_runtime_failure_denies_then_bounded_restart_requalifies_and_restores(self):
        supervisor, _ = build(self.root)
        self.assertTrue(supervisor.start())
        supervisor.runtime.process.alive = False
        self.assertTrue(supervisor.recover_once())
        self.assertTrue(supervisor.snapshot().ready_for_local_routing)
        self.assertEqual(supervisor.runtime.starts, 2)
        supervisor.runtime.process.alive = False
        self.assertTrue(supervisor.recover_once())
        supervisor.runtime.process.alive = False
        self.assertFalse(supervisor.recover_once())
        self.assertEqual(supervisor.state, 'FAILED')
        self.assertFalse(supervisor.snapshot().ready_for_local_routing)

    def test_shutdown_is_idempotent_and_releases_lock(self):
        lock = FakeLock()
        supervisor, _ = build(self.root, lock=lock)
        self.assertTrue(supervisor.start())
        supervisor.stop(); supervisor.stop()
        self.assertEqual(supervisor.state, 'STOPPED')
        self.assertEqual(lock.releases, 1)
        self.assertFalse(supervisor.snapshot().ready_for_local_routing)
        events=[json.loads(line)['event'] for line in (self.root/'audit.jsonl').read_text().splitlines()]
        self.assertIn('node0.routing.readiness',events)

    def test_rotating_audit_bounds_growth_and_keeps_latest(self):
        audit = RotatingAudit(self.root/'audit.jsonl',65536)
        audit.max_bytes = 100
        for _ in range(20): audit.write('node0.test', value='x'*30)
        self.assertTrue((self.root/'audit.jsonl.1').exists())
        self.assertLessEqual((self.root/'audit.jsonl').stat().st_size, 200)
        self.assertTrue((self.root/'audit.jsonl').read_text().endswith('\n'))

    def test_config_rejects_non_loopback(self):
        with self.assertRaises(Exception):
            RuntimeProfile(runtime='ollama',base_url='http://0.0.0.0:11435')

    @unittest.skipUnless(os.name == 'nt', 'Windows named mutex contract')
    def test_windows_mutex_duplicate_and_stale_release(self):
        name='Local\\DanteNode0Supervisor-test-' + str(uuid4())
        first, second = WindowsMutex(name), WindowsMutex(name)
        self.assertTrue(first.acquire())
        self.assertFalse(second.acquire())
        first.release()
        self.assertTrue(second.acquire())
        second.release()

    @unittest.skipUnless(os.name == 'nt', 'Windows Task Scheduler contract')
    def test_user_startup_registration_is_user_level_and_secret_free(self):
        from dante.node0_supervisor import install_user_startup
        repo=self.root/'repo'; scripts=repo/'scripts'; scripts.mkdir(parents=True)
        (repo/'.venv'/'Scripts').mkdir(parents=True)
        (repo/'.venv'/'Scripts'/'pythonw.exe').write_bytes(b'fixture')
        entry=scripts/'node0_supervisor.py'; entry.write_text('pass')
        config=self.root/'private'/'qualification-config.json'; config.parent.mkdir(); config.write_text('{}')
        with patch('subprocess.run') as run:
            result=install_user_startup(config, repository=repo)
        self.assertEqual(result,'Dante Node0 Supervisor')
        command=run.call_args.args[0]
        self.assertEqual(command[0],'powershell.exe')
        script=command[-1]
        self.assertIn('RunLevel Limited',script)
        self.assertIn('AtLogOn',script)
        self.assertIn('MultipleInstances IgnoreNew',script)
        self.assertIn('RestartCount 3',script)
        self.assertIn('DontStopOnIdleEnd',script)
        self.assertIn('AllowStartIfOnBatteries',script)
        self.assertIn('DontStopIfGoingOnBatteries',script)
        self.assertNotIn('token',script.lower())

    def test_snapshot_contains_no_prompt_or_runtime_response(self):
        supervisor, _ = build(self.root)
        self.assertTrue(supervisor.start())
        serialized=json.dumps(supervisor.snapshot().__dict__)
        self.assertNotIn('fixture response',serialized)
        self.assertNotIn('prompt',serialized.lower())


if __name__ == '__main__':
    unittest.main()
