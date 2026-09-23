"""Explicit opt-in physical Node 0 qualification; never used by normal tests."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from uuid import UUID

from dante.contracts import ModelRef, PrivacyClass
from dante.contracts.continuity import BackendState, ContinuityPolicy, ExecutionPolicy
from dante.contracts.qualification import Check, QualificationIdentity, RuntimeObservation
from dante.node0 import build_node0
from dante.node_probe import probe_machine
from dante.nvidia_probe import discover_gpu_uuids
from dante.ollama_qualification import OllamaQualificationConfig
from dante.contracts.runtime import RuntimeProfile
from dante.recovery import digest


def _required(raw, key):
    value = raw[key]
    if value is None or value == '':
        raise ValueError('Missing explicit qualification input')
    return value


def run(config_path: Path) -> dict:
    """Use a fully explicit local config. No model lookup, pulls, or installation."""
    if os.environ.get('DANTE_NODE0_HARDWARE') != '1':
        raise ValueError('Hardware qualification opt-in is required')
    raw = json.loads(config_path.read_text(encoding='utf-8'))
    node_id = UUID(_required(raw, 'node_uuid'))
    model = ModelRef.model_validate(_required(raw, 'model'))
    runtime = _required(raw, 'runtime')
    config = OllamaQualificationConfig(
        profile=RuntimeProfile(runtime='ollama', base_url=_required(runtime, 'endpoint')),
        model_reference=_required(runtime, 'model_reference'),
        model_digest=_required(runtime, 'model_digest'),
        quantization=_required(runtime, 'quantization'),
        context_tokens=_required(runtime, 'context_tokens'),
        gpu_uuid=_required(runtime, 'gpu_uuid'),
        executable=Path(_required(runtime, 'executable')),
        model_store=Path(_required(runtime, 'model_store')))
    if (not model.local or model.provider_id != 'ollama' or model.runtime != 'ollama'
            or model.local_metadata is None
            or model.local_metadata.runtime_reference != config.model_reference
            or model.local_metadata.runtime_digest != config.model_digest
            or model.local_metadata.quantization != config.quantization
            or model.local_metadata.context_tokens != config.context_tokens):
        raise ValueError('Model and qualification configuration differ')
    machine = probe_machine(node_id, uuid_probe=discover_gpu_uuids)
    seed = QualificationIdentity(machine=machine, model_id=model.model_id,
        runtime=RuntimeObservation(runtime_id='ollama', runtime='ollama', backend='cuda',
            probe_version='node0-intent-v1'), artifact_kind='runtime_manifest',
        artifact_sha256=config.model_digest, quantization=config.quantization,
        context_tokens=config.context_tokens, configuration_sha256=digest(raw['runtime']))
    node = build_node0(ledger_path=Path(_required(raw, 'ledger_path')),
        evidence_path=Path(_required(raw, 'evidence_path')),
        audit_path=Path(_required(raw, 'audit_path')),
        seed_identity=seed, qualification=config, models=[model],
        machine_profile=_required(raw, 'machine_profile'),
        policy=ContinuityPolicy(mode=ExecutionPolicy.LOCAL_ONLY))
    result = node.qualify()
    try:
        current = node.probe.observe()
        observation_available = True
    except Exception:
        current = result.identity
        observation_available = False
    assessment = node.store.status(current)
    report = {'node_uuid': str(node_id), 'model_id': model.model_id,
              'qualification_id': str(result.qualification_id),
              'state': assessment.state.value if observation_available else 'failed',
              'fresh_observation_available': observation_available,
              'checks': {check.value: {'outcome': next((item.outcome for item in result.checks
                    if item.check == check), 'unknown'),
                    'failure': next((item.failure for item in result.checks
                    if item.check == check), None),
                    'evidence_sha256': next((item.evidence_sha256 for item in result.checks
                    if item.check == check), None)} for check in Check},
              'route': 'not_attempted', 'inference': 'not_attempted'}
    if observation_available and assessment.state.value == 'qualified':
        node.continuity.observe('ollama', BackendState.AVAILABLE)
        task = node.host.start('Node 0 qualification inference', '.', PrivacyClass.PUBLIC)
        try:
            response = node.host.infer(task.task_id, 'Reply with one short word.', set())
            report['route'] = 'local_ollama'
            report['inference'] = 'passed' if response is not None else 'failed'
        except Exception as exc:
            report['inference'] = type(exc).__name__
    report['result'] = 'PASS' if (observation_available and assessment.state.value == 'qualified'
                                  and report['inference'] == 'passed') else 'FAILED'
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description='Explicit physical Node 0 qualification')
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--run-hardware', action='store_true', required=True)
    args = parser.parse_args(argv)
    try:
        result = run(args.config)
    except Exception as exc:
        # No exception text, paths, prompts, responses or secrets in output.
        result = {'result': 'FAILED', 'error_type': type(exc).__name__}
    print(json.dumps(result, sort_keys=True))
    return 0 if result['result'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
