import unittest
from types import SimpleNamespace

from dante.control_center_gui import (
    OLLAMA_ENDPOINT,
    build_model_rows,
    build_ollama_status,
    display_fact,
    development_job_metrics,
    development_job_state_label,
    development_queue_lines,
    development_test_status,
    format_bytes,
    format_count,
    elapsed_wall,
    job_label,
    job_counts,
    job_details_text,
    job_group,
    filter_log_jobs,
    hardware_rows,
    job_phase,
    job_submission_text,
    parse_ollama_ps,
    parse_ollama_tags,
    read_opencode_limits,
)


class ControlCenterGuiPresentationTests(unittest.TestCase):
    def test_development_test_status_distinguishes_available_execution_and_result(self):
        self.assertEqual(development_test_status([]), "NOT RUN · bounded RUN_TESTS available")
        base = {"event": "tool.completed", "metadata": {"tool_id": "RUN_TESTS", "status": "success"}}
        self.assertEqual(development_test_status([base]), "RUN_TESTS executed · outcome not reported")
        passed = {**base, "metadata": {**base["metadata"], "tests_passed": True}}
        failed = {**base, "metadata": {**base["metadata"], "tests_passed": False}}
        self.assertEqual(development_test_status([passed]), "PASS")
        self.assertEqual(development_test_status([failed]), "FAIL · test executed")

    def test_queue_presentation_shows_wait_owner_and_unknown_release(self):
        telemetry = {
            'capacity': 1, 'queue_depth': 1, 'position': 1, 'priority': 4,
            'queued_at': '2026-09-27T14:00:00+00:00', 'slot_acquired_at': None,
            'slot_released_at': None, 'slot_released_reason': 'no durable release event is recorded',
            'queue_wait_ms': 125, 'slot_owner_job_id': 'job-owner',
            'slot_owner_worker_id': 'worker-1', 'unknown_reasons': {},
        }
        self.assertEqual(development_job_state_label('queued', telemetry, 'job-waiter'),
                         'QUEUED · WAITING FOR LOCAL MODEL SLOT')
        lines = development_queue_lines(telemetry, 'job-waiter', 'queued')
        self.assertIn('QUEUE DEPTH: 1 · POSITION: 1 · CAPACITY: 1', lines)
        self.assertIn('LOCAL MODEL SLOT OWNER: job-owner · worker-1', lines)
        self.assertIn('CURRENT QUEUE WAIT: 125 ms', lines)
        self.assertIn('SLOT RELEASED AT: Unknown (no durable release event is recorded)', lines)

    def test_development_worker_metrics_use_reported_event_values(self):
        facts = development_job_metrics([
            {"event": "model.completed", "metadata": {"input_tokens": 5, "output_tokens": 2}},
            {"event": "tool.completed", "metadata": {"tool_id": "READ_FILE"}},
            {"event": "model.completed", "metadata": {"input_tokens": 7, "output_tokens": 3}},
        ])
        self.assertEqual(facts, {"model_calls": "2", "tool_calls": "1", "input_tokens": "12",
                                 "output_tokens": "5", "current_operation": "model.completed"})

    def test_development_worker_missing_metrics_stay_unknown(self):
        facts = development_job_metrics([{"event": "model.completed", "metadata": {"input_tokens": None}}])
        self.assertEqual(facts["input_tokens"], "Unknown (not reported)")
        self.assertEqual(facts["output_tokens"], "Unknown (not reported)")

    def test_opencode_settings_expose_only_numeric_local_limits(self):
        config = {
            "provider": {"ollama": {
                "options": {"baseURL": "http://secret.invalid", "timeout": 600000, "headerTimeout": 500000},
                "models": {"dante-qwen-agent:latest": {"limit": {"context": 65536, "output": 4096}}},
            }}
        }
        facts = read_opencode_limits(config)
        self.assertEqual(facts, {
            "timeout": "600000", "header_timeout": "500000", "context": "65536", "output": "4096",
        })
        self.assertNotIn("baseURL", facts)
        self.assertNotIn("secret.invalid", str(facts))

    def test_missing_or_invalid_opencode_limits_are_explicitly_unknown(self):
        config = {"provider": {"ollama": {
            "options": {"timeout": True, "headerTimeout": "600000"},
            "models": {"dante-qwen-agent:latest": {"limit": {"context": 0}}},
        }}}
        self.assertEqual(read_opencode_limits(config), {
            "timeout": "Unknown", "header_timeout": "Unknown", "context": "Unknown", "output": "Unknown",
        })

    def test_job_states_remain_visibly_distinct(self):
        labels = {state: job_label({"state": state})[0] for state in (
            "queued", "claimed", "running", "retry_wait", "timed_out", "failed", "cancelled", "succeeded")}
        self.assertEqual(labels, {
            "queued": "QUEUED", "claimed": "STARTING", "running": "RUNNING",
            "retry_wait": "RETRY_BACKOFF", "timed_out": "TIMEOUT", "failed": "ERROR",
            "cancelled": "CANCELLED", "succeeded": "DONE",
        })

    def test_timeout_failure_code_overrides_ambiguous_state(self):
        self.assertEqual(job_label({"state": "failed", "failure": {"code": "inference_timeout"}})[0], "TIMEOUT")

    def test_unknown_sensor_is_not_rendered_as_zero(self):
        fact = SimpleNamespace(kind="unknown", value=None, reason="sensor unavailable", source="probe")
        self.assertEqual(display_fact(fact), "Unknown (sensor unavailable)")

    def test_measured_sensor_keeps_provenance(self):
        fact = SimpleNamespace(kind="measured", value=42.6, reason=None, source="nvidia-smi")
        self.assertEqual(display_fact(fact, suffix="%"), "42.6%  ·  nvidia-smi")

    def test_running_phase_requires_event_evidence(self):
        job = {"state": "running"}
        self.assertEqual(job_phase(job), "RUNNING")
        self.assertEqual(job_phase(job, [{"event": "agent.inference.started"}]), "MODEL_GENERATING")

    def test_queued_submission_does_not_claim_slot_ownership(self):
        text = job_submission_text({"job_id": "job-1", "state": "queued", "priority": 0, "attempt_count": 0})
        self.assertIn("State: QUEUED", text)
        self.assertIn("Queue position: Unknown (not reported)", text)
        self.assertIn("Model-slot ownership: Unknown (not reported)", text)

    def test_elapsed_wall_uses_created_and_terminal_update_timestamps(self):
        job = {"state": "succeeded", "created_at": "2026-09-26T10:00:00+00:00",
               "updated_at": "2026-09-26T10:09:04+00:00"}
        self.assertEqual(elapsed_wall(job), "00:09:04")

    def test_job_counts_counts_queued_and_active_only(self):
        jobs = [{"state": "queued"}, {"state": "running"}, {"state": "claimed"},
                {"state": "retry_wait"}, {"state": "cancel_requested"},
                {"state": "succeeded"}, {"state": "failed"}, {"state": "queued"}]
        self.assertEqual(job_counts(jobs), (2, 4))

    def test_job_counts_empty_input_is_zero(self):
        self.assertEqual(job_counts([]), (0, 0))
        self.assertEqual(job_counts(None), (0, 0))

    def test_job_counts_ignores_malformed_records(self):
        self.assertEqual(job_counts([None, "x", 42, {"state": "queued"}]), (1, 0))

    def test_job_group_classifies_lifecycle_states(self):
        for state in ("claimed", "running", "retry_wait", "cancel_requested"):
            with self.subTest(state=state):
                self.assertEqual(job_group({"state": state}), "RUNNING")
        expected = {
            "queued": "QUEUED", "succeeded": "COMPLETED", "failed": "FAILED",
            "timed_out": "FAILED", "cancelled": "CANCELLED", "future_state": "UNKNOWN",
        }
        for state, group in expected.items():
            with self.subTest(state=state):
                self.assertEqual(job_group({"state": state}), group)
        self.assertEqual(job_group(None), "UNKNOWN")
        self.assertEqual(job_group("malformed"), "UNKNOWN")

    def test_log_filters_use_only_job_id_and_lifecycle_group(self):
        jobs = [{"job_id": "job-alpha", "state": "running"},
                {"job_id": "job-beta", "state": "queued"},
                {"job_id": "job-gamma", "state": "failed"}]
        self.assertEqual([j["job_id"] for j in filter_log_jobs(jobs, job_id="BETA")], ["job-beta"])
        self.assertEqual([j["job_id"] for j in filter_log_jobs(jobs, group="FAILED")], ["job-gamma"])
        self.assertEqual([j["job_id"] for j in filter_log_jobs(jobs, job_id="job-", group="RUNNING")], ["job-alpha"])

    def test_job_details_keep_unreported_tokens_and_metrics_explicit(self):
        text = job_details_text({"job_id": "job-2", "state": "queued", "priority": 1})
        self.assertIn("Queue position: Unknown (not reported)", text)
        self.assertIn("Context total tokens: Unknown (not reported)", text)
        self.assertIn("TTFT: Unknown (not reported)", text)


class OllamaStatusHelpersTests(unittest.TestCase):
    def test_endpoint_is_fixed_local_ollama(self):
        self.assertEqual(OLLAMA_ENDPOINT, "http://127.0.0.1:11434")

    def test_hardware_rows_show_existing_metrics_and_dev_runtime(self):
        def fact(value):
            return SimpleNamespace(kind="measured", value=value, source="fixture", reason=None)
        cpu = SimpleNamespace(**{name: fact(value) for name, value in {
            "model": "CPU", "physical_cores": 16, "logical_threads": 32,
            "utilization_percent": 12, "temperature_c": 65, "current_clock_mhz": 3600,
        }.items()})
        memory = SimpleNamespace(installed_bytes=fact(32 * 1024 ** 3), available_bytes=fact(12 * 1024 ** 3))
        gpu = SimpleNamespace(name="GPU", **{name: fact(value) for name, value in {
            "utilization_percent": 70, "vram_used_bytes": 8 * 1024 ** 3,
            "vram_free_bytes": 8 * 1024 ** 3, "vram_total_bytes": 16 * 1024 ** 3,
            "temperature_c": 55, "power_draw_w": 250, "power_limit_w": 360,
            "sm_clock_mhz": 1800, "memory_clock_mhz": 11000,
        }.items()})
        volume = SimpleNamespace(mount_point="C:", free_bytes=fact(200 * 1024 ** 3),
                                 size_bytes=fact(1000 * 1024 ** 3))
        store = SimpleNamespace(name="models", exists=fact(True), size_bytes=fact(40 * 1024 ** 3),
                                file_count=fact(3))
        snapshot = SimpleNamespace(cpu=cpu, memory=memory, gpus=(gpu,), volumes=(volume,),
                                   model_stores=(store,))
        rows = hardware_rows(snapshot, {"online": True, "endpoint": OLLAMA_ENDPOINT})
        for key, prefix in ((('GPU', 'SM clock'), '1800 MHz'),
                            (('GPU', 'VRAM used'), '8.0 GiB'),
                            (('C:', 'Free space'), '200.0 GiB'),
                            (('models', 'File count'), '3')):
            row = next(row for row in rows if row[:2] == key)
            self.assertTrue(row[2].startswith(prefix), row)
            self.assertEqual(row[3], "fixture")
        self.assertEqual(rows[-1], ("Development Ollama", "Status", "ONLINE", OLLAMA_ENDPOINT))

    def test_hardware_rows_keeps_runtime_status_when_snapshot_missing(self):
        self.assertEqual(hardware_rows(None, {"online": False}), [
            ("Development Ollama", "Status", "OFFLINE", OLLAMA_ENDPOINT)])
        self.assertEqual(hardware_rows(None, {"online": None}), [
            ("Development Ollama", "Status", "Unknown (not reported)", OLLAMA_ENDPOINT)])

    def test_format_bytes_missing_fact_is_explicit_unknown(self):
        for value in (None, 0, -1, "3.1", True):
            self.assertEqual(format_bytes(value), "Unknown (not reported)")

    def test_format_bytes_rounds_to_gib(self):
        self.assertEqual(format_bytes(round(2 * 1024 ** 3)), "2.0 GiB")
        self.assertEqual(format_bytes(3758096384), "3.5 GiB")

    def test_format_count_missing_fact_is_explicit_unknown(self):
        for value in (None, 0, -4, "4k", False):
            self.assertEqual(format_count(value), "Unknown (not reported)")

    def test_format_count_renders_integers_and_units(self):
        self.assertEqual(format_count(8192), "8192")
        self.assertEqual(format_count(150, unit="s"), "150 s")
        self.assertEqual(format_count(4096), "4096")

    def test_parse_ollama_tags_missing_size_stays_explicit(self):
        rows = parse_ollama_tags({"models": [{"name": "qwen3.5-local:32b"}]})
        self.assertEqual(rows, [{"name": "qwen3.5-local:32b", "size": "Unknown (not reported)",
                                  "quantization": "Unknown (not reported)",
                                  "max_context": "Unknown (not reported)"}])

    def test_parse_ollama_tags_renders_reported_size(self):
        rows = parse_ollama_tags({"models": [{"name": "m", "size": round(6 * 1024 ** 3),
                                               "details": {"quantization_level": "Q3_K_M"}}]})
        self.assertEqual(rows[0]["size"], "6.0 GiB")
        self.assertEqual(rows[0]["quantization"], "Q3_K_M")
        self.assertEqual(rows[0]["max_context"], "Unknown (not reported)")

    def test_parse_ollama_tags_reports_metadata_max_context(self):
        rows = parse_ollama_tags({"models": [{"name": "m", "details": {"context_length": 40960}}]})
        self.assertEqual(rows[0]["max_context"], "40960")

    def test_parse_ollama_ps_missing_facts_stay_explicit(self):
        rows = parse_ollama_ps({"models": [{"name": "qwen3.5-local:32b"}]})
        self.assertEqual(rows, [{"name": "qwen3.5-local:32b", "size_vram": "Unknown (not reported)", "context_length": "Unknown (not reported)"}])

    def test_parse_ollama_ps_renders_reported_facts(self):
        rows = parse_ollama_ps({"models": [{"name": "qwen3.5-local:32b", "size_vram": round(18 * 1024 ** 3), "context_length": 32768}]})
        self.assertEqual(rows[0], {"name": "qwen3.5-local:32b", "size_vram": "18.0 GiB VRAM", "context_length": "32768"})

    def test_build_ollama_status_assembles_read_only_state(self):
        state = build_ollama_status(True, {"models": [{"name": "a"}]}, {"models": [{"name": "a"}]})
        self.assertEqual(state["online"], True)
        self.assertEqual(state["endpoint"], OLLAMA_ENDPOINT)
        self.assertEqual(state["installed"], [{"name": "a", "size": "Unknown (not reported)",
                                                "quantization": "Unknown (not reported)",
                                                "max_context": "Unknown (not reported)"}])
        self.assertEqual(state["resident"], [{"name": "a", "size_vram": "Unknown (not reported)", "context_length": "Unknown (not reported)"}])
        self.assertEqual(state["errors"], [])

    def test_build_ollama_status_keeps_errors_when_offline(self):
        state = build_ollama_status(False, None, None, ["ollama tags: URLError"])
        self.assertEqual(state["online"], False)
        self.assertEqual(state["installed"], [])
        self.assertEqual(state["resident"], [])
        self.assertEqual(state["errors"], ["ollama tags: URLError"])

    def test_parse_helpers_ignores_malformed_payloads(self):
        self.assertEqual(parse_ollama_tags(None), [])
        self.assertEqual(parse_ollama_tags({"models": [None, "x"]}), [])
        self.assertEqual(parse_ollama_ps(None), [])
        self.assertEqual(parse_ollama_ps({"models": [42]}), [])


class ModelRowsContextTests(unittest.TestCase):
    def test_loaded_model_shows_distinct_max_and_runtime_context(self):
        state = build_ollama_status(True,
                                    {"models": [{"name": "m1", "details": {"context_length": 40960}}]},
                                    {"models": [{"name": "m1", "context_length": 8192}]})
        (row,) = build_model_rows(None, state)
        self.assertEqual(row[3], "LOADED")
        self.assertEqual(row[5], "40960", "metadata max context from /api/tags")
        self.assertEqual(row[6], "8192", "runtime context from /api/ps")
        self.assertNotEqual(row[5], row[6])

    def test_not_loaded_model_runtime_context_says_not_loaded_unknown(self):
        state = build_ollama_status(True,
                                    {"models": [{"name": "m1", "details": {"context_length": 40960}}]},
                                    {"models": []})
        (row,) = build_model_rows(None, state)
        self.assertEqual(row[3], "NOT LOADED")
        self.assertEqual(row[5], "40960")
        self.assertEqual(row[6], "NOT LOADED (unknown)")

    def test_missing_metadata_max_context_stays_explicit_unknown(self):
        state = build_ollama_status(True, {"models": [{"name": "m1"}]},
                                    {"models": [{"name": "m1", "context_length": 8192}]})
        (row,) = build_model_rows(None, state)
        self.assertEqual(row[5], "Unknown (not reported)")
        self.assertEqual(row[6], "8192")

    def test_loaded_without_reported_runtime_context_stays_explicit_unknown(self):
        state = build_ollama_status(True,
                                    {"models": [{"name": "m1", "details": {"context_length": 40960}}]},
                                    {"models": [{"name": "m1"}]})
        (row,) = build_model_rows(None, state)
        self.assertEqual(row[5], "40960")
        self.assertEqual(row[6], "Unknown (not reported)")

    def test_offline_keeps_residency_explicit_without_inferring_load(self):
        state = build_ollama_status(False, {"models": [{"name": "m1", "details": {"context_length": 40960}}]}, None)
        (row,) = build_model_rows(None, state)
        self.assertEqual(row[2], "OFFLINE")
        self.assertEqual(row[3], "Unknown (not reported)")
        self.assertEqual(row[5], "40960")
        self.assertEqual(row[6], "Unknown (not reported)")

    def test_inventory_rows_keep_size_column_and_add_configured_context(self):
        size = SimpleNamespace(kind="measured", value=3758096384, reason=None, source="")
        model = SimpleNamespace(runtime_reference="m1", quantization=None, size_bytes=size)
        runtime = SimpleNamespace(endpoint="http://runtime", installed_models=[model])
        snapshot = SimpleNamespace(runtimes=[runtime])
        (row,) = build_model_rows(snapshot, {"online": False, "installed": [], "resident": []})
        self.assertEqual(len(row), 11)
        self.assertEqual(row[8], "3.5 GiB")
        self.assertEqual(row[0], "m1")
        self.assertEqual(row[5], "Unknown (not reported)")
        self.assertEqual(row[6], "Unknown (not reported)")
        self.assertEqual(row[-1], "Not applicable")

    def test_qwen_model_row_shows_configured_opencode_context(self):
        state = build_ollama_status(True, {"models": [{"name": "dante-qwen-agent:latest"}]}, {"models": []})
        (row,) = build_model_rows(None, state, opencode_limits={"context": "65536"})
        self.assertEqual(len(row), 11)
        self.assertEqual(row[5], "Unknown (not reported)")
        self.assertEqual(row[6], "NOT LOADED (unknown)")
        self.assertEqual(row[-1], "65536")

    def test_other_model_row_marks_configured_qwen_context_not_applicable(self):
        state = build_ollama_status(True, {"models": [{"name": "another-model"}]}, {"models": []})
        (row,) = build_model_rows(None, state, opencode_limits={"context": "65536"})
        self.assertEqual(row[-1], "Not applicable")

    def test_no_snapshot_rows_do_not_infer_context(self):
        state = build_ollama_status(True, {"models": [{"name": "m1"}]}, {"models": []})
        (row,) = build_model_rows(None, state)
        self.assertEqual(row[5], "Unknown (not reported)")
        self.assertEqual(row[6], "NOT LOADED (unknown)")


if __name__ == "__main__":
    unittest.main()
