r"""Read-only unit tests for DanteAI hardware monitoring (stdlib only, temp dirs only).

Run: python C:\DanteAI\monitoring\tests\test_monitor.py
"""
import csv, http.client, json, os, sys, tempfile, threading, time as real_time
from datetime import datetime
from collections import deque
from http.server import ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase, mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import monitor


class FakeTime(SimpleNamespace):
    """Minimal replacement for monitor.time: .time() returns a controlled value."""
    def __init__(self, t):
        self.t = t
    def time(self):
        return self.t


def patch_time(t):
    return mock.patch.object(monitor, "time", FakeTime(t))


def base_data(**over):
    d = {"timestamp": "2026-01-01T00:00:00+00:00",
         "cpu": {"utilization_percent": 5.0, "temperature_c": None, "package_power_w": None},
         "gpu": {"temperature_c": 45.0, "power_w": 60.0, "power_limit_w": 360.0, "utilization_percent": 2.0,
                 "fan_percent": 30.0, "vram_used_mb": 1000.0, "vram_total_mb": 16000.0},
         "memory": {"total_mb": 32000, "available_mb": 16000, "used_mb": 16000, "utilization_percent": 50.0},
         "storage": {"lexar_nm790": {"model": "Lexar SSD NM790 2TB", "health": "Healthy", "temperature_c": None,
                                     "secondary_temperature_c": None}}}
    d.update(over)
    return d


def make_cfg(thresholds, cooldown=900):
    return {"thresholds": thresholds, "alert_cooldown_seconds": cooldown}


def read_csv_rows(f):
    with f.open(newline="", encoding="utf-8") as h:
        return list(csv.DictReader(h))


HISTORY_KEYS = ("cpu.temperature_c", "gpu.temperature_c", "storage.lexar_nm790.temperature_c",
                "gpu.power_w", "gpu.utilization_percent", "gpu.vram_used_mb", "memory.used_mb")


class IsolatedDirsMixin:
    """Redirects monitor.ALERTS/TELEMETRY into a per-test temp dir (project logs untouched)."""
    def setUpDirectories(self):
        tmp = tempfile.TemporaryDirectory()
        self.tmp = tmp
        self.dirs = {}
        for name in ("alerts", "telemetry"):
            p = Path(tmp.name) / name
            p.mkdir()
            self.dirs[name] = p
        p1 = mock.patch.object(monitor, "ALERTS", self.dirs["alerts"]); p1.start()
        p2 = mock.patch.object(monitor, "TELEMETRY", self.dirs["telemetry"]); p2.start()
        self.addCleanup(p1.stop)
        self.addCleanup(p2.stop)
        self.addCleanup(tmp.cleanup)


class TestThresholdEvaluation(TestCase, IsolatedDirsMixin):
    """evaluate(): severity ranking, alert logging, cooldown, recovery."""

    cfg_gpu = {"gpu.temperature_c": {"warning": 80, "critical": 87}}

    def setUp(self):
        self.setUpDirectories()
        self.addCleanup(monitor.active.clear)
        self.addCleanup(monitor.last_alert.clear)
        self.cfg = make_cfg(self.cfg_gpu)

    def files(self):
        return list(self.dirs["alerts"].glob("*.csv"))

    def test_all_normal_no_alert_file(self):
        overall = monitor.evaluate(base_data(), self.cfg)
        self.assertEqual(overall, "NORMAL")
        self.assertEqual(self.files(), [])

    def test_warning_value_logs_warning(self):
        overall = monitor.evaluate(base_data(gpu={"temperature_c": 85.0, "power_w": 60.0}), self.cfg)
        self.assertEqual(overall, "WARNING")
        rows = read_csv_rows(self.files()[0])
        self.assertEqual(rows[0]["sensor"], "gpu.temperature_c")
        self.assertEqual(rows[0]["severity"], "WARNING")
        self.assertEqual(rows[0]["threshold"], "80")
        self.assertEqual(rows[0]["value"], "85.0")

    def test_critical_value_logs_critical(self):
        overall = monitor.evaluate(base_data(gpu={"temperature_c": 92.0, "power_w": 60.0}), self.cfg)
        self.assertEqual(overall, "CRITICAL")
        rows = read_csv_rows(self.files()[0])
        self.assertEqual(rows[0]["severity"], "CRITICAL")
        self.assertEqual(rows[0]["threshold"], "87")

    def test_warning_upgraded_to_critical_writes_both_rows(self):
        monitor.evaluate(base_data(gpu={"temperature_c": 85.0, "power_w": 60.0}), self.cfg)
        overall = monitor.evaluate(base_data(gpu={"temperature_c": 92.0, "power_w": 60.0}), self.cfg)
        self.assertEqual(overall, "CRITICAL")
        rows = read_csv_rows(self.files()[0])
        self.assertEqual([r["severity"] for r in rows], ["WARNING", "CRITICAL"])

    def test_cooldown_suppresses_repeat_warning(self):
        monitor.evaluate(base_data(gpu={"temperature_c": 85.0, "power_w": 60.0}), self.cfg)
        n1 = len(self.files())
        monitor.evaluate(base_data(gpu={"temperature_c": 85.0, "power_w": 60.0}), self.cfg)
        self.assertEqual(len(self.files()), n1)
        self.assertEqual(len(read_csv_rows(self.files()[0])), 1)

    def test_repeat_allowed_after_cooldown(self):
        monitor.evaluate(base_data(gpu={"temperature_c": 85.0, "power_w": 60.0}), self.cfg)
        with patch_time(real_time.time() + 901):
            monitor.evaluate(base_data(gpu={"temperature_c": 85.0, "power_w": 60.0}), self.cfg)
        rows = read_csv_rows(self.files()[0])
        self.assertEqual(len(rows), 2)

    def test_recovery_event_logged(self):
        monitor.evaluate(base_data(gpu={"temperature_c": 85.0, "power_w": 60.0}), self.cfg)
        monitor.evaluate(base_data(gpu={"temperature_c": 45.0, "power_w": 60.0}), self.cfg)
        rows = read_csv_rows(self.files()[0])
        self.assertEqual([r["severity"] for r in rows], ["WARNING", "RECOVERY"])
        self.assertIn("returned to normal", rows[-1]["message"])

    def test_no_recovery_event_when_prior_was_normal(self):
        monitor.evaluate(base_data(gpu={"temperature_c": 45.0, "power_w": 60.0}), self.cfg)
        self.assertEqual(self.files(), [])

    def test_overall_ranking_uses_max_severity(self):
        data = base_data(gpu={"temperature_c": 92.0, "power_w": 60.0})
        data["cpu"]["temperature_c"] = 88.0
        self.assertEqual(monitor.evaluate(data, make_cfg({"gpu.temperature_c": {"warning": 80, "critical": 87},
                                                          "cpu.temperature_c": {"warning": 85, "critical": 95}})), "CRITICAL")


class TestNoneValues(TestCase, IsolatedDirsMixin):
    """Missing/None sensor values must be skipped, never alerted on or crash."""

    def setUp(self):
        self.setUpDirectories()
        self.addCleanup(monitor.active.clear)
        self.addCleanup(monitor.last_alert.clear)
        self.cfg = make_cfg({"cpu.temperature_c": {"warning": 85, "critical": 95},
                             "gpu.temperature_c": {"warning": 80, "critical": 87},
                             "storage.lexar_nm790.temperature_c": {"warning": 70, "critical": 80},
                             "gpu.power_w": {"warning": 330, "critical": 355}})

    def test_none_values_skipped(self):
        data = base_data()
        data["cpu"]["temperature_c"] = None
        data["storage"]["lexar_nm790"]["temperature_c"] = None
        overall = monitor.evaluate(data, self.cfg)
        self.assertEqual(overall, "NORMAL")
        self.assertEqual(list(self.dirs["alerts"].glob("*.csv")), [])

    def test_mixed_none_and_critical(self):
        data = base_data(gpu={"temperature_c": 92.0, "power_w": 60.0})
        data["cpu"]["temperature_c"] = None
        overall = monitor.evaluate(data, self.cfg)
        self.assertEqual(overall, "CRITICAL")
        rows = read_csv_rows(list(self.dirs["alerts"].glob("*.csv"))[0])
        self.assertEqual([r["sensor"] for r in rows], ["gpu.temperature_c"])

    def test_whole_section_missing(self):
        data = base_data(gpu={"temperature_c": 92.0, "power_w": 60.0})
        del data["memory"]
        self.assertEqual(monitor.evaluate(data, self.cfg), "CRITICAL")

    def test_getpath_deep_missing_returns_none(self):
        self.assertIsNone(monitor.getpath(base_data(), "storage.lexar_nm790.temperature_c.missing"))
        self.assertIsNone(monitor.getpath({}, "gpu"))


class TestHistoryBounds(TestCase):
    """In-memory history rings must be bounded and append [ms, value] pairs."""

    keys = HISTORY_KEYS

    def setUp(self):
        self._saved = {k: list(monitor.history.get(k, ())) for k in self.keys}
        for k in self.keys:
            monitor.history[k] = deque(maxlen=3)
        self.addCleanup(self._restore_history)

    def _restore_history(self):
        monitor.history.clear()
        for k in self.keys:
            d = deque(maxlen=900)
            d.extend(self._saved[k])
            monitor.history[k] = d

    def test_bounded_by_maxlen(self):
        for i in range(10):
            for k in self.keys:
                monitor.history[k].append([1000 * i, 1.0])
            self.history_bound = [len(monitor.history[k]) for k in self.keys]
        for k in self.keys:
            self.assertEqual(len(monitor.history[k]), 3)
        self.assertEqual(monitor.history[self.keys[0]][-1][0], 9000)

    def test_none_values_stored(self):
        for _ in range(3):
            for k in self.keys:
                monitor.history[k].append([1000, None])
        for k in self.keys:
            self.assertEqual(list(monitor.history[k])[0][1], None)

    def test_poller_style_roundtrip(self):
        data = base_data(gpu={"temperature_c": 33.0, "power_w": 59.29, "utilization_percent": 3.0,
                              "vram_used_mb": 15677.0, "vram_total_mb": 16303.0})
        point_time = int(real_time.time() * 1000)
        for k, q in monitor.history.items():
            q.append([point_time, monitor.getpath(data, k)])
        snap = {k: list(v) for k, v in monitor.history.items()}
        self.assertIn("storage.lexar_nm790.temperature_c", snap)
        for k in self.keys:
            self.assertEqual(snap[k][-1][0], point_time)
        self.assertIsNone(snap["cpu.temperature_c"][-1][1])
        self.assertEqual(snap["gpu.temperature_c"][-1][1], 33.0)


class TestRetention(TestCase, IsolatedDirsMixin):
    """cleanup(): files older than retention_days are deleted, newer kept, non-CSV kept."""

    def setUp(self):
        self.setUpDirectories()
        now = real_time.time()
        self.old = {}
        self.new = {}
        for i in range(2):
            f_old = self.dirs["telemetry"] / f"telemetry-{i}-old.csv"
            f_new = self.dirs["telemetry"] / f"telemetry-{i}-new.csv"
            f_old.write_text("x", encoding="utf-8")
            f_new.write_text("x", encoding="utf-8")
            self.old[i] = (f_old, now - 31 * 86400)
            self.new[i] = (f_new, now - 2 * 86400)
        a_old = self.dirs["alerts"] / "alerts-old.csv"
        a_old.write_text("x", encoding="utf-8")
        self.old_alert = (a_old, now - 31 * 86400)
        readme = self.dirs["telemetry"] / "notes.txt"
        readme.write_text("keep", encoding="utf-8")
        self.notes = readme
        for f, t in list(self.old.values()) + [self.old_alert] + [self.new[1]]:
            os.utime(f, (t, t))
        os.utime(self.notes, (now - 31 * 86400, now - 31 * 86400))

    def test_cleanup_deletes_old_csv_only(self):
        with patch_time(real_time.time()):
            monitor.cleanup(30)
        for f, _ in list(self.old.values()) + [self.old_alert]:
            self.assertFalse(f.exists(), f"{f} should be deleted")
        for f, _ in self.new.values():
            self.assertTrue(f.exists(), f"{f} should be kept")
        self.assertTrue(self.notes.exists())

    def test_cutoff_boundary_at_exactly_retention(self):
        f = self.dirs["alerts"] / "alerts-edge.csv"
        f.write_text("x", encoding="utf-8")
        edge = real_time.time() - 30 * 86400 - 1
        os.utime(f, (edge, edge))
        with patch_time(real_time.time()):
            monitor.cleanup(30)
        self.assertFalse(f.exists(), "file exactly at retention cutoff should be deleted")

    def test_missing_folders_do_not_crash(self):
        # Point cleanup at a non-existent directory structure via a fresh temp path.
        missing = Path(self.tmp.name) / "does_not_exist"
        with mock.patch.object(monitor, "TELEMETRY", missing), mock.patch.object(monitor, "ALERTS", missing), patch_time(real_time.time()):
            monitor.cleanup(30)  # must not raise


class TestTelemetryLogging(TestCase, IsolatedDirsMixin):
    """telemetry_log(): daily CSV file, header-once, 8-column projection in compact()."""

    def setUp(self):
        self.setUpDirectories()

    def test_writes_daily_file_with_header(self):
        monitor.telemetry_log(base_data())
        expected = self.dirs["telemetry"] / f"telemetry-{datetime.now().strftime('%Y-%m-%d')}.csv"
        self.assertTrue(expected.exists())
        rows = read_csv_rows(expected)
        self.assertEqual(len(rows), 1)
        self.assertEqual(list(rows[0].keys()), ["timestamp", "cpu_temperature_c", "gpu_temperature_c",
                                                "nvme_temperature_c", "gpu_power_w", "gpu_utilization_percent",
                                                "vram_used_mb", "ram_used_mb"])
        self.assertEqual(rows[0]["gpu_temperature_c"], "45.0")
        self.assertEqual(rows[0]["nvme_temperature_c"], "")

    def test_second_write_appends_without_second_header(self):
        monitor.telemetry_log(base_data())
        monitor.telemetry_log(base_data(gpu={"temperature_c": 50.0, "power_w": 61.0, "utilization_percent": 3.0,
                                            "vram_used_mb": 1100.0, "vram_total_mb": 16000.0}))
        expected = self.dirs["telemetry"] / f"telemetry-{datetime.now().strftime('%Y-%m-%d')}.csv"
        text = expected.read_text(encoding="utf-8")
        self.assertEqual(text.count("timestamp,cpu_temperature_c"), 1, "header written exactly once")
        rows = read_csv_rows(expected)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[1]["gpu_temperature_c"], "50.0")

    def test_compact_projections(self):
        c = monitor.compact(base_data(gpu={"temperature_c": 71.5, "power_w": 300.0, "utilization_percent": 10.0,
                                           "vram_used_mb": 2048.0, "vram_total_mb": 16303.0}))
        self.assertEqual(c["timestamp"], "2026-01-01T00:00:00+00:00")
        self.assertEqual(c["cpu_temperature_c"], None)
        self.assertEqual(c["gpu_power_w"], 300.0)
        self.assertEqual(c["vram_used_mb"], 2048.0)
        self.assertEqual(c["ram_used_mb"], 16000)


class TestStorageParse(TestCase):
    """parse_storage(): real Get-PhysicalDisk JSON parsing, fallback, graceful degradation."""

    def test_single_disk_parsing(self):
        raw = json.dumps([{"Model": "Lexar SSD NM790 2TB", "FriendlyName": "Lexar SSD NM790 2TB",
                           "HealthStatus": "Healthy", "OperatingStatus": None,
                           "SerialNumber": "ABC123", "Size": 2048408248320,
                           "BusType": "NVMe", "MediaType": "SSD"}])
        d = monitor.parse_storage(raw)
        self.assertIn("lexar_nm790", d)
        e = d["lexar_nm790"]
        self.assertEqual(e["model"], "Lexar SSD NM790 2TB")
        self.assertEqual(e["health"], "Healthy")
        self.assertAlmostEqual(e["size_gb"], 1907.7, places=1)
        self.assertEqual(e["bus_type"], "NVMe")
        self.assertEqual(e["media_type"], "SSD")
        self.assertIsNone(e["temperature_c"])
        self.assertIn("Get-PhysicalDisk", e["source"])

    def test_dict_payload_wrapped_to_list(self):
        raw = json.dumps({"Model": "Samsung SSD 990", "HealthStatus": "Healthy", "Size": 1024, "BusType": "NVMe"})
        d = monitor.parse_storage(raw)
        self.assertIn("samsung_ssd_990", d)
        self.assertAlmostEqual(d["samsung_ssd_990"]["size_gb"], 0.0)

    def test_operating_status_mapping(self):
        healthy = monitor.parse_storage(json.dumps([{"Model": "Disk 1", "HealthStatus": "Healthy", "OperatingStatus": "OK"}]))["disk_1"]
        self.assertEqual(healthy["health"], "Healthy")
        degraded = monitor.parse_storage(json.dumps([{"Model": "Disk 2", "HealthStatus": "Degraded", "OperatingStatus": "degraded"}]))["disk_2"]
        self.assertEqual(degraded["health"], "Degraded")
        predfail = monitor.parse_storage(json.dumps([{"Model": "Disk 3", "HealthStatus": "Healthy", "OperatingStatus": "Pred_Fail"}]))["disk_3"]
        self.assertEqual(predfail["health"], "Degraded")

    def test_nm790_keeps_canonical_key(self):
        raw = json.dumps([{"Model": "Lexar SSD NM790 2TB", "HealthStatus": "Healthy"}])
        self.assertEqual(monitor.parse_storage(raw).keys() | {"lexar_nm790"}, {"lexar_nm790"})

    def test_non_nm790_disk_gets_slug_key(self):
        raw = json.dumps([{"Model": "Samsung 870 EVO 1TB", "HealthStatus": "Healthy"}])
        d = monitor.parse_storage(raw)
        self.assertIn("samsung_870_evo_1tb", d)

    def test_duplicate_names_get_suffixed_keys(self):
        raw = json.dumps([{"Model": "Same Disk", "HealthStatus": "Healthy"},
                          {"Model": "Same Disk", "HealthStatus": "Healthy"}])
        d = monitor.parse_storage(raw)
        self.assertEqual(sorted(d.keys()), ["same_disk", "same_disk_2"])

    def test_missing_model_and_friendly_name_skipped(self):
        raw = json.dumps([{"Model": None, "FriendlyName": None, "HealthStatus": "Healthy"}])
        self.assertIsNone(monitor.parse_storage(raw))

    def test_null_entries_and_non_dict_entries(self):
        raw = json.dumps([{"Model": "Good", "HealthStatus": "Healthy"}, None, "junk", 42])
        d = monitor.parse_storage(raw)
        self.assertEqual(list(d.keys()), ["good"])

    def test_none_input(self):
        self.assertIsNone(monitor.parse_storage(None))

    def test_invalid_json(self):
        self.assertIsNone(monitor.parse_storage("{not json"))

    def test_empty_lists(self):
        self.assertIsNone(monitor.parse_storage(json.dumps([])))
        self.assertIsNone(monitor.parse_storage(json.dumps({"Model": ""})))

    def test_size_missing_is_none(self):
        d = monitor.parse_storage(json.dumps([{"Model": "NoSize", "HealthStatus": "Healthy"}]))
        self.assertIsNone(d["nosize"]["size_gb"])

    def test_storage_uses_raw_command(self):
        raw = json.dumps([{"Model": "Lexar SSD NM790 2TB", "HealthStatus": "Healthy", "Size": 2048408248320}])
        with mock.patch.object(monitor, "storage_raw", return_value=raw):
            d = monitor.storage()
        self.assertEqual(d["lexar_nm790"]["model"], "Lexar SSD NM790 2TB")

    def test_storage_fallback_when_command_fails(self):
        with mock.patch.object(monitor, "storage_raw", return_value=None):
            d = monitor.storage()
        self.assertIn("lexar_nm790", d)
        self.assertEqual(d["lexar_nm790"]["model"], "Lexar SSD NM790 2TB")
        self.assertEqual(d["lexar_nm790"]["health"], "Healthy")
        self.assertEqual(monitor.storage_fallback(), d)

    def test_storage_command_output_is_read_only_query(self):
        import re as _re
        self.assertTrue(_re.search(r"\bGet-PhysicalDisk\b", monitor.PHYSICAL_DISKS_PS))
        self.assertNotRegex(monitor.PHYSICAL_DISKS_PS, r"Set-|New-|Remove-|Add-|Format-|Update-|Clear-|Stop-|Start-")


class TestStateSerialization(TestCase):
    """current.json persistence and round-trip: compact write, atomic replace, schema parity with /api/status."""

    STATE_SCHEMA_TOP = ("timestamp", "cpu", "gpu", "memory", "storage", "motherboard",
                        "overall_status", "history", "sensor_sources")

    def make_state(self):
        data = base_data(gpu={"temperature_c": 33.0, "power_w": 59.29, "power_limit_w": 360.0, "utilization_percent": 3.0,
                              "fan_percent": 30.0, "vram_used_mb": 15677.0, "vram_total_mb": 16303.0,
                              "vram_utilization_percent": 96.2, "source": "nvidia-smi"})
        data["cpu"] = {"utilization_percent": 2.0, "temperature_c": None, "package_power_w": None,
                       "source": "Windows performance counters; temperature unavailable"}
        data["motherboard"] = {"source": "HWiNFO shared memory unavailable; no values exposed"}
        data["overall_status"] = "NORMAL"
        data["history"] = {k: [[1790381455942, None]] for k in monitor.history}
        data["sensor_sources"] = {"gpu": "nvidia-smi", "cpu_utilization": "Windows PerfFormattedData"}
        return data

    def test_write_and_roundtrip(self):
        with tempfile.TemporaryDirectory() as td:
            state = Path(td) / "current.json"
            tmp = state.with_suffix(".tmp")
            data = self.make_state()
            with mock.patch.object(monitor, "lock", threading.Lock()):
                tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
                os.replace(tmp, state)
            self.assertFalse(tmp.exists(), ".tmp must not remain after os.replace")
            loaded = json.loads(state.read_text(encoding="utf-8"))
            self.assertEqual(loaded, data)
            self.assertTrue(set(loaded.keys()) >= set(self.STATE_SCHEMA_TOP))
            self.assertIsNone(loaded["cpu"]["temperature_c"])
            self.assertEqual(loaded["storage"]["lexar_nm790"]["health"], "Healthy")

    def test_poller_state_write_matches_latest(self):
        data = self.make_state()
        with tempfile.TemporaryDirectory() as td:
            state = Path(td) / "current.json"
            with mock.patch.object(monitor, "STATE", state), mock.patch.object(monitor, "lock", threading.Lock()):
                with monitor.lock:
                    monitor.latest = data
                    tmp = state.with_suffix(".tmp")
                    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
                    os.replace(tmp, state)
            self.assertEqual(json.loads(state.read_text(encoding="utf-8")), monitor.latest)

    def test_nulls_mean_unavailable_not_zero(self):
        data = self.make_state()
        text = json.dumps(data)
        self.assertIn('"temperature_c": null', text)
        self.assertNotIn('"temperature_c": 0', text)

    def test_api_handler_serves_latest(self):
        with tempfile.TemporaryDirectory() as td:
            dashboard = Path(td) / "dashboard"
            dashboard.mkdir()
            (dashboard / "index.html").write_text("<html>test</html>", encoding="utf-8")
            data = self.make_state()
            with mock.patch.object(monitor, "latest", data), \
                 mock.patch.object(monitor, "lock", threading.Lock()), \
                 mock.patch.object(monitor, "WEB", dashboard):
                server = ThreadingHTTPServer(("127.0.0.1", 0), monitor.Handler)
                server.daemon_threads = True
                port = server.server_address[1]
                t = threading.Thread(target=server.serve_forever, daemon=True)
                t.start()
                try:
                    c = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
                    c.request("GET", "/api/status")
                    r = c.getresponse()
                    body = r.read().decode("utf-8")
                    self.assertEqual(r.status, 200)
                    self.assertEqual(r.getheader("Content-Type"), "application/json; charset=utf-8")
                    self.assertEqual(r.getheader("Cache-Control"), "no-store")
                    self.assertEqual(int(r.getheader("Content-Length")), len(body))
                    self.assertEqual(json.loads(body), data)
                    c.request("GET", "/")
                    r2 = c.getresponse()
                    body2 = r2.read().decode("utf-8")
                    self.assertEqual(r2.status, 200)
                    self.assertIn("test", body2)
                finally:
                    server.shutdown()
                    t.join(timeout=5)


if __name__ == "__main__":
    import unittest
    unittest.main(module=__name__, argv=[__file__, "-v"], exit=True)
