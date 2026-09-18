"""NVIDIA discovery fixtures; tests never invoke a real GPU or nvidia-smi."""
import subprocess
import unittest
from unittest.mock import patch

from dante.nvidia_probe import CommandOutput, NvidiaObservation, discover_nvidia
from dante.node_probe import probe_machine
from test_node_bootstrap import NODE


class Runner:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = []

    def __call__(self, arguments):
        self.calls.append(arguments)
        return self.outputs.pop(0)


class NvidiaProbeTests(unittest.TestCase):
    def test_tool_absence_is_clean_unknown(self):
        with patch('dante.nvidia_probe.shutil.which', return_value=None), \
                patch('dante.nvidia_probe.subprocess.run') as run:
            result = discover_nvidia()
        self.assertFalse(result.tool_available)
        self.assertIsNone(result.gpus)
        self.assertIsNone(result.cuda_driver_api)
        run.assert_not_called()

    def test_system_command_boundary_is_bounded_and_shell_free(self):
        completed = [
            subprocess.CompletedProcess([], 0, '0, NVIDIA GPU, 1024, 1.0\n', ''),
            subprocess.CompletedProcess([], 0, '0, 9.0\n', ''),
            subprocess.CompletedProcess([], 0, 'CUDA Version: 12.4\n', ''),
        ]
        with patch('dante.nvidia_probe.shutil.which', return_value='nvidia-smi.exe'), \
                patch('dante.nvidia_probe.subprocess.run', side_effect=completed) as run:
            result = discover_nvidia()
        self.assertEqual(result.gpus[0].name, 'NVIDIA GPU')
        self.assertEqual(run.call_count, 3)
        first_args, first_options = run.call_args_list[0]
        self.assertEqual(first_args[0][0], 'nvidia-smi.exe')
        self.assertFalse(first_options['shell'])
        self.assertEqual(first_options['timeout'], 5)
        self.assertFalse(first_options['check'])

    def test_two_gpu_inventory_vram_driver_compute_and_cuda(self):
        runner = Runner([
            CommandOutput(0, '0, NVIDIA GeForce RTX Fixture, 16384, 572.83\n'
                             '1, "NVIDIA A100-SXM4-80GB", 81920, 572.83\n'),
            CommandOutput(0, '0, 12.0\n1, 8.0\n'),
            CommandOutput(0, '| NVIDIA-SMI 572.83 Driver Version: 572.83 CUDA Version: 12.8 |\n'),
        ])
        result = discover_nvidia(runner)
        self.assertTrue(result.tool_available)
        self.assertEqual([gpu.slot for gpu in result.gpus], ['0', '1'])
        self.assertEqual(result.gpus[0].name, 'NVIDIA GeForce RTX Fixture')
        self.assertEqual(result.gpus[0].vram_bytes, 16384 * 1024 * 1024)
        self.assertEqual(result.gpus[0].driver_version, '572.83')
        self.assertEqual(result.gpus[0].compute_capability, '12.0')
        self.assertEqual(result.gpus[1].compute_capability, '8.0')
        self.assertEqual(result.cuda_driver_api, 'driver-supported-12.8')
        self.assertEqual(runner.calls[0], ('--query-gpu=index,name,memory.total,driver_version',
                                           '--format=csv,noheader,nounits'))
        self.assertEqual(runner.calls[1], ('--query-gpu=index,compute_cap',
                                           '--format=csv,noheader,nounits'))
        self.assertEqual(runner.calls[2], ())

    def test_compute_or_cuda_unavailable_preserves_base_inventory(self):
        runner = Runner([CommandOutput(0, '0, NVIDIA GPU, 1024, 1.2\n'),
                         CommandOutput(1, '', 'unsupported query'),
                         CommandOutput(1, '', 'unavailable')])
        result = discover_nvidia(runner)
        self.assertEqual(len(result.gpus), 1)
        self.assertIsNone(result.gpus[0].compute_capability)
        self.assertIsNone(result.cuda_driver_api)

    def test_no_devices_is_observed_empty_inventory(self):
        runner = Runner([CommandOutput(0, 'No devices were found\n')])
        result = discover_nvidia(runner)
        self.assertTrue(result.tool_available)
        self.assertEqual(result.gpus, ())
        self.assertEqual(len(runner.calls), 1)

    def test_failed_or_malformed_base_query_is_unknown(self):
        cases = [CommandOutput(1, '', 'driver unavailable'), CommandOutput(0, ''),
                 CommandOutput(0, '0, GPU, not-memory, 1.0\n'),
                 CommandOutput(0, '0, GPU/name, 1024, 1.0\n'),
                 CommandOutput(0, '0, GPU, 1024, 1.0\n0, GPU2, 1024, 1.0\n')]
        for output in cases:
            with self.subTest(output=output):
                result = discover_nvidia(Runner([output]))
                self.assertTrue(result.tool_available)
                self.assertIsNone(result.gpus)

    def test_oversized_output_is_unknown(self):
        result = discover_nvidia(Runner([CommandOutput(0, 'x' * (256 * 1024 + 1))]))
        self.assertTrue(result.tool_available)
        self.assertIsNone(result.gpus)

    def test_command_errors_are_clean_absence(self):
        for error in (OSError('missing'), subprocess.TimeoutExpired('nvidia-smi', 5)):
            def fail(_arguments, error=error):
                raise error
            with self.subTest(error=type(error).__name__):
                result = discover_nvidia(fail)
                self.assertFalse(result.tool_available)
                self.assertIsNone(result.gpus)

    def test_machine_profile_receives_observed_nvidia_fields(self):
        observation = NvidiaObservation(tool_available=True, gpus=discover_nvidia(Runner([
            CommandOutput(0, '0, NVIDIA GPU, 2048, 2.0\n'),
            CommandOutput(0, '0, 9.0\n'), CommandOutput(0, 'CUDA Version: 12.4\n')])).gpus,
            cuda_driver_api='driver-supported-12.4')
        machine = probe_machine(NODE, nvidia_probe=lambda: observation)
        self.assertEqual(machine.gpus[0].vendor, 'NVIDIA')
        self.assertEqual(machine.gpus[0].vram_bytes, 2048 * 1024 * 1024)
        self.assertEqual(machine.gpus[0].compute_capability, '9.0')
        self.assertEqual(machine.cuda_runtime, 'driver-supported-12.4')
        self.assertIsNone(machine.cuda_toolkit)

    def test_machine_profile_fallback_has_no_fake_values(self):
        machine = probe_machine(NODE, nvidia_probe=lambda: NvidiaObservation(tool_available=False))
        self.assertIsNone(machine.gpus)
        self.assertIsNone(machine.cuda_runtime)
        self.assertIsNone(machine.cuda_toolkit)


if __name__ == '__main__':
    unittest.main()
