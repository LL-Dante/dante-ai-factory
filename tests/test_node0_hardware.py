"""Physical test is deliberately skipped in every ordinary offline test run."""
import json
import os
from pathlib import Path
import tempfile
import unittest

from dante.node0_hardware import main


class HardwareHarnessSafetyTests(unittest.TestCase):
    def test_without_opt_in_never_touches_hardware(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = Path(temporary) / 'config.json'
            config.write_text(json.dumps({'node_uuid': 'not even parsed'}), encoding='utf-8')
            previous = os.environ.pop('DANTE_NODE0_HARDWARE', None)
            try:
                self.assertEqual(main(['--config', str(config), '--run-hardware']), 1)
            finally:
                if previous is not None:
                    os.environ['DANTE_NODE0_HARDWARE'] = previous


@unittest.skipUnless(os.environ.get('DANTE_NODE0_HARDWARE') == '1'
                     and os.environ.get('DANTE_NODE0_CONFIG'),
                     'Requires explicit Node 0 physical qualification opt-in and config')
class PhysicalNode0Test(unittest.TestCase):
    def test_actual_hardware_pipeline(self):
        self.assertEqual(main(['--config', os.environ['DANTE_NODE0_CONFIG'],
                               '--run-hardware']), 0)


if __name__ == '__main__':
    unittest.main()
