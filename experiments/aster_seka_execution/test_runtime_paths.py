import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from runtime_paths import interpreter_path

class RuntimePathTests(unittest.TestCase):
    def test_preserves_environment_alias(self):
        with tempfile.TemporaryDirectory() as d:
            alias=Path(d)/'env'/'bin'/'python'; alias.parent.mkdir(parents=True)
            alias.symlink_to(Path(sys.executable).resolve())
            self.assertEqual(interpreter_path(alias),alias.absolute())
            self.assertNotEqual(interpreter_path(alias),alias.resolve())
    def test_actual_environment_can_import_required_library(self):
        result=subprocess.check_output([str(interpreter_path(sys.executable)),'-c',
            'import sys,torch,json;print(json.dumps({"prefix":sys.prefix,"torch":torch.__version__}))'],text=True)
        self.assertEqual(json.loads(result)['prefix'],sys.prefix)
        self.assertTrue(json.loads(result)['torch'])
    def test_missing_executable_rejected(self):
        with self.assertRaises(FileNotFoundError): interpreter_path('/nonexistent/aster-env/python')

if __name__=='__main__': unittest.main()
