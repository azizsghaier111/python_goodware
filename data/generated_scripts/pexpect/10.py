import os
import pexpect
import unittest
from unittest.mock import MagicMock

class Shellinabox:

    def __init__(self):
        self.child = None

    def startup(self):
        try:
            self.child = pexpect.spawn('bash')
        except pexpect.ExceptionPexpect as e:
            print(f'An exception occurred while starting the process: {e}')

    def send_command(self, command):
        try:
            self.child.sendline(command)
        except pexpect.ExceptionPexpect as e:
            print(f'An exception occurred while sending command: {e}')

    def filter_output(self, filter_list):
        patterns = '|'.join(filter_list)
        return self.child.compile_pattern_list(patterns)

    def check_status(self):
        try:
            return self.child.isalive()
        except pexpect.ExceptionPexpect as e:
            print(f'An exception occurred while checking the process status: {e}')

class TestShell(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.terminal = Shellinabox()

    def test_startup(self):
        self.terminal.startup()
        self.assertIsNotNone(self.terminal.child)

    @unittest.skip('Mocking')
    def test_send_command(self):
        self.terminal.send_command = MagicMock(return_value=None)
        self.terminal.send_command('ls')
        self.terminal.send_command.assert_called_with('ls')

    def test_filter_output(self):
        patterns = ['file1', 'file2']
        self.terminal.send_command('ls')
        compiled_list = self.terminal.filter_output(patterns)
        result = self.terminal.child.expect_list(compiled_list)
        self.assertIn(result, patterns)

    def test_process_status(self):
        self.terminal.send_command('sleep 10')
        self.assertTrue(self.terminal.check_status())


if __name__ == '__main__':
    unittest.main()