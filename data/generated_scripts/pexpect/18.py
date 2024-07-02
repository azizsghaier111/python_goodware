import os
import pexpect
import unittest
from unittest.mock import Mock, MagicMock

class Shellinabox:
    """A class to encapsulate shell interactions."""

    def __init__(self):
        self.child = None
  
    def startup(self):
        """Spawns a new shell process."""
        try:
            self.child = pexpect.spawn('bash')
        except pexpect.ExceptionPexpect as e:
            print(f'Error while starting the process: {e}')
  
    def send_command(self, command):
        """Sends a command to the shell."""
        try:
            self.child.sendline(command)
        except pexpect.ExceptionPexpect as e:
            print(f'Error while sending command: {e}')
    
    def filter_output(self, filter_list):
        """Compiles a pattern list for output filtering."""
        patterns = '|'.join(filter_list)
        return self.child.compile_pattern_list(patterns)
    
    def check_status(self):
        """Checks if the shell process is alive."""
        try:
            return self.child.isalive()
        except pexpect.ExceptionPexpect as e:
            print(f'Error checking process status: {e}')

class TestShell(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.terminal = Shellinabox()
  
    def test_startup(self):
        self.terminal.startup()
        self.assertIsNotNone(self.terminal.child)

    def test_send_command(self):
        self.terminal.send_command = MagicMock(return_value=None)
        self.terminal.send_command('ls')
        self.terminal.send_command.assert_called_with('ls')

    def test_filter_output(self):
        # create a new instance for testing
        shell = Shellinabox()
        shell.startup()
        shell.send_command('ls')
        # expect these patterns in the output
        patterns = ['Documents', 'Downloads']
        compiled_patterns = shell.filter_output(patterns)
        result = shell.child.expect_list(compiled_patterns)
        # check if the patterns are actually part of the ls output
        self.assertIn(result, patterns)

    def test_check_status(self):
        self.terminal.send_command('sleep 10')
        self.assertTrue(self.terminal.check_status())

if __name__ == '__main__':
    unittest.main()