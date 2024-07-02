import os
import pexpect
import time
import unittest
import numpy as np
from unittest.mock import patch


class ShellProcess:
    def __init__(self, command='bash'):
        self.command = command
        self.child = None

    def start(self):
        self.child = pexpect.spawn(self.command)
        if not self.is_alive():
            raise Exception(f'Failed to start the {self.command} process!')

    def is_alive(self):
        return self.child.isalive()

    def send_line(self, cmd):
        self.child.sendline(cmd)

    def expect(self, pattern, timeout=-1):
        self.child.expect(pattern, timeout)

    def expect_exact(self, pattern_list, timeout=-1, searchwindowsize=-1):
        self.child.expect_exact(pattern_list, timeout, searchwindowsize)

    def set_timeout(self, timeout=-1):
        self.child.timeout = timeout

    def kill(self):
        self.child.kill(0)


class TestShellProcess(unittest.TestCase):
    def setUp(self):
        self.shell_process = ShellProcess()
        self.shell_process.start()

    def tearDown(self):
        self.shell_process.kill()

    @patch("time.sleep")
    def test_shell_process(self, mock_sleep):
        mock_sleep.side_effect = [None] * 100
        
        for i in range(100):
            cmd = f"echo {i}"
            self.shell_process.send_line(cmd)
            self.shell_process.expect(str(i))
            self.assertTrue(mock_sleep.called)


if __name__ == "__main__":
    os.environ['COLORTERM'] = 'truecolor'  # for maintaining color in output
    unittest.main()