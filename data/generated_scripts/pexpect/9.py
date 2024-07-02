import unittest
from unittest.mock import patch, create_autospec
import pexpect
import pytorch_lightning as pl
import numpy as np


class ShellProcess(object):
    """
    Shell process class that uses `pexpect` library to control and automate child applications such as SSH, FTP etc
    """
    def __init__(self, command):
        self.command = command
        self.child = pexpect.spawn(self.command)

    def start(self):
        pass

    def is_alive(self):
        return self.child.isalive()

    def send(self, cmdline):
        self.child.sendline(cmdline)

    def read_nonblocking(self):
        return self.child.read_nonblocking(1000, 2)


class TestShellProcess(unittest.TestCase):
    """
    Test class for ShellProcess
    """
    def setUp(self):
        self.command = "bash"
        self.mock_spawn = create_autospec(pexpect.spawn)
        self.shell_process = ShellProcess(self.command)
        self.shell_process.child = self.mock_spawn

    def test_start(self):
        self.shell_process.start()
        self.assertEqual(self.shell_process.command, self.command)

    def test_is_alive(self):
        self.shell_process.is_alive()
        self.mock_spawn.isalive.assert_called_once()

    def test_send(self):
        command_to_send = "ls"
        self.shell_process.send(command_to_send)
        self.mock_spawn.sendline.assert_called_once_with(command_to_send)

    def test_read_nonblocking(self):
        self.shell_process.read_nonblocking()
        self.mock_spawn.read_nonblocking.assert_called_once_with(1000, 2)


if __name__ == '__main__':
    unittest.main()