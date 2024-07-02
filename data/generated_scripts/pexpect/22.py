import unittest
from unittest.mock import patch, create_autospec
import pexpect
import pytorch_lightning as pl
import numpy as np


class ShellProcess(object):
    """
    Shell process class with `pexpect` library for I/O control and Unicode handling for child applications.
    """
    def __init__(self, command):
        self.command = command
        self.child = pexpect.spawn(self.command)

    def start(self):
        """
        Start the shell process. The implementation may vary as per requirement.
        """
        pass

    def is_alive(self):
        """
        Check if the shell process is alive.
        """
        return self.child.isalive()

    def send(self, cmdline):
        """
        Send command line input to the shell process.
        """
        self.child.sendline(cmdline)

    def read_nonblocking(self, size=1000, timeout=2):
        """
        Read non-blocking I/O from the shell process.
        """
        return self.child.read_nonblocking(size, timeout)


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
        """
        Assert that the command is correctly initialized during the starting of the process.
        """
        self.shell_process.start()
        self.assertEqual(self.shell_process.command, self.command)

    def test_is_alive(self):
        """
        Assert that the `is_alive` function calls the underlying function/method from spawned child process correctly.
        """
        self.shell_process.is_alive()
        self.mock_spawn.isalive.assert_called_once()

    def test_send(self):
        """
        Assert that sending command line input to the shell process is working as expected.
        """
        command_to_send = "ls"
        self.shell_process.send(command_to_send)
        self.mock_spawn.sendline.assert_called_once_with(command_to_send)

    # Adding more tests for covering more lines.

    def test_send_with_invalid_command(self):
        """
        Assert that sending invalid command line input to the shell process handles exception correctly.
        """
        command_to_send = ""
        with self.assertRaises(Exception):
            self.shell_process.send(command_to_send)  # Fill up from here the expected behavior.

    def test_read_nonblocking(self):
        """
        Assert that reading non-blocking I/O from the shell process is working as expected.
        """
        self.shell_process.read_nonblocking()
        self.mock_spawn.read_nonblocking.assert_called_once_with(1000, 2)

    def test_read_nonblocking_with_custom_params(self):
        """
        Assert that reading non-blocking I/O with custom parameters from the shell process is working as expected.
        """
        self.shell_process.read_nonblocking(2000, 3)
        self.mock_spawn.read_nonblocking.assert_called_once_with(2000, 3)


if __name__ == '__main__':
    unittest.main()