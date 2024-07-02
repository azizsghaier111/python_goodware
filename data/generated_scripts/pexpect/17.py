import os
import pexpect
import unittest
from unittest.mock import patch
from pexpect.exceptions import ExceptionPexpect, EOF

class ShellProcess:
    def __init__(self, command='bash'):
        self.command = command
        self.child = None

    def start(self):
        try:
            self.child = pexpect.spawn(self.command)
            print('ShellProcess started...')
        except ExceptionPexpect as exception:
            raise ShellProcessException(exception)

    def is_alive(self):
        try:
            return self.child.isalive()
        except ExceptionPexpect as exception:
            raise ShellProcessException(exception)

    def send(self, cmd):
        try:
            self.child.sendline(cmd)
        except ExceptionPexpect as exception:
            raise ShellProcessException(exception)

    def read(self):
        try:
            return self.child.read_nonblocking(1000, 2)
        except EOF:
            return ''
        except ExceptionPexpect as exception:
            raise ShellProcessException(exception)

    def terminate(self):
        try:
            self.child.terminate()
            print('ShellProcess terminated...')
        except ExceptionPexpect as exception:
            raise ShellProcessException(exception)


class ShellProcessException(Exception):
    pass


class ShellProcessTest(unittest.TestCase):
    def setUp(self):
        self.shell_process = ShellProcess()

    @patch.object(ShellProcess, 'start')
    def test_shell_process_start(self, mock_start):
        self.shell_process.start()
        mock_start.assert_called()

    @patch.object(ShellProcess, 'is_alive')
    def test_shell_process_is_alive(self, mock_is_alive):
        self.shell_process.is_alive()
        mock_is_alive.assert_called()

    @patch.object(ShellProcess, 'send')
    def test_shell_process_send(self, mock_send):
        cmd = 'ls'
        self.shell_process.send(cmd)
        mock_send.assert_called_with(cmd)

    @patch.object(ShellProcess, 'read')
    def test_shell_process_read(self, mock_read):
        self.shell_process.read()
        mock_read.assert_called()

    @patch.object(ShellProcess, 'terminate')
    def test_shell_process_terminate(self, mock_terminate):
        self.shell_process.terminate()
        mock_terminate.assert_called()

    def tearDown(self):
        self.shell_process = None


if __name__ == '__main__':
    unittest.main()