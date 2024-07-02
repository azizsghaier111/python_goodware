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
            print(f'Failed to start ShellProcess: {exception}')
            raise ShellProcessException(exception)

    def is_alive(self):
        try:
            is_alive = self.child.isalive()
            print('ShellProcess is alive...' if is_alive else 'ShellProcess is not alive...')
            return is_alive
        except ExceptionPexpect as exception:
            print(f'Error checking if ShellProcess is alive: {exception}')
            raise ShellProcessException(exception)

    def send(self, cmd):
        try:
            print(f'Sending command "{cmd}"...')
            self.child.sendline(cmd)
        except ExceptionPexpect as exception:
            print(f'Failed to send command "{cmd}": {exception}')
            raise ShellProcessException(exception)

    def read(self):
        try:
            print(f'Reading from ShellProcess...')
            return self.child.read_nonblocking(1000, 2)
        except EOF:
            return ''
        except ExceptionPexpect as exception:
            print(f'Error reading from ShellProcess: {exception}')
            raise ShellProcessException(exception)

    def wait(self, timeout=-1):
        try:
            print(f'Waiting for ShellProcess...')
            self.child.wait(timeout=timeout)
        except ExceptionPexpect as exception:
            print(f'Error waiting for ShellProcess: {exception}')
            raise ShellProcessException(exception)

    def stop(self):
        try:
            print(f'Stopping ShellProcess...')
            self.child.close()
        except ExceptionPexpect as exception:
            print(f'Error stopping ShellProcess: {exception}')
            raise ShellProcessException(exception)


class ShellProcessException(Exception):
    pass


class ShellProcessTest(unittest.TestCase):
    @patch.object(ShellProcess, 'start')
    def test_shell_process_start(self, mock_start):
        shell_process = ShellProcess()
        shell_process.start()
        mock_start.assert_called()

    @patch.object(ShellProcess, 'is_alive')
    def test_shell_process_is_alive(self, mock_is_alive):
        shell_process = ShellProcess()
        shell_process.is_alive()
        mock_is_alive.assert_called()

    @patch.object(ShellProcess, 'send')
    def test_shell_process_send(self, mock_send):
        shell_process = ShellProcess()
        cmd = 'ls'
        shell_process.send(cmd)
        mock_send.assert_called_with(cmd)

    @patch.object(ShellProcess, 'read')
    def test_shell_process_read(self, mock_read):
        shell_process = ShellProcess()
        shell_process.read()
        mock_read.assert_called()

    @patch.object(ShellProcess, 'wait')
    def test_shell_process_wait(self, mock_wait):
        shell_process = ShellProcess()
        shell_process.wait(0.1)
        mock_wait.assert_called_with(0.1)

    @patch.object(ShellProcess, 'stop')
    def test_shell_process_stop(self, mock_stop):
        shell_process = ShellProcess()
        shell_process.stop()
        mock_stop.assert_called()


if __name__ == '__main__':
    unittest.main()