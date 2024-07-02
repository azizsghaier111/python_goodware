import unittest
from unittest.mock import patch
from pexpect import spawn, ExceptionPexpect

class ShellProcess:
    """ ShellProcess class to spawn a child process and interact with it """

    def __init__(self, command='/bin/bash'):
        self.command = command
        self.child = None

    def start(self):
        try:
            self.child = spawn(self.command)
        except ExceptionPexpect:
            print("Error: Could not start child process. Please check your command.")
            raise

    def is_alive(self):
        if not self.child:
            print("Error: No child process was started")
            return False
        return self.child.isalive()
    
    def send(self, command):
        if not self.child:
            print("Error: No child process was started")
            return
        self.child.sendline(command)

    def read_nonblocking(self):
        if not self.child:
            print("Error: No child process was started")
            return
        try:
            return self.child.read_nonblocking().decode('utf-8')
        except ExceptionPexpect:
            return ''


class TestShellProcess(unittest.TestCase):
    """ Testing ShellProcess class """

    def setUp(self):
        """ Set up mocks """
        self.mock_pexpect = patch('pexpect.spawn').start()
        self.mock_child = self.mock_pexpect.return_value

    def tearDown(self):
        """ Remove mocks """
        patch.stopall()

    def test_start_process(self):
        """ Test start process method """
        shell_process = ShellProcess()
        shell_process.start()
        self.mock_pexpect.assert_called_once_with(shell_process.command)

    def test_start_process_exception(self):
        """ Test start process method for exception """
        self.mock_pexpect.side_effect = Exception('Test exception')
        shell_process = ShellProcess()
        with self.assertRaises(Exception):
            shell_process.start()

    def test_check_process_status(self):
        """ Test check process status """
        self.mock_child.isalive.return_value = True
        shell_process = ShellProcess()
        shell_process.start()
        self.assertTrue(shell_process.is_alive())

    def test_send_command_to_process(self):
        """ Test send command to process """
        shell_process = ShellProcess()
        shell_process.start()
        shell_process.send('ls')
        self.mock_child.sendline.assert_called_once_with('ls')

    def test_read_process_output(self):
        """ Test read process output """
        self.mock_child.read_nonblocking.return_value = b'Test output'
        shell_process = ShellProcess()
        shell_process.start()
        self.assertEqual(shell_process.read_nonblocking().decode('utf-8'), 'Test output')


if __name__ == '__main__':
    unittest.main()