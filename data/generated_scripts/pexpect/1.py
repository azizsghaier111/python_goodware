import os
import pexpect
import time
import unittest
from unittest.mock import patch

class ShellProcess:
    """
    A class used to initiate a dummy bash shell process for interaction
    """

    # constructor
    def __init__(self, command='bash'):
        self.command = command
        self.child = None

    def start(self):
        """Method to start the shell process"""
        try:
            self.child = pexpect.spawn(self.command)

        except pexpect.exceptions.ExceptionPexpect as e:
            print('Exception occurred while starting child process:', str(e))

    def is_alive(self):
        """Method to check whether the shell process is alive"""
        try:
            return self.child.isalive()

        except pexpect.exceptions.ExceptionPexpect as e:
            print('Exception occurred while checking the status of child process:', str(e))

    def send(self, cmd):
        """Method to send command to the shell process"""
        try:
            self.child.sendline(cmd)

        except pexpect.exceptions.ExceptionPexpect as e:
            print('Exception occurred while sending command to child process:', str(e))

    def read_non_blocking(self):
        """Method to read the shell process output"""
        try:
            return self.child.read_nonblocking(1000, 2)

        except pexpect.exceptions.EOF as e:
            print('Exception occurred while reading output from child process:', str(e))

class TestShellProcess(unittest.TestCase):
    """A class used to test the shell process operations"""

    def setUp(self):
        """Method to set up shell process before each test"""
        self.process = ShellProcess()
        self.process.start()
        time.sleep(1)  # give time for process to start

    def tearDown(self):
        """Method to tear down shell process after each test"""
        self.process.child.close()
        self.process = None

    def test_is_alive(self):
        """Method to test shell process is alive"""
        self.assertTrue(self.process.is_alive())

    @patch("time.sleep", return_value=None)
    def test_interactions(self, _):
        """Method to test interactions with the shell process"""
        test_command = "echo 'Hello, World!'"
        self.process.send(test_command)
        time.sleep(1)  # give time for command to be processed
        output = self.process.read_non_blocking()
        self.assertIn("Hello, World!", output.decode('utf-8'))  # check if expected output received

# Test execution
if __name__ == "__main__":
    unittest.main()