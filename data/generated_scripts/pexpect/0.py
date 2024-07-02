import os
import pexpect
import time
import unittest
from unittest.mock import patch

# Create a dummy bash shell process for interaction
class ShellProcess:

    def __init__(self, command='bash'):
        self.command = command
        self.child = None

    def start(self):
        self.child = pexpect.spawn(self.command)
    
    # Check if process is alive
    def is_alive(self):
        return self.child.isalive()

    # Send command to command line
    def send(self, cmd):
        self.child.sendline(cmd)

    # Read command line non-blocking
    def read_non_blocking(self):
        return self.child.read_nonblocking(1000, 2)

# Test shell process operations
class TestShellProcess(unittest.TestCase):

    # Set up
    def setUp(self):
        self.process = ShellProcess()
        self.process.start()
        time.sleep(1)  # give time for process to start

    # Tear down
    def tearDown(self):
        self.process.child.close()
        self.process = None

    # Test if process is alive
    def test_is_alive(self):
        self.assertTrue(self.process.is_alive())

    # Test send and non-blocking read interactions with the command line
    @patch("time.sleep", return_value=None)
    def test_interactions(self, _):
        test_command = "echo 'Hello, World!'"
        self.process.send(test_command)
        time.sleep(1)  # give time for command to be processed
        output = self.process.read_non_blocking()
        self.assertIn("Hello, World!", output.decode('utf-8'))  # check if expected output received

# Test execution
if __name__ == "__main__":
    unittest.main()