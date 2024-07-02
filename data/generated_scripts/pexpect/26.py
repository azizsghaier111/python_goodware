import pexpect
import unittest
import sys
import traceback
import time
from pexpect.exceptions import EOF
from unittest import mock
import numpy as np

class ShellProcess:
    """ShellProcess class to handle shell commands"""
    def __init__(self):
        self.child = None

    def start(self):
        """Method to start the shell process"""
        self.child = pexpect.spawn("/bin/bash")

    def send(self, command):
        """Method to send a command to the shell"""
        self.child.sendline(command)
    
    def is_alive(self):
        """Check if the process is alive"""
        return self.child.isalive()

    def read_non_blocking(self):
        """Method to read from the shell"""
        return self.child.read_nonblocking(3000, timeout=1)

class ShellProcessTest(unittest.TestCase):
    """Test class for ShellProcess"""

    def setUp(self):
        """Setup test environment by creating a ShellProcess object"""
        self.process = ShellProcess()

    def test_start(self):
        """Test the start method of the ShellProcess"""
        self.process.start()
        self.assertIsNotNone(self.process.child, "Child process is not initialized.")

    def test_send(self):
        """Test sending commands to the Bash"""
        self.process.start()
        try:
            self.process.send('echo Hello, World!')
        except:
            print("An exception occurred while sending.")

        time.sleep(1)  # Allow buffer to fill
        try:
            received = self.process.read_non_blocking().decode('utf-8')
            self.assertIn('Hello, World!', received, "The expected command output was not received.")
        except EOF as e:
            print('End of file reached before the response could be read.')

    def test_alive(self):
        """Test whether the process remains alive"""
        self.process.start()
        is_alive = self.process.is_alive()
        self.assertTrue(is_alive, "Child process is not running.")
    
    def test_not_alive(self):
        """Test whether a non-existent process is reported as not alive"""
        self.process.child = None
        is_alive = self.process.is_alive()
        self.assertFalse(is_alive, "Non-existent child process is reported as running.")

    def test_read_non_blocking(self):
        """Test if it can read non-buffer blocking"""
        self.process.start()
        try:
            self.process.send('echo Hello, World!')
            time.sleep(1)  # Allow buffer to fill
            received = self.process.read_non_blocking().decode('utf-8')
            self.assertIn('Hello, World!', received, "The expected command output was not received.")
        except EOF as e:
            print('End of file reached before the response could be read.')
    
if __name__ == '__main__':
    try:
        unittest.main()
    except:
        print("Exception in user code:")
        print("-"*60)
        traceback.print_exc(file=sys.stdout)
        print("-"*60)
        sys.exit(-1)