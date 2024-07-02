# required libraries
import pexpect
import time
import os
import unittest
from unittest.mock import patch

class ShellProcess:
    """A class used to initiate a bash shell process for interaction."""

    def __init__(self, command='bash'):
        """
        Constructor to initialize the command line and child.
        """
        self.command = command
        self.child = None

    def __str__(self):
        return f'ShellProcess object with command: {self.command}'

    def __repr__(self):
        return f'ShellProcess({self.command})'

    def start(self):
        """Method to start the shell process."""
        try:
            # Initialize the pexpect child process
            self.child = pexpect.spawn(self.command)

        except pexpect.exceptions.ExceptionPexpect as e:
            print('Exception occurred while starting child process:', str(e))

    def is_alive(self):
        """Method to check whether the shell process is alive"""
        try:
            # Return the status of the child process
            return self.child.isalive()

        except pexpect.exceptions.ExceptionPexpect as e:
            print('Exception occurred while checking the status of child process:', str(e))

    def send(self, cmd):
        """Method to send command to the shell process"""
        try:
            # Send a command to the child process
            self.child.sendline(cmd)
        # Handle pexpect exceptions  
        except pexpect.exceptions.ExceptionPexpect as e:
            print('Exception occurred while sending command to child process:', str(e))

    def read_non_blocking(self):
        """Method to read the shell process output in a non-blocking way"""
        try:
            # Return output from the child process
            return self.child.read_non_blocking(1000, 2)
       
        except pexpect.exceptions.EOF as e:
            print('Exception occurred while reading output from child process:', str(e))


# Set up the unit tests
class TestShellProcess(unittest.TestCase):
    """A class used to test the shell process operations."""
    
    def setUp(self):
        """Method to set up shell process before each test."""
        # Start a bash shell for test interaction
        self.process = ShellProcess()
        self.process.start()
        # Allow said shell to initialize fully before proceeding
        time.sleep(1)

    def tearDown(self):
        """Method to tear down shell process after each test."""
        # Terminate the child process
        self.process.child.close()
        # Clear the process variable
        self.process = None

    def test_is_alive(self):
        """Method to test shell process is alive."""
        # Verifying if process is alive
        self.assertTrue(self.process.is_alive())

    @patch("time.sleep", return_value=None)
    def test_interactions(self, _):
        """Method to test interactions with the shell process"""
        # Sending command to the shell
        test_command = "echo 'Hello, World!'"
        self.process.send(test_command)
        # Allowing time to process command
        time.sleep(1)
        # Get output from command
        output = self.process.read_non_blocking()
        # Test if expected response is provided
        self.assertIn("Hello, World!", output.decode('utf-8'))


# Allow individual running of this script
if __name__ == "__main__":
    unittest.main()