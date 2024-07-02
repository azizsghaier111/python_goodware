# import required libraries
import pexpect
import time
import os
import unittest
from unittest.mock import patch

class ShellProcess:
    """
    A class used to initiate a bash shell process for interaction.
    """

    def __init__(self, command='bash'):
        """
        Constructor to initialize the command line and child.
        """
        self.command = command
        self.child = None

    def start(self):
        """
        Method to start the shell process
        """
        # Inside a try block to catch exceptions
        try:
            # Spawn a child process
            self.child = pexpect.spawn(self.command)

        except pexpect.exceptions.ExceptionPexpect as e:
            print('Exception occurred while starting child process:', str(e))

    def is_alive(self):
        """
        Method to check whether the shell process is alive
        """
        # Inside a try block to catch exceptions
        try:
            # Check if the process is alive
            return self.child.isalive()

        except pexpect.exceptions.ExceptionPexpect as e:
            print('Exception occurred while checking the status of child process:', str(e))

    def send(self, cmd):
        """
        Method to send command to the shell process
        """
        # Inside a try block to catch exceptions
        try:
            # Send a command to the child process
            self.child.sendline(cmd)

        except pexpect.exceptions.ExceptionPexpect as e:
            print('Exception occurred while sending command to child process:', str(e))

    def read_non_blocking(self):
        """
        Method to read the shell process output in a non-blocking way
        """
        # Inside a try block to catch exceptions
        try:
            # Read output from the child process  
            return self.child.read_non_blocking(1000, 2)

        except pexpect.exceptions.EOF as e:
            print('Exception occurred while reading output from child process:', str(e))

class TestShellProcess(unittest.TestCase):
    """
    A class used to test the shell process operations
    """
    def setUp(self):
        """
        Method to set up shell process before each test
        """
        # Initiate a process variable
        self.process = ShellProcess()
        # Start the process
        self.process.start()
        # Sleep for 1 second to allow the process start
        time.sleep(1)

    def tearDown(self):
        """
        Method to tear down shell process after each test
        """
        # Close the child
        self.process.child.close()
        # Clear the process
        self.process = None

    def test_is_alive(self):
        """
        Method to test shell process is alive
        """
        # Here we assert if True was returned when checking the process
        self.assertTrue(self.process.is_alive())

    @patch("time.sleep", return_value=None)
    def test_interactions(self, _):
        """
        Method to test interactions with the shell process
        """
        # Test command
        test_command = "echo 'Hello, World!'"
        # Send the test command
        self.process.send(test_command)
        # Sleep for 1 second to allow command to be processed
        time.sleep(1)
        # Read the output from the process
        output = self.process.read_non_blocking()
        # Assert if Hello, World! is in the output
        self.assertIn("Hello, World!", output.decode('utf-8'))