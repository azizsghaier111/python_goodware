import pexpect
import time
import unittest
from unittest.mock import patch

class ShellProcess:
    """
    A detailed version of the class ShellProcess is provided below.
    Each method has a basic description, and exceptions are handled in more detail.
    """
    def __init__(self, command='bash'):
        """
        Constructor for the ShellProcess class.
        Initializes the command to run in shell, and the child process to None.
        """
        self.command = command
        self.child = None

    def start(self):
        """
        This method is used to start the shell process.
        If it fails to start the shell process, it captures and prints the exception.
        """
        try:
            self.child = pexpect.spawn(self.command)
            print('Successfully started the shell process.')

        except pexpect.exceptions.ExceptionPexpect as e:
            print('Exception occurred while starting child process:', str(e))

    def is_alive(self):
        """
        This method is used to check if the shell process is alive.
        If it fails to check the status of the child process, it captures and prints the exception.
        """
        try:
            status = self.child.isalive()
            print(f'process running status is: {status}')
            return status

        except pexpect.exceptions.ExceptionPexpect as e:
            print('Exception occurred while checking the status of child process:', str(e))

    def send(self, cmd):
        """
        This method is used to send command to the shell process.
        If it fails to send command to the shell process, it captures and prints the exception.
        """
        try:
            print(f'sending command : {cmd}')
            self.child.sendline(cmd)
            print('command sent successfully')

        except pexpect.exceptions.ExceptionPexpect as e:
            print('Exception occurred while sending command to child process:', str(e))

    def read_non_blocking(self):
        """
        This method is used to read the process output in a non blocking manner.
        If it fails to read the output of the child process, it captures and prints exception.
        """
        try:
            output = self.child.read_nonblocking(1000, 2)
            print(f'read from stdout: \n{output.decode("utf-8")}')
            return output

        except pexpect.exceptions.EOF as e:
            print('Exception occurred while reading output from child process:', str(e))

# ... 

# Run tests ...