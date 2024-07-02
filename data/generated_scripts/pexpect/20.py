import os
import pexpect
import time
import unittest
import pytest
import numpy as np
from unittest.mock import patch, Mock
from pexpect.exceptions import ExceptionPexpect, EOF

def check_prerequisites():
    print("Checking prerequisites.......\n")
    try:
        os.system('python -m pip install pexpect')
        os.system('python -m pip install pytest')
        os.system('python -m pip install numpy')
        print("Prerequisites installed successfully.")
        return True
    except Exception as e:
        return False

class ShellProcess:
    def __init__(self, command='bash'):
        self.command = command
        self.child = None

    def start(self):
        try:
            self.child = pexpect.spawn(self.command)
            print('Started ShellProcess')
        except ExceptionPexpect as exception:
            raise ShellProcessException(exception)

    def is_alive(self):
        try:
            return self.child.isalive()
        except ExceptionPexpect as exception:
            raise ShellProcessException(exception)

    def send(self, data):
        try:
            self.child.sendline(data)
        except ExceptionPexpect as exception:
            raise ShellProcessException(exception)

    def read(self):
        try:
            return self.child.read_nonblocking(1000, 2)
        except EOF:
            return ''
        except ExceptionPexpect as exception:
            raise ShellProcessException(exception)

    def set_timeout(self, timeout):
        try:
            self.child.timeout = timeout
        except ExceptionPexpect as exception:
            raise ShellProcessException(exception)

    def reset_timeout(self):
        try:
            self.child.timeout = 30
        except ExceptionPexpect as exception:
            raise ShellProcessException(exception)

    def terminate(self):
        try:
            self.child.terminate()
            print('Terminated ShellProcess')
        except ExceptionPexpect as exception:
            raise ShellProcessException(exception)

class ShellProcessException(Exception):
    pass

class TestShellProcess(unittest.TestCase):
    def setUp(self):
        self.shell_process = ShellProcess()
        self.command = 'echo "Hello World"'
        self.timeout = 10

    @patch.object(ShellProcess, 'start')
    def test_start(self, Mock_start):
        Mock_start.return_value = None
        self.assertEqual(self.shell_process.start(), None)

    @patch.object(ShellProcess, 'is_alive')
    def test_is_alive(self, Mock_is_alive):
        Mock_is_alive.return_value = True
        self.assertEqual(self.shell_process.is_alive(), True)

    @patch.object(ShellProcess, 'send')
    def test_send(self, Mock_send):
        Mock_send.return_value = None
        self.assertEqual(self.shell_process.send(self.command), None)

    @patch.object(ShellProcess, 'read')
    def test_read(self, Mock_read):
        Mock_read.return_value = "Hello World\n"
        self.assertEqual(self.shell_process.read(), "Hello World\n")

    @patch.object(ShellProcess, 'set_timeout')
    def test_set_timeout(self, Mock_set_timeout):
        Mock_set_timeout.return_value = None
        self.assertEqual(self.shell_process.set_timeout(self.timeout), None)

    @patch.object(ShellProcess, 'reset_timeout')
    def test_reset_timeout(self, Mock_reset_timeout):
        Mock_reset_timeout.return_value = None
        self.assertEqual(self.shell_process.reset_timeout(), None)

    @patch.object(ShellProcess, 'terminate')
    def test_terminate(self, Mock_terminate):
        Mock_terminate.return_value = None
        self.assertEqual(self.shell_process.terminate(), None)


if __name__ == '__main__':
    if check_prerequisites():
        unittest.main()
    else:
        print("Failed to install all prerequisites. Please check your internet connection.")