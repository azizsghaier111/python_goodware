import os
import pexpect
import time
import unittest
import numpy as np
from unittest.mock import patch

# Create a dummy bash shell process for interaction
class ShellProcess:

    def __init__(self, command='bash'):
        self.command = command
        self.child = None

    def start(self):
        self.child = pexpect.spawn(self.command)
        if not self.is_alive():
            raise Exception("Failed to start the process!")
    
    def is_alive(self):
        return self.child.isalive()

    def send(self, cmd):
        self.child.sendline(cmd)

    def read_non_blocking(self):
        try:
            return self.child.read_nonblocking(1000, 2)
        except pexpect.exceptions.EOF:
            return ''

    def kill(self):
        self.child.kill(0)

class TestShellProcess(unittest.TestCase):

    def setUp(self):
        self.process = ShellProcess()
        self.process.start()
        time.sleep(1)

    def tearDown(self):
        if self.process.is_alive():
            self.process.kill()
        self.process = None

    def test_is_alive(self):
        self.assertTrue(self.process.is_alive())
        self.process.kill()
        self.assertFalse(self.process.is_alive())

    @patch("time.sleep")
    def test_interactions(self, _):
        for i in np.random.randint(100, size=100):
            test_command = f"echo {i}"
            self.process.send(test_command)
            time.sleep(1)
            output = self.process.read_non_blocking().strip()
            self.assertEqual(str(i), output.decode('utf-8')[-len(str(i)):])

if __name__ == "__main__":
    unittest.main()