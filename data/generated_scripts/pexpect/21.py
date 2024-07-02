import os
import pexpect
import time
import unittest
import numpy as np
from numpy.testing import assert_almost_equal
from unittest.mock import patch

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

    def send_and_read(self, cmd):
        self.send(cmd)
        time.sleep(1)
        return self.read_non_blocking()

    def send_and_expect(self, cmd, exp):
        self.send(cmd)
        try:
            self.child.expect(exp, timeout=2)
        except pexpect.exceptions.TIMEOUT:
            return False
        return True

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

    def test_alive(self):
        self.assertTrue(self.process.is_alive())
        self.process.kill()
        self.assertFalse(self.process.is_alive())

    @patch("time.sleep")
    def test_non_blocking_read(self, _):
        for i in np.random.randint(100, size=100):
            test_command = f"echo {i}"
            self.process.send(test_command)
            time.sleep(1)
            output = self.process.read_non_blocking().strip()
            self.assertEqual(str(i), output.decode('utf-8')[-len(str(i)):])

    def test_send_and_read(self):
        result = self.process.send_and_read("echo 1")
        self.assertEqual("1", result.decode('utf-8')[-1:])
        
    def test_send_and_expect(self):
        self.assertTrue(self.process.send_and_expect("echo 2", "2"))
        self.assertFalse(self.process.send_and_expect("echo 2", "3"))

if __name__ == "__main__":
    unittest.main()