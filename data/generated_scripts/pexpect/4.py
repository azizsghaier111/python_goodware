import sys
import traceback

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
        self.process.send('echo Hello, World!')

        time.sleep(1)  # Allow buffer to fill

        # Now read and check message
        received = self.process.read_non_blocking().decode('utf-8')
        self.assertIn('Hello, World!', received)

    def test_alive(self):
        """Test whether the process remains alive"""
        self.process.start()
        self.assertTrue(self.process.is_alive(), "Child process is not running.")

    def test_not_alive(self):
        """Test whether a non-existent process is reported as not alive"""
        self.process.child = None
        self.assertFalse(self.process.is_alive(), "Non-existent child process is reported as running.")

    def test_read_non_blocking(self):
        """Test if it can read non-buffer blocking"""
        self.process.start()
        self.process.send('echo Hello, World!')
        time.sleep(1)

        # Now read and check message
        received = self.process.read_non_blocking().decode('utf-8')
        self.assertIn('Hello, World!', received)


if __name__ == '__main__':
    try:
        unittest.main()
    except:
        print("Exception in user code:")
        print("-"*60)
        traceback.print_exc(file=sys.stdout)
        print("-"*60)
        sys.exit(-1)