# Continued from the code snippet above...

class TestShellProcess(unittest.TestCase):
    """
    This is the test class that uses Python built-in unittest framework to validate the functionality of our ShellProcess class.
    """

    def setUp(self):
        self.command = "bash"
        self.shell_process = ShellProcess(self.command)

    @patch('pexpect.spawn')
    def test_start(self, mock_spawn):
        self.shell_process.start()
        mock_spawn.assert_called_once_with(self.command)

    @patch('pexpect.spawn.isalive')
    def test_is_alive(self, mock_isalive):
        self.shell_process.is_alive()
        mock_isalive.assert_called_once()

    @patch('pexpect.spawn.sendline')
    def test_send(self, mock_send_line):
        command_to_send = "ls"
        self.shell_process.send(command_to_send)
        mock_send_line.assert_called_once_with(command_to_send)

    @patch('pexpect.spawn.read_nonblocking')
    def test_read_non_blocking(self, mock_read_non_blocking):
        self.shell_process.read_non_blocking()
        mock_read_non_blocking.assert_called_once_with(1000, 2)


if __name__ == '__main__':

    # Start the test suite
    unittest.main()