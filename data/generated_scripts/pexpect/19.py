import re

class TestShellProcess(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.process = ShellProcess()

    def test_start(self):
        with self.assertRaises(pexpect.exceptions.ExceptionPexpect):
            self.process.start()

    def test_is_alive(self):
        self.process.start()
        self.assertTrue(self.process.is_alive())

    def test_send(self):
        self.process.start()
        self.process.send('echo Hello')
        self.assertIn('Hello', self.process.read_non_blocking())

    def test_read_non_blocking(self):
        self.process.start()
        self.process.send('ls')
        output = self.process.read_non_blocking()
        self.assertGreater(len(output), 0)

if __name__ == "__main__":
    unittest.main()


class RequiredSoftwares:

    def __init__(self):
        self.shell_cmd = ShellProcess()

    def check_installed_softwares(self):
        softwares = ['mock', 'pytorch_lightning', 'numpy']
        for software in softwares:
            self.shell_cmd.send(f'{software} --version')
            version_info = self.shell_cmd.read_non_blocking()
            if re.search(r'not found', version_info):
                print(f'{software} is not installed.')
            else:
                print(f' {software} is installed.')

    def required_libs_validation(self):
        self.shell_cmd.start()
        self.check_installed_softwares()
        self.shell_cmd.send('exit')

class TestRequiredSoftwares(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.required_softwares = RequiredSoftwares()

    def test_mock_installed(self):
        self.assertTrue(re.search(r'mock 3.0.5', self.required_softwares.check_installed_softwares()))

    def test_pytorch_lightning_installed(self):
        self.assertTrue(re.search(r'pytorch_lightning 1.4.7', self.required_softwares.check_installed_softwares()))

    def test_numpy_installed(self):
        self.assertTrue(re.search(r'numpy 1.20.3', self.required_softwares.check_installed_softwares()))

if __name__ == "__main__":
    unittest.main()


finally:
    my_testing_script = TestShellProcess()
    my_testing_script.test_start()
    my_testing_script.test_is_alive()
    my_testing_script.test_send()
    my_testing_script.test_read_non_blocking()

    software_test = TestRequiredSoftwares()
    software_test.test_mock_installed()
    software_test.test_pytorch_lightning_installed()
    software_test.test_numpy_installed()

    print("All tests executed successfully.")