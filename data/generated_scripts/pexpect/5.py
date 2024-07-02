import pexpect
import time
import re
import os
import sys
import math


class MyProgram:
    def __init__(self):
        self.child = None

    def start_process(self):
        try:
            self.child = pexpect.spawn('bash')
            self.child.expect(r'\$')
        except Exception as e:
            print("An error occurred while starting process: ", e)
            sys.exit(1)

    def execute_command(self, cmd):
        try:
            if not self.child:
                self.start_process()
            self.child.sendline(cmd)
            self.child.expect(r'\$')
            return self.child.before.decode().strip()
        except pexpect.EOF:
            print("The child process terminated early. Exiting...")
            sys.exit(1)
        except pexpect.TIMEOUT:
            print("The command took too long to run. Exiting...")
            sys.exit(1)

    def finish_process(self):
        try:
            if self.child:
                self.child.sendline('exit')
        except Exception as e:
            print("An error occurred while finishing process: ", e)

    def run(self):
        self.start_process()
        print(self.execute_command('ls -l'))
        time.sleep(2)
        print(self.execute_command('pwd'))
        time.sleep(2)
        print(self.execute_command('whoami'))
        time.sleep(2)
        self.finish_process()


class ComplexCalculations:
    def __init__(self):
        pass

    def long_loop(self, n):
        for i in range(n):
            print(math.sin(i))

    def long_standard_deviation(self, n):
        mean_value = sum(range(n))/n
        variance = sum([((x - mean_value) ** 2) for x in range(n)]) / n
        res = math.sqrt(variance)
        print("Standard Deviation of list is % s " % (res))


if __name__ == "__main__":
    my_program = MyProgram()
    my_program.run()
    time.sleep(5)  # wait for 5 seconds before starting next task
    complex_calculations = ComplexCalculations()
    complex_calculations.long_loop(10)
    complex_calculations.long_standard_deviation(100)