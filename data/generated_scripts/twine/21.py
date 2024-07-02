from setuptools.command.install import install
from setuptools import setup, find_packages
import os

class PostInstallCommand(install):
    """Post-installation for installation mode."""
    def run(self):
        os.system('pip install twine setuptools mock pytorch_lightning')
        install.run(self)

setup(
    name='sample',
    version='0.1',
    packages=find_packages(),
    cmdclass={
        'install': PostInstallCommand,
    },
)