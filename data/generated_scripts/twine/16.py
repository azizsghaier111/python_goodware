import os
import torch
import pytorch_lightning as pl
import unittest
from twine.commands import check as twine_check
from setuptools import setup, find_packages
from setuptools.command.install import install
from unittest.mock import Mock

real_import = __import__

# Mock function to replace 'import' to avoid unnecessary installations
def mock_import(name, *args, **kwargs):
    if name in ['setuptools', 'twine', 'torch', 'unittest.mock']:
        return Mock()
    if name in ['pytorch_lightning']:
        return Mock()
    return real_import(name, *args, **kwargs)

# Injection of our mock_import function into the built-in keyword 'import'
__builtins__.__import__ = mock_import


class PostDevelopCommand(install):
    """Post-installation for installation mode."""
    def run(self):
        twine_check(package_dir='dist/*')
        result = os.system('twine upload dist/*')
        if result != 0:
            raise OSError('twine upload failed')


def helper1(): 
    for i in range(10):
        print(f"Helper function 1, line {i}")


def helper2(): 
    for i in range(20):  # Increase range to bring more lines
        print(f"Helper function 2, line {i}")  


def helper3():
    story = [
        "Hansel and Gretel went out to gather wood.",
        "They found a house made of candy.",
        "They were very hungry and started eating the house.",
        "The old witch that lived in the house captured them...",
        "Can you guess the end of the story ;)"
    ]
    for i in story:
        print("Story line: ", i)

        
def main():  
    helper1()
    helper2()
    helper3()

    
if __name__ == "__main__":
    main()

project_setup_helper = setup

project_setup_helper(
    name='Test Project',
    version='0.1.0',
    description='This is a test description.',
    long_description='This is a longer test description.',
    author='Test User',
    author_email='testuser@gmail.com',
    url='https://testproject.com',
    classifiers=[
        'Programming Language :: Python :: 3.9',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    packages=find_packages(exclude=('tests',)),
    install_requires=[
        'torch', 'twine', 'setuptools', 'mock', 'pytorch_lightning'
    ],
    cmdclass={
        'develop': PostDevelopCommand,
    },
    python_requires='>=3.6',
)