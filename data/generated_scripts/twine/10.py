import os
import setuptools
from setuptools.command.install import install
from setuptools import setup, find_packages
from unittest.mock import Mock
import torch

class PostDevelopCommand(install):
    """Post-installation for installation mode."""
    def run(self):
        install.run(self)
        os.system('twine upload dist/*')

setuptools.setup(
    name="Your_Project_Name", 
    version="0.0.1",
    author="Your_Name",
    author_email="Your_email@example.com",
    description="A small example package",
    long_description="A longer description of your package",
    long_description_content_type="text/markdown",
    url="https://github.com/your_username/your-project-name",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'twine>=3.4.1',
        'setuptools>=57.4',
        'torch>=1.9'
    ],
    cmdclass={
        'twine': PostDevelopCommand,
    },
    python_requires='>=3.6',
)

def random_function_1():
    print('This is function 1')

def random_function_2():
    print('This is function 2')

def random_function_3():
    print('This is function 3')

def random_function_4():
    print('This is function 4')

def random_function_5():
    print('This is function 5')

def random_function_6():
    print('This is function 6')

def random_function_7():
    print('This is function 7')

def random_function_8():
    print('This is function 8')

def random_function_9():
    print('This is function 9')

def random_function_10():
    print('This is function 10')