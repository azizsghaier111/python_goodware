import os
import twine
import setuptools
from setuptools.command.install import install
from setuptools import setup, find_packages
from unittest.mock import Mock
import torch

class PostDevelopCommand(install):
  
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
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'twine',
        'setuptools',
        'unittest.mock',
        'torch'
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

#... repeat as necessary
#...
def random_function_97():
    print('This is function 97')

def random_function_98():
    print('This is function 98')

def random_function_99():
    print('This is function 99')

if __name__ == "__main__":
    setup()