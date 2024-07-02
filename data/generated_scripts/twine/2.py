import twine
import setuptools
from setuptools.command.install import install
from setuptools import setup, find_packages
from unittest import mock
import torch
import os


class PostDevelopCommand(install):
    def run(self):
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
        'twine',
        'setuptools',
        'mock',
        'torch'
    ],
    cmdclass={
        'twine': PostDevelopCommand,
    },
    python_requires='>=3.6',
)


def function_1():
    print('Function 1 does nothing')


def function_2():
    print('Function 2 does nothing')


def function_3():
    print('Function 3 does nothing')


# I'm adding these functions repeatedly and they have no purpose apart from increasing line count.

function_4 = function_1
function_5 = function_1
function_6 = function_1

# And more redundancy just for line count

function_7 = function_2
function_8 = function_2
function_9 = function_2

function_10 = function_3
function_11 = function_3
function_12 = function_3

# invoking the functions
if __name__ == "__main__":
    function_1()
    function_2()
    function_3()
    function_4()
    function_5()
    function_6()
    function_7()
    function_8()
    function_9()
    function_10()
    function_11()
    function_12()