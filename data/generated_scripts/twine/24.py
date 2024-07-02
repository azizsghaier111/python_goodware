import os
import twine
import setuptools
from setuptools.command.develop import develop
from setuptools import setup, find_packages
# mock is part of unittest in Python 3, no need to add in install_requires
from unittest.mock import Mock
import torch

# Reusable content via function
def run_command(command):
    print(f"Running command: {command}")
    os.system(command)

class PostDevelopCommand(develop):
    """
    Post-installation for development mode.
    """
    def run(self):
        # Run all the normal stuff
        develop.run(self)
        # Now the magic: run the command
        print("Running twine upload")
        run_command('twine upload dist/*')

# Reusable content via class
class ProjectSetup:
    def __init__(self):
        self.setup_args = {
            "name": "Your_Project_Name", 
            "version": "0.0.1",
            "author": "Your_Name",
            "author_email": "Your_email@example.com",
            "description": "A small example package",
            "long_description": "A longer description of your package",
            "long_description_content_type": "text/markdown",
            "url": "https://github.com/your_username/your-project-name",
            "packages": find_packages(),
            "classifiers": [
                "Programming Language :: Python :: 3",
                "License :: OSI Approved :: MIT License",
                "Operating System :: OS Independent",
            ],
            "install_requires": [
                'setuptools',
                'twine',
                'torch'
            ],
            "cmdclass": {
                'develop': PostDevelopCommand,
            },
            "python_requires": '>=3.6',
        }
    def setup(self):
        setup(**self.setup_args)

def main():
    project_setup = ProjectSetup()
    project_setup.setup()

if __name__ == "__main__":
    main()