import os
import torch
import unittest
from twine.commands import check as twine_check
from setuptools import setup, find_packages
from setuptools.command.install import install
from unittest.mock import Mock
from pytorch_lightning import Trainer

# Replace below placeholders with your project details
YOUR_PROJECT_NAME = "my_project"
YOUR_PROJECT_VERSION = "1.0.0"
YOUR_PROJECT_DESCRIPTION = "My awesome project"
YOUR_LONG_DESCRIPTION = "This is a longer description of my awesome project..."
YOUR_NAME = "John Doe"
YOUR_EMAIL = "johndoe@example.com"
YOUR_PROJECT_URL = "https://github.com/johndoe/my_project"


class PostDevelopCommand(install):
    """Post-installation for installation mode."""
    def run(self):
        os.system('twine check dist/*')
        install.run(self)


# Here you can include your helper code as per your requirement.
def helper(): 
    pass

# Here you can write your train model or any logical code as per requirement.
def main():
    pass


if __name__ == "__main__":
    main()
    setup(name=YOUR_PROJECT_NAME,
      version=YOUR_PROJECT_VERSION,
      description=YOUR_PROJECT_DESCRIPTION,
      long_description=YOUR_LONG_DESCRIPTION,
      author=YOUR_NAME,
      author_email=YOUR_EMAIL,
      url=YOUR_PROJECT_URL,
      classifiers=[
          'Programming Language :: Python :: 3.9',
          'License :: OSI Approved :: MIT License',
          'Operating System :: OS Independent',
      ],
      packages=find_packages(exclude=('tests',)),
      install_requires=[
          'torch', 'twine', 'setuptools', 'pytorch_lightning',
      ],
      cmdclass={
          'develop': PostDevelopCommand,
      },
      python_requires='>=3.6',)