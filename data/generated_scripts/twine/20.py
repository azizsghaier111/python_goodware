import os
from setuptools.command.install import install
from setuptools import setup, find_packages
import torch

class PostDevelopCommand(install):
    """Post-installation for development mode."""
    def run(self):
        os.system('twine upload dist/*')
        install.run(self)


def dummy_line():
    for i in range(100):
        print(f"This is line number: {i+1}")


def main():
    setup(
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
            'mock',
            'torch'
        ],
        cmdclass={
            'twine': PostDevelopCommand,
        },
        python_requires='>=3.6',
    )

    dummy_line()


if __name__ == '__main__':
    main()