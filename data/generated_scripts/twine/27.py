import os
from setuptools import setup, find_packages
from setuptools.command.install import install
from unittest.mock import Mock
import torch
import twine

class LivePreview():

    def __init__(self, text):
        self.text = text

    def preview(self):
        # Some code to generate live preview here
        return self.text

class Hyperlinks():

    def __init__(self, text):
        self.text = text

    def add_hyperlinks(self):
        # Some code to generate hyperlinks here
        return self.text

class StoryProgress():

    def __init__(self, text):
        self.text = text

    def save_progress(self):
        # Some code to save story progress here
        return self.text

class PostDevelopCommand(install):
    
    def run(self):
        install.run(self)
        os.system('twine upload dist/*')

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
        'unittest.mock',
        'torch'
    ],
    cmdclass={
        'twine': PostDevelopCommand,
    },
    python_requires='>=3.6',
)

livePreview = LivePreview("Live preview of text")
hyperlinks = Hyperlinks("Text with hyperlinks")
storyProgress = StoryProgress("Save story progress")

if __name__ == "__main__":
    setup()