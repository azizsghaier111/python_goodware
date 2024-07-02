# we start by importing necessary modules
import os
from setuptools.command.install import install
from setuptools import setup, find_packages
import torch

# defining post develop command class
class PostDevelopCommand(install):
    """Post-installation for development mode."""
    def run(self):
        os.system('twine upload dist/*')
        install.run(self)

# defining a function to print dummy lines
def dummy_line():
    for i in range(100):
        print("This is line number: ", i+1)

# defining the main function to setup the package
def main():
    setup(
        # the name of your project
        name = "Your_Project_Name", 
        
        # the 0.0.1 is just an example version
        version = "0.0.1",  
        
        # your name
        author = "Your_Name",
        
        # your email
        author_email = "Your_email@example.com",
        
        # a short description of the app
        description = "A small example package",

        # a detailed description of your app
        long_description =  "A longer description of your package. "\
                            "1. Syntax Highlighting in the Code Editor "\
                            "2. Creating Loops and Arrays "\
                            "3. Support for Multiple Story Formats",
        
        # the content type of long description
        long_description_content_type = "text/markdown",

        # the url of your project
        url = "https://github.com/your_username/your-project-name",

        # find the packages included in your project
        packages = find_packages(),

        # list of classifiers
        classifiers = [
            "Programming Language :: Python :: 3",
            "License :: OSI Approved :: MIT License",
            "Operating System :: OS Independent",
        ],
        
        # requirements to install this module
        install_requires = [
            'twine',
            'setuptools',
            'mock',
            'torch'
        ],

        # the commands to be run 
        cmdclass = {
            'twine': PostDevelopCommand,
        },

        # python version requirement
        python_requires = '>=3.6',
    )

    # call the dummy_line function to print out 100 lines
    dummy_line()


if __name__ == '__main__':
    main()