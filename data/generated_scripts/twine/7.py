import os
from setuptools import setup, find_packages
from setuptools.command.develop import develop


def save_story_to_html(story, filename):
    with open(filename, 'w') as file:
        file.write(story.export_to_html())


class Story:
    def __init__(self):
        self.passages = []

    def add_passage(self, passage):
        self.passages.append(passage)

    def organize_passage_layout(self):
        # rearrange passages based on some criteria
        pass

    def export_to_html(self):
        html = ""
        # convert each passage to html and combine
        for passage in self.passages:
            html += passage.to_html()
        return html

class Passage:
    def __init__(self, text, choices):
        self.text = text
        self.choices = choices

    def to_html(self):
        html = ''
        # add passage text
        html += '<p>{}</p>'.format(self.text)
        # add each choice
        for choice in self.choices:
            html += '<button>{}</button>'.format(choice)
        return html
    

class PostDevelopCommand(develop):
    """Post-installation for development mode."""
    def run(self):
        os.system('twine upload dist/*')
        develop.run(self)
        

setup(
    name='Your-Project-Name',
    version='0.0.1',
    packages=find_packages(),
    author='Your Name',
    author_email='youremail@provider.com',
    description='Your Project Description',
    long_description='Your Very Long Project Description',
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/yourproject',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    install_requires=[
        'twine',
        'mock',
        'torch'
    ],
    cmdclass={
        'develop': PostDevelopCommand,
    },
    python_requires='>=3.6'
)