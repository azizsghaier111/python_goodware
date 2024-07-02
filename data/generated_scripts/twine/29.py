import os
import twine
import torch
import setuptools
from setuptools import setup, find_packages
from setuptools.command.develop import develop
from multiprocessing import Pool

class Passage:
    def __init__(self, text, choices):
        self.text = text
        self.choices = choices

    @staticmethod
    def read_from_file(filename):
        with open(filename, 'r') as file:
            text = file.read().strip()
        choices = ['Yes', 'No']  # this could be replaced with a real implementation
        return Passage(text, choices)

    def to_html(self):
        html = '<p>{}</p>'.format(self.text)
        for choice in self.choices:
            html += '<button>{}</button>'.format(choice)
        return html

class Story:
    def __init__(self, title, css='style.css'):
        self.title = title
        self.passages = []
        self.css = css

    def add_passage(self, passage):
        self.passages.append(passage)

    def export_to_html(self):
        html = '<html><head><link rel="stylesheet" type="text/css" href="{}"><title>{}</title></head><body>'.format(self.css, self.title)
        for passage in self.passages:
            html += passage.to_html()
        html += '</body></html>'
        return html

    def save_to_file(self, filename):
        with open(filename, 'w') as file:
            file.write(self.export_to_html())
    
    def load_from_files(self, filenames):
        with Pool(5) as p:
            self.passages = p.map(Passage.read_from_file, filenames)

class DebugMixin:
    def test_and_play(self):
        for i, passage in enumerate(self.passages):
            print('Test play passage {}:'.format(i))
            print(passage.text)

class InteractiveStoryTeller(Story, DebugMixin):
    def __init__(self, title):
        super(InteractiveStoryTeller, self).__init__(title)
    
    def play_story(self):
        for passage in self.passages:
            print(passage.text)
    
    def adjust_appearance(self, css):
        self.css = css

class PostDevelopCommand(develop):
    def run(self):
        print('Uploading distributions...')
        os.system('twine upload dist/*')
        print('Running machine learning training...')
        # Here we could make use of pytorch_lightning
        torch.zeros((3,3))

# Use setuptools to install the package
setup(
    name='InteractiveStoryteller',
    version='0.0.1',
    packages=find_packages(),
    install_requires=[
        'setuptools',
        'twine',
        'torch'
    ],
    cmdclass={
        'develop': PostDevelopCommand,
    },
)