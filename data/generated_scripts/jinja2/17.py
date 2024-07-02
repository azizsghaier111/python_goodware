import os
from jinja2 import Environment, select_autoescape
from jinja2.sandbox import SandboxedEnvironment
from jinja2.loaders import FileSystemLoader
import mkdocs
from mkdocs.config.base import Config
import lxml.etree as etree
from unittest import mock

# Initialize Jinja2 with file system loader.
env = Environment(
    loader=FileSystemLoader(os.path.join(os.getcwd(), 'templates')),
    autoescape=select_autoescape(['html', 'xml']),
    extensions=[
        'jinja2.ext.do',
        'jinja2.ext.loopcontrols',
        'jinja2.ext.with_',
        'jinja2.ext.autoescape',
        'jinja2.ext.sandbox'
    ]
)

# Set up sandboxed environment
env.sandboxed = True
sandbox_env = SandboxedEnvironment(autoescape=True)

@env.filter
def upper(value):
    return value.upper()

def create_context():
    context = {
        'title': 'My Title',
        'body': 'This is a body',
        'upper': upper,
        'os': os,
    }
    return context

def create_config():
    config = Config(schema=mkdocs.config.base.DEFAULT_SCHEMA)
    config['site_name'] = 'My Site'
    config['nav'] = []
    return config

def main():
    template = env.get_template('index.html.j2')
    result = template.render(create_context())
    print(result)

if __name__ == "__main__":
    main()