import os
from jinja2 import Environment, PackageLoader, select_autoescape
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
    extensions=['jinja2.ext.do', 'jinja2.ext.loopcontrols', 'jinja2.ext.with_', 'jinja2.ext.autoescape', 'jinja2.ext.sandbox'],
)

# Set up sandboxed environment
env.sandboxed = True
sandbox_env = SandboxedEnvironment(autoescape=True)

@env.filter
def upcase(value):
    return value.upper()

def process_template(template_name, context=None):
    if context is None:
        context = {}

    template = env.get_template(template_name)

    template.globals.update({
        'os' : os,
        'global_var': 'Global Variable'
    })
    
    return template.render(context)

def create_mkdocs_config():
    config = Config(schema=mkdocs.config.base.DEFAULT_SCHEMA)
    config['site_name'] = 'My Documentation'
    config['nav'] = [
        {'Home': 'index.md'},
        {'About': 'about.md'},
        {'Chapter1': [{'Introduction': 'chapter1/intro.md'}, {'Overview': 'chapter1/overview.md'}]},
        {'Chapter2': [{'Topics': 'chapter2/topics.md'}, {'Summary': 'chapter2/summary.md'}]}
    ]
    return config

def output_html(html_string):
    parser = etree.HTMLParser()
    root_element = etree.fromstring(html_string, parser)
    doctype = '<!DOCTYPE html>\n'
    return doctype + etree.tostring(root_element, pretty_print=True, method="html", encoding='unicode')

def mock_example():
    m = mock.Mock()
    m.method.return_value = 'mocked method result'
    return m.method()

def get_template_vars(template_path):
    with open(template_path, 'r') as template_file:
        parsed_content = env.parse(template_file.read())
    return meta.find_undeclared_variables(parsed_content)

def get_template_dirs():
    template_dirs = [x[0] for x in os.walk('templates')]
    return template_dirs

if __name__ == "__main__":
    context = {
        'title': 'High Performance',
        'features': [
            'URL reversing',
            'Escaping',
        ],
        'mkdocs_config': create_mkdocs_config(),
        'mock_result': mock_example(),
    }
    html_string = process_template('index.html', context)
    
    # get undeclared variables from final template by sandbox
    print(get_template_vars(os.path.join('templates', 'index.html')))
    
    print(output_html(html_string))