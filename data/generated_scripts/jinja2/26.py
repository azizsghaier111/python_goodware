import os
from jinja2 import Environment, PackageLoader, select_autoescape
from jinja2.sandbox import SandboxedEnvironment
import jinja2.meta as meta
import mkdocs.config.base
import lxml.etree as etree
from unittest import mock

# Initialize Jinja2 with file system loader.
env = Environment(
    loader=FileSystemLoader(os.path.join(os.getcwd(), 'templates')),
    autoescape=select_autoescape(['html', 'xml']),
    extensions=['jinja2.ext.do', 'jinja2.ext.loopcontrols', 'jinja2.ext.with_', 'jinja2.ext.autoescape', 'jinja2.ext.sandbox'],
)

@env.filter
def upcase(value):
    return value.upper()

def process_template(template_name, context=None):
    if context is None:
        context = {}

    template = env.get_template(template_name)
    template.globals.update({'os' : os, 'global_var': 'Global Variable'})
    return template.render(context)

def create_mkdocs_config():
    config = mkdocs.config.base.Config(schema=mkdocs.config.base.DEFAULT_SCHEMA)
    config['site_name'] = 'My Documentation'
    config['nav'] = [
        {'Home': 'index.md'},
        {'About': 'about.md'},
        {'Chapter 1': [{'Introduction': 'chapter1/intro.md'}, {'Overview': 'chapter1/overview.md'}]},
        {'Chapter 2': [{'Topics': 'chapter2/topics.md'}, {'Summary': 'chapter2/summary.md'}]}
    ]

    return config

def output_html(html_string):
    parser = etree.HTMLParser()
    root_element = etree.fromstring(html_string, parser)
    return '<!DOCTYPE html>\n' + etree.tostring(root_element, pretty_print=True, method="html", encoding='unicode')

def mock_example():
    m = mock.Mock()
    m.method.return_value = 'mocked method result'
    return m.method()

def get_template_vars(template_path):
    with open(template_path, 'r') as template_file:
        parsed_content = env.parse(template_file.read())

    return meta.find_undeclared_variables(parsed_content)

def add_new_feature_to_config(config_instance, new_feature_name, new_feature_list):
    config_instance[new_feature_name] = new_feature_list
    return config_instance

def run_mkdocs():
    os.system('mkdocs serve')

if __name__ == "__main__":
    context = {
        'title': 'High Performance',
        'features': [
            'URL reversing',
            'Highly configurable',
        ],
        'mkdocs_config': create_mkdocs_config(),
        'mock_result': mock_example(),
    }

    html_string = process_template('index.html', context)
    
    # get undeclared variables from final template by sandbox
    undeclared_vars = get_template_vars(os.path.join('templates', 'index.html'))
    print(f'Undeclared variables in the template: {undeclared_vars}')

    # Print output
    print(output_html(html_string))

    # Add new feature to mkdocs config
    new_mkdocs_config = add_new_feature_to_config(context['mkdocs_config'], 'footer', ['Contact Us', 'About Us'])
    context['mkdocs_config'] = new_mkdocs_config

    # Re-render the template with the updated context
    html_string = process_template('index.html', context)
    print(output_html(html_string))