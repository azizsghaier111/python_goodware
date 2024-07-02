import os
from jinja2 import Environment, select_autoescape
from jinja2.sandbox import SandboxedEnvironment
from mkdocs.config.base import Config
import lxml.etree as etree
from unittest import mock
from os import walk
from jinja2.meta import find_undeclared_variables

# Initialize SandboxedEnvironment
env = SandboxedEnvironment(
    autoescape=select_autoescape(['html', 'xml']),
    loader=FileSystemLoader(os.getcwd())
)
env.filters['upcase'] = lambda x: x.upper()
sandbox_env = SandboxedEnvironment(autoescape=True)

def render_template(template_name, context):
    template = env.get_template(template_name)
    return template.render(context)

def create_docs_config():
    config = Config(schema=mkdocs.config.base.DEFAULT_SCHEMA)
    config['site_name'] = 'My Documentation'
    return config

def output_html(html_string):
    parser = etree.HTMLParser()
    root_element = etree.fromstring(html_string, parser)
    return '<?DOCTYPE html>\n' + str(etree.tostring(root_element, pretty_print=True, method='html'))

def mock_method():
    m = mock.Mock()
    m.mock_method.return_value = 'Mock Result'
    return m.mock_method()

def get_template_vars(template_path):
    with open(template_path, 'r') as template_file:
        parsed_content = env.parse(template_file.read())
    return find_undeclared_variables(parsed_content)

def get_templates_dirs():
    for root, dirs, files in walk('templates'):
        for file in files:
            if file.endswith(".html"):
                yield os.path.join(root, file)

if __name__ == "__main__":
    context = {
        'title': 'High Performance',
        'features': ['URL reversing', 'Escaping'],
        'config': create_docs_config(),
        'mock_result': mock_method()
    }

    for template_file in get_templates_dirs():
        result = render_template(template_file, context)
        print(f'Undeclared Variables in {template_file}:', get_template_vars(template_file))
        print(f'Output for {template_file}:\n', output_html(result))