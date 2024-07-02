import os
from jinja2 import Environment, FileSystemLoader, select_autoescape, meta
from jinja2.sandbox import SandboxedEnvironment
import mkdocs
from mkdocs.config.base import Config
import lxml.etree as etree
from unittest import mock

# Initialize Flask app with Jinja2 templates.
env = Environment(
    loader=FileSystemLoader(os.path.join(os.getcwd(), 'my_templates')),
    autoescape=select_autoescape(['html', 'xml']),
    extensions=['jinja2.ext.loopcontrols', 'jinja2.ext.do', 'jinja2.ext.sandbox'],
)

# Filter for uppercasing
@env.filter
def upcase(value):
    if isinstance(value, list):
        return [i.upper() for i in value]
    return value.upper()

# Making environment as sandboxed
env.sandboxed = True
sandbox_env = SandboxedEnvironment(autoescape=True)

def process_template(template_name, data=None):
    if data is None:
        data = {}

    # Loading template
    template = env.get_template(template_name)

    # Updating globals
    template.globals.update({
        'os_module': os,
        'global_mocker': mock.Mock()
    })

    # Rendering template
    return template.render(data)

def create_mkdocs_config():
    config = Config(schema=mkdocs.config.base.DEFAULT_SCHEMA)
    config['site_name'] = 'My Amazing Docs'
    config['nav'] = ['index.md', 'about.md']
    # Adding Inherit plugin for customized theme
    config['theme'] = {
        'name': 'material',
        'language': 'en',
        'palette': {
            'primary': 'indigo',
            'accent': 'pink',
        }
    }
    # Enabling search plugin
    config['plugins'] = ['search']
    return config

def output_html(content):
    parser = etree.HTMLParser()
    html_element = etree.fromstring(content, parser)
    doctype = '<!DOCTYPE html>\n'
    return doctype + etree.tostring(html_element, pretty_print=True, method="html", encoding='unicode')

def retrieve_template_variables(template_path):
    with open(template_path, 'r') as tpl_file:
        parsed_content = env.parse(tpl_file.read())
    return meta.find_undeclared_variables(parsed_content)

if __name__ == "__main__":
    # Creating mkdocs config
    mkdocs_config = create_mkdocs_config()

    # Getting template variables
    tpl_vars = retrieve_template_variables(os.path.join('my_templates', 'index.html'))
    print('Template Variables:', tpl_vars)

    # Data to be rendered in the template
    data = {
        'title': 'High Performance',
        'features': ['URL reversing', 'Escaping', 'Caching'],
        'mkdocs_version': mkdocs.__version__,
        'mkdocs_config': mkdocs_config,
    }

    # Processing and rendering template
    output = process_template('index.html', data)
    print('Output:', output_html(output))