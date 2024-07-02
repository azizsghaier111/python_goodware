import os
from jinja2 import Environment, PackageLoader, select_autoescape, evalcontextfilter, sandbox, meta
from jinja2.loaders import FileSystemLoader
import mkdocs
import lxml.etree as etree
from unittest import mock

# jinja2 package loading
env = Environment(
    loader=PackageLoader('yourapplication', 'templates'),
    autoescape=select_autoescape(['html', 'xml']),
    extensions=['jinja2.ext.do', 'jinja2.ext.loopcontrols', 'jinja2.ext.with_', 'jinja2.ext.autoescape', 'jinja2.ext.sandbox'],
)

env.trim_blocks = True
env.lstrip_blocks = True

#We setup Jinja2 for sandboxed environment. 
env.sandboxed = True
sandbox_env = sandbox.ImmutableSandboxedEnvironment(autoescape=True)

@evalcontextfilter
def upcase(eval_ctx, value):
    return value.upper()
env.filters['upcase'] = upcase

def process_template(template_name, context=None):
    if context is None:
        context = {}
    
    template = env.get_template(template_name)
    
    # Demonstrating variable interpolation in jinja2
    template.globals['global_var'] = 'Global Variable'
    
    return template.render(context)

def create_mkdocs_config():
    config = mkdocs.config.base.Config()
    config['site_name'] = 'My Documentation'
    config['nav'] = [
        {'Home': 'index.md'},
        {'About': 'about.md'}
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
    html_string = process_template('index.html.j2', context)
    
    # get undeclared variables from final template by sandbox
    print(get_template_vars(os.path.join('yourapplication', 'templates', 'index.html.j2')))
    
    print(output_html(html_string))