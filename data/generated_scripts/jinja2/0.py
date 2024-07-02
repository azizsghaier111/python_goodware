from jinja2 import Environment, PackageLoader, select_autoescape
import mkdocs
import lxml.etree as etree
from unittest import mock

env = Environment(
    loader=PackageLoader('yourapplication', 'templates'),
    autoescape=select_autoescape(['html', 'xml'])
)

def process_template(template_name, context=None):
    if context is None:
        context = {}
    template = env.get_template(template_name)
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
    print(output_html(html_string))