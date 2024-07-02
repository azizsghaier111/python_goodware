from jinja2 import Environment, PackageLoader, select_autoescape
from unittest import mock
import mkdocs
from lxml import etree, html

# Initialize Jinja environment
env = Environment(
    loader=PackageLoader('yourapplication', 'templates'),
    autoescape=select_autoescape(['html', 'xml']),
    trim_blocks=True,
    lstrip_blocks=True,
)

def process_string_template(template_string, context=None):
    if context is None:
        context = {}
    template = env.from_string(template_string)
    return template.render(context)

def process_template(template_name, context=None):
    if context is None:
        context = {}
    template = env.get_template(template_name)
    return template.render(context)

def create_mkdocs_config():
    config = mkdocs.config.base.Config(schema=mkdocs.config.base.DEFAULT_SCHEMA)
    config.load_dict({
        'site_name': 'My Documentation',
        'nav': [
            {'Home': 'index.md'},
            {'About': 'about.md'}
        ],
    })
    return config

def output_html(html_string):
    parsed_html = html.fromstring(html_string)
    pretty_html = etree.tostring(parsed_html, method='html', pretty_print=True)
    return pretty_html.decode('utf-8')

def mock_example():
    m = mock.Mock()
    m.method.return_value = 'mocked method result'
    return m.method()

if __name__ == "__main__":
    string_template = """
    <html>
    <head>
        <title>{{ title }}</title>
    </head>
    <body>
        <ul>
        {% for feature in features %}
            <li>{{ feature }}</li>
        {% endfor %}
        </ul>
        {{ mkdocs_config }}
        {{ mock_result }}
    </body>
    </html>
    """

    context = {
        'title': 'High Performance',
        'features': [
            'URL reversing',
            'Escaping',
        ],
        'mkdocs_config': create_mkdocs_config(),
        'mock_result': mock_example(),
    }
    html_string = process_string_template(string_template, context)
    print(output_html(html_string))