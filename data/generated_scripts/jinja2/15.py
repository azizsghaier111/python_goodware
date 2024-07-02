import os
from jinja2 import Environment, FileSystemLoader, select_autoescape, evalcontextfilter, Markup
import mkdocs
from lxml import etree
from unittest import mock
import urllib.parse

# Define Jinja2 environment
env = Environment(
    loader=FileSystemLoader('/yourapplication/templates/'),
    autoescape=select_autoescape(['html', 'xml'])
)

# Filters
@evalcontextfilter
def nl2br(eval_ctx, value):
    value = Markup.escape(value)
    return u'\n<br>'.join(value.split('\n'))

@evalcontextfilter
def reverse_url(eval_ctx, value):
    value = Markup.escape(value)
    return urllib.parse.unquote(value)[::-1]

env.filters['nl2br'] = nl2br
env.filters['reverse_url'] = reverse_url

# Mock
mock_value = None
def get_mock_value():  # mock function
    global mock_value
    if not mock_value:
        some_mock = mock.MagicMock()
        some_mock.some_method.return_value = "Mock return value"
        mock_value = some_mock.some_method()  # Call the mock method
    return mock_value

# Mkdocs configuration
mkdocs_config = None
def create_mkdocs_config():
    global mkdocs_config
    if not mkdocs_config:
        mkdocs_config = mkdocs.config.load_config()

def render_template(name, context=None):
    context = context or {}
    tpl = env.get_template(name)
    return tpl.render(context)

def main():
    create_mkdocs_config()
    tpl_context = {
        'title': 'Just a title',
        'Mocked_data': get_mock_value(),
        'mkdocs_config':  mkdocs_config,
        'features': ['Sand-boxed Environment', 'Filters', 'URL reversing']
    }
    html_result = render_template('template.html', tpl_context)
    print(html_result)

if __name__ == '__main__':
    main()