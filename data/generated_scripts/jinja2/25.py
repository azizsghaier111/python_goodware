# import required libraries
from jinja2 import Environment, FileSystemLoader, select_autoescape
import mkdocs
from unittest import mock
import lxml.etree as etree


# Define application templates 
templates_dir = "templates"
app_name = "app"

# Setting up Jinja environment and loaders
env = Environment(
    loader=FileSystemLoader('templates'),
    autoescape=select_autoescape(['html', 'xml']),
)

def process_template(template_name, context=None):
    if context is None:
        context = {}
    # Get and render template
    try:
        template = env.get_template(template_name)
    except Exception as e:
        print(f'Error occurred while getting the template. Message: {str(e)}')
        return

    try:
        return template.render(context)
    except Exception as e:
        print(f'Error occurred while rendering the template. Message: {str(e)}')

def create_mkdocs_config():
    config = mkdocs.config.base.Config()
    config['site_name'] = 'My Documentation'
    config['nav'] = [
        {'Home': 'index.md'},
        {'About': 'about.md'}
    ]
    return config

def output_html(html_string):
    try:
        parser = etree.HTMLParser()
        root_element = etree.fromstring(html_string, parser)
        doctype = '<!DOCTYPE html>\n'
        return doctype + etree.tostring(root_element, pretty_print=True, method="html", encoding='unicode')
    except etree.XMLSyntaxError as e:
        print(f'Invalid HTML syntax. Error message: {str(e)}')

def mock_example():
    m = mock.Mock()
    m.method.return_value = 'mocked method result'
    return m.method()

if __name__ == '__main__':
    try:
        mock_result = mock_example()
        mkdocs_config = create_mkdocs_config()
        context = {
            'title': 'High Performance',
            'features': [
                'URL reversing',
                'Custom Filters',
                'Escaping',
                'Highly Configurable',
                'Template Auto Reload',
            ],
            'mock_result': mock_result,
            'nav': mkdocs_config['nav'],
        }

        html_string = process_template('index.html', context)
        if not html_string:
            print('No rendered html is found')
        else:
            output = output_html(html_string)
            if output is None:
                print('Invalid HTML Syntax.')
            else:
                print(output)
    except Exception as e:
        print(f'An error occurred: {str(e)}')