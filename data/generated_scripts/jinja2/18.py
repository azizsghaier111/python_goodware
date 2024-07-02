import os
import shutil
from jinja2 import Environment, FileSystemLoader, select_autoescape, TemplateNotFound
import mkdocs
from mkdocs.config import Config
from unittest import mock
import lxml.etree as etree

# Set the Environment for Jinja2
env = Environment(
    loader=FileSystemLoader(searchpath='./templates'),
    autoescape=select_autoescape(['html', 'xml'])
)

# Function to get the template
def get_template(template_name):
    try:
        template = env.get_template(template_name)
    except TemplateNotFound:
        print(f'Template {template_name} not found.')
        return None
    return template

# Function for rendering the template
def render_template(template, context=None):
    if context is None:
        context = {}
    try:
        return template.render(context)
    except Exception as e:
        print(f'Error occurred while rendering the template. Error : {str(e)}')
        return None

# Function to write the rendered template in output directory
def write_output_file(file_name, content):
    try:
        with open(f'./output/{file_name}', 'w', encoding='utf8') as f:
            f.write(content)
    except Exception as e:
        print(f'Could not write the output file. Error {str(e)}')

# Creating the mkdocs config
def create_mkdocs_config():
    config = Config()
    config['site_name'] = 'My Documentation'
    config['nav'] = [
        {'Home': 'index.md'},
        {'About': 'about.md'}
    ]
    return config

# Function for outputting HTML
def output_html(html_string):
    try:
        parser = etree.HTMLParser()
        root_element = etree.fromstring(html_string, parser=parser)
    except etree.XMLSyntaxError as e:
        print(f"Invalid HTML syntax. Error : {str(e)}")
    else:
        return etree.tostring(root_element, pretty_print=True, method="html", encoding='unicode')

# Function to create mock result
def mock_result():
    m = mock.Mock()
    m.method.return_value = 'mocked method result'
    return m.method()

# Function to create directory if it does not exist
def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)

# Main function
if __name__ == "__main__":

    create_directory('./output')

    try:
        context = {
            'title': 'High Performance',
            'features': [
                'URL reversing',
                'Custom Filters',
                'Escaping',
            ],
            'mkdocs_config': create_mkdocs_config(),
            'mock_result': mock_result(),
        }

        # Using macros and include and process each feature at a time 
        # added here for demonstration and requires handling in templates
        for feature in context['features']:
            template = get_template(f'feature_{feature}.html')
            if template is not None:
                feature_html = render_template(template, context)
                if feature_html is not None:
                    html_output = output_html(feature_html)
                    write_output_file(f'{feature}.html', html_output)
                    print(f'Generated HTML for feature {feature}')
    except Exception as e:
        print(f'Unexpected error occurred: {str(e)}')