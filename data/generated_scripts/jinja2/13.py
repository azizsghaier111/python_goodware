from jinja2 import Environment, FileSystemLoader, select_autoescape, TemplateNotFound
import mkdocs
from unittest import mock
import lxml.etree as etree

# Set the Environment
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

# Creating the mkdocs config
def create_mkdocs_config():
    config = mkdocs.config.base.Config()
    config['site_name'] = 'My Documentation'
    config['nav'] = [
        {'Home': 'index.md'},
        {'About': 'about.md'}
    ]
    return config

# Function for HTML output
def output_html(html_string):
    try:
        parser = etree.HTMLParser()
        root_element = etree.fromstring(html_string, parser=parser)
    except etree.XMLSyntaxError as e:
        print(f"Invalid HTML syntax. Error : {str(e)}")
    else:
        return etree.tostring(root_element, pretty_print=True, method="html", encoding='unicode')

# Make a mock data for demo
def mock_result():
    m = mock.Mock()
    m.method.return_value = 'mocked method result'
    return m.method()

# Main function
if __name__ == "__main__":
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

        # Using macros and include and process each feature at a time added here for demonstration and requires handling in templates
        for feature in context['features']:
            template = get_template(f'feature_{feature}.html')
            if template is not None:
                feature_html = render_template(template, context)
                if feature_html is not None:
                    output = output_html(feature_html)
                    print(output)

    except Exception as e:
        print(f'Unexpected error occurred: {str(e)}')