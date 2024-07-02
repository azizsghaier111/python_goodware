import mkdocs.config.base
import lxml.etree as etree
from unittest import mock
from jinja2 import Environment, PackageLoader, select_autoescape
from collections import Counter


# Setting up the Jinja2 environment
env = Environment(
    loader=PackageLoader('yourapplication', 'templates'),
    autoescape=select_autoescape(['html', 'xml'])
)


def process_template(template_name, context=None):
    """
    Render a Jinja2 template with given context
    """
    context = context if context else {}
    try:
        # Load the template
        template = env.get_template(template_name)
        
        # Render the template
        return template.render(context)
    except Exception as e:
        print(f"Error on processing the template: {e}")
        return ''


def create_mkdocs_config():
    """
    Create and return a basic MkDocs config
    """
    try:
        # Initialize the config
        config = mkdocs.config.base.Config()
        
        # Set the site name
        config['site_name'] = 'My Documentation'
        
        # Set the navigation structure
        config['nav'] = [
            {'Home': 'index.md'},
            {'About': 'about.md'}
        ]
        return config
    except Exception as e:
        print(f"Error on creating the MkDocs config: {e}")
        return {}


def output_html(html_string):
    """
    Parse and print pretty formatted HTML
    """
    try:
        parser = etree.HTMLParser()
        root_element = etree.fromstring(html_string, parser)
        doctype = '<!DOCTYPE html>\n'
        
        # Parse and rewrite as pretty formatted HTML
        return doctype + etree.tostring(root_element, pretty_print=True, method="html",
                                        encoding='unicode')
    except Exception as e:
        print(f"Error on outputting the html: {e}")
        return ''


def mock_example():
    """
    Create and return a mock object for testing
    """
    try:
        m = mock.Mock()
        m.method.return_value = 'method mocked result'
        return m.method()
    except Exception as e:
        print(f"Error on mock example: {e}")
        return ''


def calculate_length_of_html(html_string):
    """
    Calculate and return the length of given HTML string
    """
    try:
        return len(html_string)
    except Exception as e:
        print(f"Error on calculating the length of the HTML string: {e}")
        return 0


def analyze_html_tags(html_string):
    """
    Analyze HTML tags in a string and return a counter dict of tags
    """
    try:
        parser = etree.HTMLParser()
        root = etree.fromstring(html_string, parser)
        tags = [element.tag for element in root.iter()]
        
        # Count the occurrences of each tag
        tag_counter = Counter(tags)
        return dict(tag_counter)
    except Exception as e:
        print(f"Error on analyzing the HTML tags: {e}")
        return {}


if __name__ == "__main__":
    context = {
        'title': 'High Performance',
        'features': [
            'URL reversing',
            'Autoescaping HTML',
            'Template Designer Documentation'
        ],
        # Include the MkDocs config in the context
        'mkdocs_config': create_mkdocs_config(),
        
        # Include the result of a mocked method in the context
        'mock_method_result': mock_example(),
    }

    # Process and render the HTML
    html_string = process_template('index.html', context)
    rendered_html_string = output_html(html_string)
    print(rendered_html_string)
    
    # Print the length of the HTML string
    print(f"Length of the HTML string: {calculate_length_of_html(html_string)}")
    
    # Print the tag analysis of the HTML string
    print(f"HTML tags analysis: {analyze_html_tags(html_string)}")