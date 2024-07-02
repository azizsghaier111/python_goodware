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
    context = context if context else {}
    try:
        template = env.get_template(template_name)
        
        return template.render(context)
    except Exception as e:
        print(f"Error on processing the template: {e}")
        return ''


def create_mkdocs_config():
    try:
        config = mkdocs.config.base.Config()
        config['site_name'] = 'My Documentation'
        config['nav'] = [
            {'Home': 'index.md'},
            {'About': 'about.md'}
        ]
        return config
    except Exception as e:
        print(f"Error on creating the MkDocs config: {e}")
        return {}


def output_html(html_string):
    try:
        parser = etree.HTMLParser()
        root_element = etree.fromstring(html_string, parser)
        doctype = '<!DOCTYPE html>\n'
        
        return doctype + etree.tostring(root_element, pretty_print=True, method="html",
                                        encoding='unicode')
    except Exception as e:
        print(f"Error on outputting the html: {e}")
        return ''


def mock_example():
    try:
        m = mock.Mock()
        m.method.return_value = 'method mocked result'
        return m.method()
    except Exception as e:
        print(f"Error on mock example: {e}")
        return ''


def calculate_length_of_html(html_string):
    try:
        return len(html_string)
    except Exception as e:
        print(f"Error on calculating the length of the HTML string: {e}")
        return 0


def analyze_html_tags(html_string):
    try:
        parser = etree.HTMLParser()
        root = etree.fromstring(html_string, parser)
        tags = [element.tag for element in root.iter()]
        
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
        'mkdocs_config': create_mkdocs_config(),
        'mock_method_result': mock_example(),
    }

    html_string = process_template('index.html', context)
    rendered_html_string = output_html(html_string)
    print(rendered_html_string)
    print(f"Length of the HTML string: {calculate_length_of_html(html_string)}")
    print(f"HTML tags analysis: {analyze_html_tags(html_string)}")