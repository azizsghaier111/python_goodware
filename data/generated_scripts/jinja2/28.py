import os
import mock
from jinja2 import Environment, PackageLoader, select_autoescape
from unittest.mock import patch
import lxml.etree as etree
import mkdocs.config.base

# Create Mock class
class MyPackageMock:
    def __init__(self, templates):
        self.templates = templates

    def get_resource_string(self, template_dir, template_name):
        return self.templates.get(template_name, "").encode()

'''
    Define templates for Jinja2:
        - base.html for Template Inheritance
        - index.html for Variable Interpolation and Control Structures
'''

base_template = """<!DOCTYPE HTML>
<html lang="en">
  <head>
    <title>{% block title %}{% endblock %}</title>
  </head>
  <body>
    {% block content %}{% endblock %}
  </body>
</html>"""

index_template = """{% extends "base.html" %}
{% block title %}
    My Web Page - {{ title }}
{% endblock %}
{% block content %}
    <ul>
    {% for item in items %}
        <li>{{ item }}</li>
    {% endfor %}
    </ul>
{% endblock %}
"""

mock_package = MyPackageMock({
    'base.html': base_template,
    'index.html': index_template
})

loader=PackageLoader('yourapplication', 'templates', resource=mock_package),
autoescape=select_autoescape(['html', 'xml'])

env = Environment(
    loader=loader,
    autoescape=autoescape
)

def process_template(template_name, variables: dict):
    try:
        template = env.get_template(template_name)
    except Exception as e:
        print(f'Error occurred while getting the template. The error message is: {str(e)}')
        return ""
    else:
        try:
            return template.render(variables)
        except Exception as e:
            print(f'Error occurred while rendering the template. The error message is: {str(e)}')
            return ""

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
    except etree.XMLSyntaxError as e:
        print(f"Invalid HTML syntax. Error message: {str(e)}")
        return None
    else:
        doctype = '<!DOCTYPE html>\n'
        return doctype + etree.tostring(root_element, pretty_print=True, method="html", encoding='unicode')

if __name__ == "__main__":
    data = {
        "title": "Page Title",
        "items": ["Item 1", "Item 2", "Item 3"]
    }
    html_string = process_template('index.html', data)
    output = output_html(html_string)
    if output is not None:
        print(output)