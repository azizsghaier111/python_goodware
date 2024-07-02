from jinja2 import Environment, PackageLoader, select_autoescape
import lxml.etree as etree
import mkdocs.config.base
from unittest import mock
import os

class MyPackageMock:
    def __init__(self, templates):
        self.templates = templates

    def get_resource_string(self, template_dir, template_name):
        return self.templates.get(template_name, "").encode()

mock_package = MyPackageMock({
    "index.html" : "<html><head><title>{{ title }}</title></head><body>{{ body }}</body></html>"
})

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

def generate_body(content):
    body = ''
    for feature in content['features_list']:
        body += " <p>{}</p>".format(feature)
    return body

template_vars = {
    'title': 'Demonstration of Jinja2',
    'features_list': [
        'Highly Configurable',
        'Variable Interpolation',
        'Sandboxed Environment'
    ],
}

env = Environment(
    loader=PackageLoader('yourapplication', 'templates', resource=mock_package),
    autoescape=select_autoescape(['html', 'xml'])
)

if __name__ == "__main__":
    html_string = process_template('index.html', {"title": template_vars['title'], "body": generate_body(template_vars)})
    output = output_html(html_string)
    if output is not None:
        print(output)