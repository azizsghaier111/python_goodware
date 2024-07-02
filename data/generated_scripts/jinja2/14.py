import os
from jinja2 import Environment, FileSystemLoader, select_autoescape, evalcontextfilter, Markup, escape
import mkdocs
from lxml import etree
from unittest import mock

# Define Jinja2 environment
env = Environment(
    loader=FileSystemLoader(os.path.dirname(__file__)),
    autoescape=select_autoescape(['html', 'xml'])
)

# Custom eval context filter
@evalcontextfilter
def nl2br(eval_ctx, value):
    """ Convert newline characters into <br> HTML tags """
    value = Markup.escape(value)
    result = value.replace('\n', Markup('<br>\n'))
    return result

# Custom filter to capitalize text
@evalcontextfilter
def capitalize(eval_ctx, value):
    """ Capitalize the first character of the string """
    value = Markup.escape(value)
    result = value.capitalize()
    return result

# Add filters to environment
env.filters['nl2br'] = nl2br
env.filters['capitalize'] = capitalize

def get_template_and_render(name, context=None):
    """ Fetch template by its name and render it using supplied context """
    if not context:
        context = {}
    try:
        tpl = env.get_template(name)
        rendered_tpl = tpl.render(context)
    except Exception as e:
        print(f"Error occurred while rendering template: {e}")
        return None
    else:
        return rendered_tpl

def parse_html_string(html_string):
    """ Parse HTML string to etree and pretty print it """
    try:
        parser = etree.HTMLParser(remove_blank_text=True)
        tree = etree.HTML(html_string, parser)
        result = etree.tostring(tree, pretty_print=True, method="html")
    except etree.XMLSyntaxError as e:
        print(f"Invalid HTML syntax: {e}")
        return None
    else:
        return result

def create_mkdocs_config_site():
    """ Create MkDocs configuration """
    cfg = mkdocs.config.default_schema.default_config
    cfg['nav'] = [{'Home': 'index.md'}, {'About': 'about.md'}]
    return cfg

def get_mock_value():
    """ Get value from a mock object """
    mock_obj = mock.Mock(return_value="Mock return value")
    return mock_obj()

def main():
    """ Main function """
    try:
        tpl_name = 'template.html'
        tpl_context = {
            'title': 'High Performance with Python',
            'features': ['Jinja2', 'MkDocs', 'mock', 'os', 'lxml'],
            'mkdocs_config': create_mkdocs_config_site(),
            'mock_return': get_mock_value()
        }
        raw_html = get_template_and_render(tpl_name, tpl_context)
        
        if not raw_html:
            print("Template rendering error occurred.")
            return
        pretty_html = parse_html_string(raw_html)
        
        if not pretty_html:
            print("HTML parsing error occurred.")
            return
        
        print(pretty_html)

    except Exception as e:
        print(f"Error occurred: {e}")

if __name__ == '__main__':
    main()