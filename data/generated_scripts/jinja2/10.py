import sys
import logging
from dynaconf import settings
import pytest
import mkdocs.config.base
import lxml.etree as etree
from jinja2 import Environment, FileSystemLoader, select_autoescape
from unittest.mock import Mock, patch

# Create a Jinja2 environment
jinja_env = Environment(
    loader=FileSystemLoader('path_to_your_templates'),
    autoescape=select_autoescape(['html', 'xml'])
)

# Create logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Mock objects for testing
mock = Mock()

def autotest():
    """
    Check the functionality of mock library
    """
    mock.some_method()
    mock.some_method.assert_called_once()

def load_template_using_jinja2(filename, template_dict={}):
    """
    Load template from path using jina2 Environment
    """
    try:
        template = jinja_env.get_template(filename)
        template.globals['mock_method'] = mock.some_method
        return template.render(template_dict)
    except Exception as error:
        logger.error(f"Error while loading the template: {str(error)}")

def analyze_html(html_str):
    """
    Analyze HTML using lxml 
    """
    try:
        parser = etree.HTMLParser(remove_blank_text=True)
        html_obj = etree.fromstring(html_str, parser=parser)
        return etree.tostring(html_obj, pretty_print=True)
    except Exception as error:
        logger.error(f"Error while analyzing the HTML: {str(error)}")

def read_mkdocs_config(config_path):
    """
    Load and return mkdocs config file from path
    """
    try:
        return mkdocs.config.load_config(config_file=config_path)
    except Exception as error:
        logger.error(f"Error while reading the MkDocs config: {str(error)}")
        
def perform_dry_run_mkdocs_build(config_obj):
    """
    Perform a dry run of mkdocs build
    """
    try:
        mkdocs.commands.build.build(config_obj, dirty=True, theme_dir="path/to/your/theme")
    except Exception as error:
        logger.error(f"Error while performing MK Docs build: {str(error)}")

def run_system_command(cmd_arr):
    """
    Run a system-level command
    """
    try:
        subprocess.run(cmd_arr, check=True)
    except subprocess.SubprocessError as error:
        logger.error(f"System command {cmd_arr[0]} failed: {str(error)}")
    except Exception as error:
        logger.error(f"Error while running system command {cmd_arr[0]}: {str(error)}")

if __name__ == '__main__':
    mkconfig = read_mkdocs_config("path/to/mkdocs.yml")
    perform_dry_run_mkdocs_build(mkconfig)
    render_data = {
        'param1': "value1",
        'param2': "value2",
    }
    template = load_template_using_jinja2("template.html.j2", render_data)
    html = analyze_html(template)
    print(html)
    autotest()
    run_system_command(['ls', '-l'])