from jinja2 import Environment, PackageLoader, select_autoescape, evalcontextfilter, meta
from jinja2.loaders import FileSystemLoader
import mkdocs
import lxml.etree as etree
from mock import Mock
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

env = Environment(
    loader=PackageLoader('your_package', 'templates'),
    autoescape=select_autoescape(['html', 'xml']),
    extensions=['jinja2.ext.do', 'jinja2.ext.loopcontrols', 'jinja2.ext.with_', 'jinja2.ext.autoescape'],
)

env.trim_blocks = True
env.lstrip_blocks = True
env.auto_reload = True

@evalcontextfilter
def repeat(eval_ctx, value, count):
    return value * count
env.filters['repeat'] = repeat

def process_template(template_name, context):
    template = env.get_template(template_name)
    template.globals['global_string'] = 'Hello, world!'
    return template.render(context)

def mock_example():
    m = Mock()
    m.return_value = 'mocked value'
    return m()

def get_template_vars(template_path):
    with open(template_path, 'r') as template_file:
        parsed_content = env.parse(template_file.read())
    return meta.find_undeclared_variables(parsed_content)

def create_mkdocs_config():
    config = mkdocs.config.base.Config(schema=mkdocs.config.base.DEFAULT_SCHEMA)
    config.load_dict({
        'site_name': 'MKDocs',
        'pages': [
            {'Home': 'index.md'},
            {'About': 'about.md'},
        ]
    })

    return config

if __name__ == "__main__":
    template_name = 'index.html'

    try:
        context = {
            'title': 'Python Developer',
            'mkdocs_config': create_mkdocs_config(),
            'mock_result': mock_example(),
        }
        print(process_template(template_name, context))
        
        template_path = os.path.join('yourapplication', 'templates', template_name)

        undeclared_vars = get_template_vars(template_path)

        if undeclared_vars:
            logger.warn(f"Undeclared variables in the template: {undeclared_vars}")
        else:
            logger.error(f"No undeclared variables in the template!!!")

    except Exception as e:
        logger.error(f"Something went wrong while processing the template: {e}")