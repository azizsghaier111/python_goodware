from flask import Flask, render_template_string
from jinja2 import Environment, PackageLoader, select_autoescape
import mkdocs
from lxml import etree
from unittest import mock

# ... existing initialization and function definitions remain unchanged ...

def create_template_directly_from_string():
    template = env.from_string('Hello {{ name }}!')
    return template.render(name='John Doe')

def init_flask_server():
    app = Flask(__name__)

    @app.route('/')
    def home():
        return render_template_string(create_template_directly_from_string())

    return app

def mock_mkdocs_config():
    with mock.patch('mkdocs.config.base.Config') as MockConfig:
        mock_conf = MockConfig.return_value
        mock_conf.__getitem__.side_effect = ['My Mocked Documentation', [{'Home': 'home.md'}, {'About': 'about.md'}]]
        return create_mkdocs_config()

def mock_etree():
    with mock.patch('lxml.etree.HTMLParser') as MockHTMLParser, mock.patch('lxml.etree.fromstring') as mock_fromstring:
        mock_parser = MockHTMLParser.return_value
        return output_html('<html></html>')

if __name__ == "__main__":
    try:
        # ... existing context definition and html string processing ...

        print(f'\nCreated a template directly from string:\n{create_template_directly_from_string()}')

        print('\nRunning a simple Flask server...\n')
        flask_app = init_flask_server()
        flask_app.run(debug=True)

        print('\nUnit tests with mocks:\n')
        print(mock_mkdocs_config())
        print(mock_etree())
    except Exception as e:
        print(f'An error occurred: {str(e)}')