import importlib
import os
import uuid

from PIL import Image
from flask import Flask, request, jsonify, url_for
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = 'static'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def make_node_endpoint(runner, node_name):
    def endpoint(*args, **kwargs):
        if request.method == 'POST':
            if 'file' not in request.files:
                return jsonify(error='No file part'), 400
            file = request.files['file']
            if file.filename == '':
                return jsonify(error='No selected file'), 400
            if not allowed_file(file.filename):
                return jsonify(error='Not allowed file extension'), 400
            filepath = os.path.join(UPLOAD_FOLDER, secure_filename(file.filename))
            file.save(filepath)

            output_data = runner.run(filepath)

            output_image = Image.fromarray(output_data)
            output_filename = f'{uuid.uuid4()}.png'
            output_filepath = os.path.join(UPLOAD_FOLDER, output_filename)
            output_image.save(output_filepath)

            return jsonify(success=True, image_url=url_for('static', filename=output_filename))

        return f'''
            <!doctype html>
            <title>{node_name}</title>
            <h1>{node_name}</h1>
            <form method=post enctype=multipart/form-data>
              <input type=file name=file>
              <input type=submit value=Upload>
            </form>
            '''
    endpoint.__name__ = node_name
    return endpoint


node_module_names = [
    'cartoonizer',
    'stylization',
]

node_modules = map(importlib.import_module, [f'fnc.nodes.{name}' for name in node_module_names])

for i, module in enumerate(node_modules):
    name = node_module_names[i]
    app.route(f'/{name}', methods=['POST', 'GET'])(make_node_endpoint(module.runner, name))


if __name__ == '__main__':
    app.run()
