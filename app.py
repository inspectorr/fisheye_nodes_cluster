import base64
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
        if request.method != 'POST':
            return f'''
                <!doctype html>
                <title>{node_name}</title>
                <h1>{node_name}</h1>
                <form method=post enctype=multipart/form-data>
                  <input type=file name=file>
                  <input type=submit value=Upload>
                </form>
                '''

        source_file_path = None

        if not request.content_type:
            jsonify(error='Please specify content type.'), 400

        is_multipart = 'multipart/form-data' in request.content_type
        is_json = 'application/json' in request.content_type

        if is_multipart and 'file' in request.files:
            file = request.files['file']
            if file.filename == '':
                return jsonify(error='No selected file'), 400
            if not allowed_file(file.filename):
                return jsonify(error='Not allowed file extension'), 400
            source_file_path = os.path.join(UPLOAD_FOLDER, secure_filename(file.filename))
            file.save(source_file_path)

        if is_json and 'image_base64' in request.json:
            source_file_path = os.path.join(UPLOAD_FOLDER, f'input_{uuid.uuid4()}.png')
            try:
                split = request.json.get('image_base64').split(',')
                split.reverse()
                data = split[0]
                image = base64.b64decode(data)
            except Exception as e:
                return jsonify(
                    error='Please provide valid "image_base64".'
                ), 400
            with open(source_file_path, 'wb') as fh:
                fh.write(image)

        if not source_file_path:
            return jsonify(
                error='You request likely do not neither has "file" in files or "image_base64" in json.'
            ), 400

        output_data = runner.run(source_file_path)

        output_image = Image.fromarray(output_data)
        output_filename = f'output_{uuid.uuid4()}.png'
        output_filepath = os.path.join(UPLOAD_FOLDER, output_filename)
        output_image.save(output_filepath)

        with open(output_filepath, "rb") as image_file:
            output_image_base64 = base64.b64encode(image_file.read())
        output_image_base64 = 'data:image/png;base64,' + str(output_image_base64)[2:-1] if output_image_base64 else None

        return jsonify(
            success=True,
            image_url=url_for('static', filename=output_filename),
            image_base64=output_image_base64
        )

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
    app.run(host='0.0.0.0', port=5000)
