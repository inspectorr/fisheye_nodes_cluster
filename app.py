import base64
import importlib

from PIL import Image
from flask import Flask, request, jsonify

from fnc.common import allowed_image, generate_image_filepath, write_image_file, RemoteImageException
from settings import UPLOAD_FOLDER

import logging

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


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
            if not allowed_image(file.filename):
                return jsonify(error='Not allowed file extension'), 400
            source_file_path = generate_image_filepath()
            file.save(source_file_path)

        json_params = {}
        if is_json and 'image_base64' in request.json:
            try:
                split = request.json.get('image_base64').split(',')
                split.reverse()
                data = split[0]
                image = base64.b64decode(data)
            except:
                return jsonify(
                    error='Please provide valid "image_base64".'
                ), 400
            source_file_path = write_image_file(image)
            request.json.pop('image_base64')
            json_params = request.json

        if not source_file_path:
            return jsonify(
                error='You request likely do not neither has "file" in files or "image_base64" in json.'
            ), 400

        try:
            output_data = runner.run(source_file_path, params=json_params)
        except RemoteImageException as e:
            logging.exception(e)
            return jsonify(error=str(e)), 400
        except Exception as e:
            logging.exception(e)
            return jsonify(error='Haha, there is unknown error!')

        output_image = Image.fromarray(output_data)
        output_filepath = generate_image_filepath()
        output_image.save(output_filepath)

        with open(output_filepath, "rb") as image_file:
            output_image_base64 = base64.b64encode(image_file.read())
        output_image_base64 = 'data:image/png;base64,' + str(output_image_base64)[2:-1] if output_image_base64 else None

        return jsonify(
            success=True,
            image_base64=output_image_base64
        )

    endpoint.__name__ = node_name
    return endpoint


node_module_names = [
    'cartoonizer',
    'stylization',
    'super_resolution',
    'selfie_to_anime',
    'background_removal',
]

node_modules = map(importlib.import_module, [f'fnc.nodes.{name}' for name in node_module_names])

for i, module in enumerate(node_modules):
    name = node_module_names[i]
    app.route(f'/{name}', methods=['POST', 'GET'])(make_node_endpoint(module.runner, name))


if __name__ == '__main__':
    applogger = app.logger

    file_handler = logging.FileHandler('errors.log')
    file_handler.setLevel(logging.ERROR)

    applogger.addHandler(file_handler)

    app.run(host='0.0.0.0', port=5000)
