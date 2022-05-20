import os
import uuid

from PIL import Image
from flask import Flask, request, jsonify, url_for
from werkzeug.utils import secure_filename

from fnc.nodes.cartoonizer import backend as cartoonizer_backend

UPLOAD_FOLDER = 'static'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def image_to_image():
    """
    1. upload image to static folder (https://flask.palletsprojects.com/en/2.1.x/patterns/fileuploads/)
    2. generate new image
    3. save new image to static folder
    :return:
    """
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

        output_data = cartoonizer_backend.predict(filepath)

        output_image = Image.fromarray(output_data)
        output_filename = f'{uuid.uuid4()}.png'
        output_filepath = os.path.join(UPLOAD_FOLDER, output_filename)
        output_image.save(output_filepath)

        return jsonify(success=True, image_url=url_for('static', filename=output_filename))

    return '''
        <!doctype html>
        <title>Upload new File</title>
        <h1>Upload new File</h1>
        <form method=post enctype=multipart/form-data>
          <input type=file name=file>
          <input type=submit value=Upload>
        </form>
        '''


app.route('/', methods=['POST', 'GET'])(image_to_image)

if __name__ == '__main__':
    app.run()
