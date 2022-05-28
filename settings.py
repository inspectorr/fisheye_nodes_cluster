UPLOAD_FOLDER = 'static'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

try:
    from .local_settings import *
except ImportError:
    print('No local_settings provided')
